# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA Graph runner for LingBot-VLA 2.0 flow-matching action denoising loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .modeling_lingbot_vla_v2 import LingBotVLAV2ForActionPrediction


class LingBotActionLoopCUDAGraph:
    """Captures and replays the entire 10-step Euler sampling loop of LingBot-VLA 2.0.

    Eliminates thousands of kernel launch overheads while maintaining bitwise / exact
    numerical consistency with the eager path.
    """

    def __init__(
        self,
        model: LingBotVLAV2ForActionPrediction,
        batch_size: int = 1,
        num_steps: int = 10,
        device: torch.device | str = "cuda",
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.is_captured = False

        # Static buffers for graph inputs & outputs
        cfg = model.config
        state_dim = cfg.input_features.get("observation.state", {}).get("shape", [cfg.max_state_dim])[-1]
        action_shape = (batch_size, cfg.chunk_size, cfg.max_action_dim)

        self.static_state: torch.Tensor | None = None
        self.static_noise = torch.zeros(action_shape, device=self.device, dtype=torch.float32)
        self.static_x_t = torch.zeros(action_shape, device=self.device, dtype=torch.float32)
        self.static_out = torch.zeros(action_shape, device=self.device, dtype=torch.float32)

        # Buffers for prefix KV cache and positions
        self.static_kv: dict[int, dict[str, torch.Tensor]] = {}
        self.static_masks: torch.Tensor | None = None
        self.static_positions: torch.Tensor | None = None
        self.static_cache: dict[str, Any] = {}

    def _allocate_static_buffers(
        self,
        masks: torch.Tensor,
        positions: torch.Tensor,
        kv: dict[int, dict[str, torch.Tensor]],
    ) -> None:
        self.static_masks = masks.clone()
        self.static_positions = positions.clone()
        self.static_kv = {}
        for layer_idx, kv_dict in kv.items():
            self.static_kv[layer_idx] = {
                "key_states": kv_dict["key_states"].clone(),
                "value_states": kv_dict["value_states"].clone(),
            }

    def capture(
        self,
        state: torch.Tensor,
        masks: torch.Tensor,
        kv: dict[int, dict[str, torch.Tensor]],
        positions: torch.Tensor,
        num_warmup: int = 3,
    ) -> None:
        """Warm up and capture the entire multi-step action denoising loop."""
        self._allocate_static_buffers(masks, positions, kv)
        self.static_state = torch.zeros_like(state)
        self.static_state.copy_(state)

        # Setup warm up stream
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))

        dt = torch.tensor(-1.0 / self.num_steps, device=self.device, dtype=torch.float32)
        time_init = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        def _forward_loop(state_buf, noise_buf):
            x_t = noise_buf.clone()
            t = time_init.clone()
            cache = {}
            for _ in range(self.num_steps):
                v_t = self.model.predict_velocity(
                    state=state_buf,
                    prefix_pad_masks=self.static_masks,
                    past_key_values=self.static_kv,
                    x_t=x_t,
                    timestep=t.expand(self.batch_size),
                    prefix_position_ids=self.static_positions,
                    _denoise_cache=cache,
                )
                x_t = x_t + dt * v_t
                t = t + dt
            return x_t

        with torch.cuda.stream(stream), torch.inference_mode():
            # Warm up execution before capture
            for _ in range(num_warmup):
                self.static_noise.normal_()
                _ = _forward_loop(self.static_state, self.static_noise)

            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize()

            # Begin graph capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                out = _forward_loop(self.static_state, self.static_noise)
                self.static_out.copy_(out)

        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize()
        self.is_captured = True

    def replay(
        self,
        state: torch.Tensor,
        masks: torch.Tensor,
        kv: dict[int, dict[str, torch.Tensor]],
        positions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Copy fresh request inputs to graph buffers and replay the captured loop."""
        if not self.is_captured or self.graph is None:
            raise RuntimeError("LingBotActionLoopCUDAGraph must be captured before replay.")

        # Update dynamic state, noise, and prefix KV cache into static graph buffers
        self.static_state.copy_(state)
        self.static_noise.copy_(noise)
        self.static_masks.copy_(masks)
        self.static_positions.copy_(positions)

        for layer_idx, kv_dict in kv.items():
            self.static_kv[layer_idx]["key_states"].copy_(kv_dict["key_states"])
            self.static_kv[layer_idx]["value_states"].copy_(kv_dict["value_states"])

        # Replay the entire graph
        self.graph.replay()
        return self.static_out.clone()
