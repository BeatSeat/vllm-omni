# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS DiT wrapper for LLM_GENERATION execution type.

Wraps the existing GLMTTSDiT flow-matching model + vocoder as a
LLM_GENERATION-compatible module so that OmniGenerationScheduler and
gpu_model_runner can drive it.  This enables async_chunk streaming from
the AR stage via SharedMemoryConnector / chunk_transfer_adapter.

Reference: CosyVoice3 code2wav branch in cosyvoice3.py:308-795.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.glm_tts.vocoder import (
    load_vocoder,
    mel_to_audio,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def make_pad_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Create a boolean padding mask where padded positions are True."""
    if max_len is None:
        max_len = int(lengths.max().item())
    batch_size = lengths.shape[0]
    seq_range = torch.arange(0, max_len, device=lengths.device)
    return seq_range.unsqueeze(0).expand(batch_size, max_len) >= lengths.unsqueeze(1)


def as_tensor(value: object) -> torch.Tensor | None:
    """Extract a tensor payload from raw tensors or single-element lists."""
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if isinstance(value, torch.Tensor):
        return value
    return None


def as_bool(value: object) -> bool:
    """Extract a boolean payload from tensors, lists, None, or raw values."""
    if isinstance(value, list):
        if not value:
            return False
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.reshape(-1)[0].item())
    if value is None:
        return False
    return bool(value)


def as_str(value: object) -> str | None:
    """Extract a string payload from raw values or single-element lists."""
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if value is None:
        return None
    return str(value)


def split_request_ids(
    ids: torch.Tensor,
    seq_token_counts: list[int] | None = None,
) -> list[torch.Tensor]:
    """Split concatenated input ids into per-request segments."""
    if seq_token_counts is not None:
        boundaries = [0]
        for count in seq_token_counts:
            boundaries.append(boundaries[-1] + int(count))
        total = ids.numel()
        return [ids[boundaries[i] : min(boundaries[i + 1], total)] for i in range(len(seq_token_counts))]

    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + int(s))
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
    except ImportError:
        pass

    return [ids]


class GLMTTSDiTForGeneration(nn.Module):
    """GLM-TTS DiT flow-matching stage, wrapped as LLM_GENERATION.

    Reuses GLMTTSDiT (diffusion transformer) + vocoder internally, but
    exposes the forward(input_ids, positions, **kwargs) -> OmniOutput
    interface required by OmniGenerationScheduler + gpu_model_runner.

    Attributes:
        have_multimodal_outputs: Signals scheduler to collect multimodal outputs.
        enable_update_additional_information: Allows async_chunk updates.
    """

    have_multimodal_outputs = True
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model_stage = "glm_tts_dit"

        # Resolve model path
        model_path = vllm_config.model_config.model
        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_path)

        # GLM-TTS has separate directories: llm/, flow/, vocos/ under root
        model_root = model_path
        if os.path.basename(model_root.rstrip("/\\")) in {"llm", "flow"}:
            parent = os.path.dirname(model_root.rstrip("/\\"))
            if os.path.isdir(os.path.join(parent, "flow")):
                model_root = parent
        self.model_root = model_root

        # Load flow config (JSON first, then merge YAML overrides)
        flow_dir = os.path.join(model_root, "flow")
        self.hf_flow_config: dict[str, Any] = {}
        config_json = os.path.join(flow_dir, "config.json")
        if os.path.exists(config_json):
            with open(config_json) as f:
                self.hf_flow_config = json.load(f)

        # YAML config may contain critical params not in config.json
        # (e.g. spkr_emb_adaln, remove_spkr_concat_condition)
        config_yaml = os.path.join(flow_dir, "config.yaml")
        if os.path.exists(config_yaml):
            try:
                import yaml

                with open(config_yaml) as f:
                    yaml_config = yaml.safe_load(f)
                    if isinstance(yaml_config, dict):
                        flow_cfg = yaml_config.get("flow", yaml_config)
                        if isinstance(flow_cfg, dict):
                            for k, v in flow_cfg.items():
                                if k not in self.hf_flow_config and not k.startswith("!"):
                                    self.hf_flow_config[k] = v
            except Exception as e:
                logger.warning("Failed to parse config.yaml: %s", e)
                with open(config_yaml) as f:
                    for line in f:
                        line = line.strip()
                        if ":" in line and not line.startswith("flow") and not line.startswith("#"):
                            key, value = line.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            if key and value and key not in self.hf_flow_config:
                                if value.lower() == "true":
                                    self.hf_flow_config[key] = True
                                elif value.lower() == "false":
                                    self.hf_flow_config[key] = False
                                elif value.replace(".", "").isdigit():
                                    self.hf_flow_config[key] = float(value) if "." in value else int(value)
                                else:
                                    self.hf_flow_config[key] = value

        logger.info("GLM-TTS flow config: %s", self.hf_flow_config)

        # Flow configuration
        self.mel_dim = self.hf_flow_config.get("mel_dim", 80)
        self.input_frame_rate = self.hf_flow_config.get("input_frame_rate", 25.0)
        self.mel_framerate = self.hf_flow_config.get("mel_framerate", 50)
        self.sample_rate = self.hf_flow_config.get("sample_rate", 24000)

        # Sampling configuration
        self.n_timesteps = self.hf_flow_config.get("n_timesteps", 10)
        self.t_scheduler = self.hf_flow_config.get("t_scheduler", "cosine")
        self.inference_cfg_rate = self.hf_flow_config.get("inference_cfg_rate", 0.7)
        self.speech_token_cfg = self.hf_flow_config.get("speech_token_cfg", True)

        # Speaker embedding config
        self.spk_embed_dim = self.hf_flow_config.get("spk_embed_dim", 80)
        self.spkr_emb_adaln = self.hf_flow_config.get("spkr_emb_adaLN", False)
        self.remove_spkr_concat_condition = self.hf_flow_config.get("remove_spkr_concat_condition", False)

        # Initialize DiT model
        from vllm_omni.diffusion.models.glm_tts.glm_tts_dit import GLMTTSDiT

        trans_dim = self.hf_flow_config.get("trans_dim", 768)
        depth = self.hf_flow_config.get("depth", 18)
        heads = self.hf_flow_config.get("heads", 12)
        condition_dim = self.mel_dim if self.remove_spkr_concat_condition else self.mel_dim + self.spk_embed_dim

        self.dit = GLMTTSDiT(
            trans_dim=trans_dim,
            depth=depth,
            heads=heads,
            dim_head=64,
            ff_mult=2,
            dropout=0.1,
            mel_dim=self.mel_dim,
            text_vocab_size=100000,
            text_emb_dim=512,
            conv_layers=4,
            condition_dim=condition_dim,
            spkr_emb_adaln=self.spkr_emb_adaln,
            spkr_dim=192,
            use_wavlm_emb=False,
        )

        # Speaker embedding projection (if not using AdaLN)
        if not self.remove_spkr_concat_condition:
            self.spk_embed_affine_layer = nn.Linear(192, self.spk_embed_dim)

        # Vocoder (lazy loaded)
        self._vocoder = None
        self._vocoder_sample_rate: int | None = None

        # Official GLM-TTS streaming reuses per-Euler-step diffusion latents
        # for the stable prefix of each request.
        self._stream_diff_cache_by_req: dict[str, dict[int | str, Any]] = {}
        self._stream_prompt_payload_by_req: dict[str, dict[str, torch.Tensor]] = {}

        # Weight path
        self._flow_weights_path = os.path.join(model_root, "flow", "flow.pt")

        logger.info(
            "GLMTTSDiTForGeneration init: mel_dim=%d, input_frame_rate=%.1f, "
            "mel_framerate=%d, sample_rate=%d, n_timesteps=%d",
            self.mel_dim,
            self.input_frame_rate,
            self.mel_framerate,
            self.sample_rate,
            self.n_timesteps,
        )

    # ---- LLM_GENERATION interface ----

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        """Return zero embeddings (DiT does not use token embeddings)."""
        assert input_ids.dim() == 1
        hidden = int(self.config.hidden_size)
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
        )

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        """DiT does not produce logits for sampling."""
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: Any,
    ) -> OmniOutput:
        """Run DiT flow-matching + vocoder.

        input_ids contains speech token IDs (code_predictor_codes) set by
        chunk_transfer_adapter (async) or prompt_token_ids (sync).

        Per-request metadata comes from model_intermediate_buffer:
          - speech_tokens: full speech token list (sync path)
          - prompt_token: reference speech tokens for voice cloning
          - prompt_feat: reference mel features for voice cloning
          - embedding: speaker embedding (192-dim)
          - token_offset: left context offset (async streaming)
          - stream_finished: whether AR generation is done (async)
          - req_id: request identifier (async)
        """
        runtime_info = kwargs.get("model_intermediate_buffer")
        if runtime_info is None:
            runtime_info = kwargs.get("runtime_additional_information", [])

        seq_token_counts = kwargs.get("seq_token_counts")
        flat_ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = split_request_ids(flat_ids, seq_token_counts)

        num_reqs = max(1, len(request_ids_list))
        sample_rate = torch.tensor(int(self.sample_rate), dtype=torch.int32)
        empty_audio = torch.zeros((0,), dtype=torch.float32, device=input_ids.device)
        audios: list[torch.Tensor] = [empty_audio] * num_reqs
        srs: list[torch.Tensor] = [sample_rate] * num_reqs

        if not isinstance(runtime_info, list):
            runtime_info = []

        for idx, req_ids in enumerate(request_ids_list):
            info = runtime_info[idx] if idx < len(runtime_info) and isinstance(runtime_info[idx], dict) else {}
            req_id = as_str(info.get("req_id")) if info else None
            stream_finished = as_bool(info.get("stream_finished")) if info else False
            prompt_token = as_tensor(info.get("prompt_token")) if info else None
            prompt_feat = as_tensor(info.get("prompt_feat")) if info else None
            embedding = as_tensor(info.get("embedding")) if info else None

            # Determine speech tokens
            speech_tokens_raw = info.get("speech_tokens") if info else None

            # Streaming path detection (same as CosyVoice3)
            uses_streaming = bool(info) and (
                "stream_finished" in info or "token_offset" in info or "left_context_size" in info
            )
            if uses_streaming and req_id is not None:
                prompt_payload = self._stream_prompt_payload_by_req.setdefault(req_id, {})
                if prompt_token is not None:
                    prompt_payload["prompt_token"] = prompt_token
                elif "prompt_token" in prompt_payload:
                    prompt_token = prompt_payload["prompt_token"]
                if prompt_feat is not None:
                    prompt_payload["prompt_feat"] = prompt_feat
                elif "prompt_feat" in prompt_payload:
                    prompt_feat = prompt_payload["prompt_feat"]
                if embedding is not None:
                    prompt_payload["embedding"] = embedding
                elif "embedding" in prompt_payload:
                    embedding = prompt_payload["embedding"]

            # Get valid speech tokens from input_ids
            valid_mask = (req_ids >= 0) & (req_ids < 32768)
            token = req_ids[valid_mask]

            if token.numel() == 0:
                # Empty tokens: emit silence, cleanup on finish
                if stream_finished and req_id is not None:
                    self._stream_diff_cache_by_req.pop(req_id, None)
                    self._stream_prompt_payload_by_req.pop(req_id, None)
                audios[idx] = empty_audio
                continue

            if uses_streaming:
                # Async streaming path
                token_offset = 0
                try:
                    if info and "token_offset" in info:
                        token_offset = max(0, int(info.get("token_offset", 0)))
                    elif info:
                        token_offset = max(0, int(info.get("left_context_size", 0)))
                except (TypeError, ValueError):
                    token_offset = 0

                audio = self._forward_full(
                    speech_tokens=token,
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=embedding,
                    token_offset=token_offset,
                    req_id=req_id,
                )
                if stream_finished and req_id is not None:
                    self._stream_diff_cache_by_req.pop(req_id, None)
                    self._stream_prompt_payload_by_req.pop(req_id, None)
            else:
                # Sync (non-streaming) path: use speech_tokens from info or input_ids
                if isinstance(speech_tokens_raw, list) and speech_tokens_raw:
                    token = torch.tensor(speech_tokens_raw, dtype=torch.long, device=input_ids.device)
                    valid_mask = (token >= 0) & (token < 32768)
                    token = token[valid_mask]

                audio = self._forward_full(
                    speech_tokens=token,
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=embedding,
                )

            audios[idx] = audio.reshape(-1).to(dtype=torch.float32)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audios, "sr": srs},
        )

    # ---- Core inference ----

    @torch.inference_mode()
    def _forward_full(
        self,
        speech_tokens: torch.Tensor,
        prompt_token: torch.Tensor | None = None,
        prompt_feat: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
        token_offset: int = 0,
        req_id: str | None = None,
    ) -> torch.Tensor:
        """Run full DiT flow-matching + vocoder pipeline.

        Args:
            speech_tokens: 1D speech token IDs [T_token], range [0, 32767].
            prompt_token: Reference speech tokens for voice cloning [T_prompt].
            prompt_feat: Reference mel features [T_feat, mel_dim].
            embedding: Speaker embedding [192].
            token_offset: Stable speech-token prefix already emitted.
            req_id: Streaming request id used for diffusion latent cache.

        Returns:
            Audio waveform tensor [samples].
        """
        device = speech_tokens.device

        # Ensure 2D [B=1, T]
        if speech_tokens.ndim == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        generated_token_len = int(speech_tokens.shape[1])

        # Voice cloning: prepend prompt_token to speech_tokens
        has_voice_clone = prompt_token is not None or embedding is not None
        if prompt_token is not None:
            if isinstance(prompt_token, np.ndarray):
                prompt_token = torch.from_numpy(prompt_token).long().to(device)
            else:
                prompt_token = prompt_token.to(device=device, dtype=torch.long)
            if prompt_token.ndim == 1:
                prompt_token = prompt_token.unsqueeze(0)
            prompt_token_len = prompt_token.shape[1]
            speech_tokens = torch.cat([prompt_token, speech_tokens], dim=1)
        else:
            prompt_token_len = 0
        has_prompt_audio_context = prompt_token_len > 0

        token_len = speech_tokens.shape[1]
        if token_len == 0:
            return torch.zeros(0, device=device)

        # Calculate mel length
        feat_len = int(token_len / self.input_frame_rate * self.mel_framerate)
        feat_len = max(1, feat_len)

        # Prepare padding mask
        padding_mask = ~make_pad_mask(torch.tensor([feat_len], device=device), max_len=feat_len)

        # Prepare mel condition
        mel_cond = torch.zeros([1, feat_len, self.mel_dim], device=device)

        # Voice cloning: overlay prompt_feat onto mel_cond
        prompt_feat_len = 0
        if prompt_feat is not None:
            if isinstance(prompt_feat, np.ndarray):
                prompt_feat = torch.from_numpy(prompt_feat).to(device=device, dtype=mel_cond.dtype)
            else:
                prompt_feat = prompt_feat.to(device=device, dtype=mel_cond.dtype)
            if prompt_feat.ndim == 2:
                prompt_feat = prompt_feat.unsqueeze(0)
            copy_len = min(prompt_feat.shape[1], feat_len)
            mel_cond[:, :copy_len, :] = prompt_feat[:, -copy_len:, :]
            prompt_feat_len = copy_len
        has_prompt_audio_context = has_prompt_audio_context and prompt_feat_len > 0

        # Prepare condition (mel_cond + optional speaker embedding)
        spkr_embedding = None
        if not self.remove_spkr_concat_condition:
            if has_voice_clone and embedding is not None:
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding).to(device=device, dtype=mel_cond.dtype)
                else:
                    embedding = embedding.to(device=device, dtype=mel_cond.dtype)
                if embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)
                spk_embedding_normed = torch.nn.functional.normalize(embedding, dim=1)
                spk_embedding = self.spk_embed_affine_layer(spk_embedding_normed)
                spk_embedding_expanded = spk_embedding.unsqueeze(1).expand(-1, feat_len, -1)
                condition = torch.cat([mel_cond, spk_embedding_expanded], dim=-1)
                spkr_embedding = spk_embedding_normed
            else:
                spk_zeros = torch.zeros([1, feat_len, self.spk_embed_dim], device=device)
                condition = torch.cat([mel_cond, spk_zeros], dim=-1)
        else:
            condition = mel_cond

        # For spkr_emb_adaln mode
        if self.spkr_emb_adaln:
            if has_voice_clone and embedding is not None:
                if spkr_embedding is None:
                    if isinstance(embedding, np.ndarray):
                        spkr_embedding = torch.from_numpy(embedding).to(device=device, dtype=mel_cond.dtype)
                    elif isinstance(embedding, torch.Tensor):
                        spkr_embedding = embedding.to(device=device, dtype=mel_cond.dtype)
                    if spkr_embedding is not None and spkr_embedding.ndim == 1:
                        spkr_embedding = spkr_embedding.unsqueeze(0)
                    if spkr_embedding is not None:
                        spkr_embedding = torch.nn.functional.normalize(spkr_embedding, dim=1)
            else:
                spkr_embedding = torch.zeros([1, 192], device=device, dtype=mel_cond.dtype)

        last_step_cache = None
        if req_id is not None:
            last_step_cache = self._stream_diff_cache_by_req.get(req_id)

        # Run flow matching sampling (Euler ODE)
        mel, current_step_cache = self._do_sample(
            speech_tokens=speech_tokens,
            mel_cond=mel_cond,
            condition=condition,
            padding_mask=padding_mask,
            spkr_embedding=spkr_embedding,
            n_timesteps=self.n_timesteps,
            last_step_cache=last_step_cache,
        )

        if req_id is not None:
            stable_mel_len = int(generated_token_len / self.input_frame_rate * self.mel_framerate)
            current_step_cache["override_len"] = prompt_feat_len + stable_mel_len + 1
            self._stream_diff_cache_by_req[req_id] = current_step_cache

        # Remove prompt part from output
        if has_prompt_audio_context:
            crop_len = min(prompt_feat_len, max(0, int(mel.shape[1]) - 1))
            if crop_len > 0:
                mel = mel[:, crop_len:, :]

        # For streaming: crop by token_offset (remove left context from audio)
        if token_offset > 0:
            offset_mel_frames = int(token_offset / self.input_frame_rate * self.mel_framerate)
            if offset_mel_frames > 0 and offset_mel_frames < mel.shape[1]:
                mel = mel[:, offset_mel_frames:, :]

        # Lazy-load vocoder on first use
        if self._vocoder is None:
            device = next(self.dit.parameters()).device
            self._vocoder, self._vocoder_sample_rate = load_vocoder(self.model_root, device, self.sample_rate)

        # Convert mel to audio via vocoder
        audio = mel_to_audio(self._vocoder, mel)

        # Resample if vocoder outputs different rate
        effective_sr = self._vocoder_sample_rate or self.sample_rate
        if effective_sr != self.sample_rate and audio is not None and audio.numel() > 0:
            try:
                import torchaudio

                audio = torchaudio.functional.resample(audio, effective_sr, self.sample_rate)
            except Exception as e:
                logger.warning(
                    "Failed to resample audio from %d to %d: %s",
                    effective_sr,
                    self.sample_rate,
                    e,
                )

        return audio.reshape(-1)

    def _do_sample(
        self,
        speech_tokens: torch.Tensor,
        mel_cond: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        spkr_embedding: torch.Tensor | None,
        n_timesteps: int,
        last_step_cache: dict[int | str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[int | str, Any]]:
        """Run Euler ODE sampling for flow matching.

        Euler ODE sampling for flow matching (same algorithm as upstream GLM-TTS).
        """
        dtype = mel_cond.dtype
        device = speech_tokens.device

        # Initial noise
        x = torch.randn_like(mel_cond)
        current_step_cache: dict[int | str, Any] = {}

        # Time schedule
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Euler integration
        for step in range(n_timesteps):
            cache_step = step + 1
            if last_step_cache is not None and cache_step in last_step_cache:
                x_cache = last_step_cache[cache_step].get("x")
                if isinstance(x_cache, torch.Tensor):
                    override_len = int(last_step_cache.get("override_len", x_cache.shape[1]))
                    safe_len = min(int(x.shape[1]), int(x_cache.shape[1]), max(0, override_len))
                    if safe_len > 0:
                        x[:, :safe_len, :] = x_cache[:, :safe_len, :].to(device=device, dtype=x.dtype)
            current_step_cache[cache_step] = {"x": x.detach().clone()}

            t_current = t_span[step]
            dt = t_span[step + 1] - t_current

            # Conditional forward pass
            dphi_dt = self.dit(
                middle_point=x,
                condition=condition,
                text=speech_tokens,
                time_step=t_current.unsqueeze(0),
                padding_mask=padding_mask,
                spkr_emb=spkr_embedding if self.spkr_emb_adaln else None,
            )

            # Classifier-free guidance
            if self.inference_cfg_rate > 0:
                text_uncond = torch.zeros_like(speech_tokens) if self.speech_token_cfg else speech_tokens
                spkr_uncond = torch.zeros_like(spkr_embedding) if spkr_embedding is not None else None
                cfg_dphi_dt = self.dit(
                    middle_point=x,
                    condition=torch.zeros_like(condition),
                    text=text_uncond,
                    time_step=t_current.unsqueeze(0),
                    padding_mask=padding_mask,
                    spkr_emb=spkr_uncond if self.spkr_emb_adaln else None,
                )
                dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt

            # Euler step
            x = x + dt * dphi_dt

        return x, current_step_cache

    def load_weights(self, weights: Any) -> set[str]:
        """Load DiT + vocoder weights.

        Loads directly from {model_root}/flow/flow.pt — the vLLM default
        weight iterator is ignored because DiT weights are not in
        safetensors/bin format.  This matches CosyVoice3 code2wav behavior.
        """
        # Ignore the vLLM-provided weights iterator.
        # DiT weights live in flow/flow.pt, not standard safetensors.
        weight_dict: dict[str, torch.Tensor] = {}
        if os.path.exists(self._flow_weights_path):
            logger.info("Loading GLM-TTS flow weights from %s", self._flow_weights_path)
            checkpoint = torch.load(self._flow_weights_path, map_location="cpu", weights_only=True)
            if "model" in checkpoint:
                weight_dict = checkpoint["model"]
            else:
                weight_dict = checkpoint
        else:
            logger.warning("Flow weights not found at %s", self._flow_weights_path)

        # Load DiT weights
        dit_state = {}
        for name, tensor in weight_dict.items():
            if "inv_freq" in name or "rotary_embed" in name:
                continue
            if name.startswith("estimator."):
                dit_state[name[len("estimator.") :]] = tensor
            elif name.startswith("dit."):
                dit_state[name[len("dit.") :]] = tensor

        if dit_state:
            missing, unexpected = self.dit.load_state_dict(dit_state, strict=False)
            missing = [k for k in missing if "inv_freq" not in k]
            unexpected = [k for k in unexpected if "inv_freq" not in k]
            if missing:
                logger.warning("Missing DiT weights: %s", missing[:10])
            if unexpected:
                logger.warning("Unexpected DiT weights: %s", unexpected[:10])
            logger.info("Loaded %d DiT weights", len(dit_state))

        # Load speaker embedding layer (only exists when
        # remove_spkr_concat_condition is False)
        loaded_spk: set[str] = set()
        if hasattr(self, "spk_embed_affine_layer"):
            if "spk_embed_affine_layer.weight" in weight_dict:
                self.spk_embed_affine_layer.weight.data = weight_dict["spk_embed_affine_layer.weight"]
                loaded_spk.add("spk_embed_affine_layer.weight")
            if "spk_embed_affine_layer.bias" in weight_dict:
                self.spk_embed_affine_layer.bias.data = weight_dict["spk_embed_affine_layer.bias"]
                loaded_spk.add("spk_embed_affine_layer.bias")

        # Return parameter names matching self.named_parameters() paths.
        # Since this is nested as _dit_gen inside GLMTTSForConditionalGeneration,
        # the actual parameter names from model.named_parameters() are like
        # "_dit_gen.dit.X" or "_dit_gen.spk_embed_affine_layer.X".
        # However, load_weights is called on the top-level model, so we need
        # to return names that match the top-level model's named_parameters().
        # We return both prefixed and unprefixed forms to cover all cases.
        loaded_params: set[str] = set()
        for k in dit_state.keys():
            loaded_params.add(f"dit.{k}")
            loaded_params.add(f"_dit_gen.dit.{k}")
        for k in loaded_spk:
            loaded_params.add(k)
            loaded_params.add(f"_dit_gen.{k}")
        logger.info("Loaded GLMTTSDiTForGeneration weights: %d params", len(loaded_params))
        return loaded_params
