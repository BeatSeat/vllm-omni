# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4-Voice decoder: CosyVoice flow matching + HiFi-T vocoder.

Ported from the reference ``flow_inference.py`` AudioDecoder in
https://github.com/zai-org/GLM-4-Voice.  Converts discrete speech
tokens (0..16383) to 22050 Hz waveform via:

    tokens → flow.inference() → mel-spectrogram → hift.inference() → waveform

This module is instantiated by ``GLM4VoiceForConditionalGeneration`` when
``model_stage == "glm4_voice_decoder"``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 22050
_MEL_HOP_SIZE = 256


@dataclass
class _StreamState:
    """Per-request streaming state for mel overlap and vocoder cache."""

    mel_overlap: torch.Tensor | None = None
    hift_cache_mel: torch.Tensor | None = None
    hift_cache_source: torch.Tensor | None = None
    wav_pointer: int = 0
    chunk_idx: int = 0


def _bool_value(v: Any) -> bool:
    if isinstance(v, torch.Tensor):
        return bool(v.item())
    return bool(v)


class GLM4VoiceDecoderForGeneration(nn.Module):
    """Wrapper for CosyVoice flow decoder + HiFi-T vocoder.

    Constructs flow (MaskedDiffWithXvec) and vocoder (HiFTGenerator) with
    hardcoded GLM-4-Voice parameters, then loads ``flow.pt`` / ``hift.pt``
    state dicts. No external dependencies (HyperPyYAML, matcha, CosyVoice).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = self._resolve_decoder_dtype(getattr(self.config, "decoder_dtype", "float32"))

        self.flow: nn.Module | None = None
        self.hift: nn.Module | None = None

        self.flow_n_timesteps = int(getattr(self.config, "flow_n_timesteps", 10))
        self.enable_decoder_compile = bool(getattr(self.config, "enable_decoder_compile", False))
        self.enable_tf32 = bool(getattr(self.config, "enable_tf32", False))
        self.remove_weight_norm = bool(getattr(self.config, "remove_weight_norm", True))
        self.token_overlap_len = 5
        self.mel_overlap_len: int = 0
        self.mel_window: np.ndarray | None = None
        self.mel_cache_len = 1
        self.source_cache_len: int = 0
        self._mel_window_tensor: torch.Tensor | None = None

        self._stream_lock = threading.Lock()
        self._stream_state: dict[str, _StreamState] = {}

    def _resolve_decoder_dtype(self, dtype_name: Any) -> torch.dtype:
        if isinstance(dtype_name, torch.dtype):
            dtype = dtype_name
        else:
            normalized = str(dtype_name).lower().replace("torch.", "")
            dtype = {
                "float": torch.float32,
                "float32": torch.float32,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
            }.get(normalized, torch.float32)
        if self.device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            logger.warning("GLM-4-Voice decoder dtype %s is CUDA-only; falling back to float32 on CPU", dtype)
            return torch.float32
        return dtype

    # ------------------------------------------------------------------
    # Forward — reads speech tokens from input_ids
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if self.flow is None or self.hift is None:
            return torch.zeros(1, 1, device=self.device, dtype=self.dtype)

        if input_ids is None or input_ids.numel() == 0:
            return torch.zeros(1, 1, device=self.device, dtype=self.dtype)

        token = input_ids.reshape(-1).to(device=self.device, dtype=torch.int64)
        token = token[(token >= 0) & (token < 16384)].unsqueeze(0)

        if token.numel() == 0:
            return torch.zeros(1, 1, device=self.device, dtype=self.dtype)

        is_finalize, req_id = self._parse_runtime_info(kwargs)

        prompt_token = torch.zeros(1, 0, dtype=torch.int64, device=self.device)
        prompt_feat = torch.zeros(1, 0, 80, device=self.device, dtype=self.dtype)
        embedding = torch.zeros(1, 192, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            tts_speech = self._token2wav(
                token=token,
                req_id=req_id,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                finalize=is_finalize,
            )

        self._last_audio = tts_speech
        self._last_sample_rate = _SAMPLE_RATE

        return torch.zeros(1, 1, device=self.device, dtype=self.dtype)

    def _parse_runtime_info(self, kwargs: dict[str, Any]) -> tuple[bool, str]:
        """Extract (is_finalize, req_id) from model_intermediate_buffer.

        The buffer is a list of per-request dicts. Each dict may contain
        a nested ``meta`` sub-dict with ``stream_finished``, ``finished``,
        and ``req_id`` fields (from OmniPayloadStruct serialization).
        """
        runtime_info = kwargs.get("model_intermediate_buffer")
        if runtime_info is None:
            runtime_info = kwargs.get("runtime_additional_information", [])

        if not runtime_info or not isinstance(runtime_info, list):
            return True, "0"

        raw = runtime_info[0] if isinstance(runtime_info[0], dict) else {}
        if not raw:
            return True, "0"

        meta = raw.get("meta")
        if isinstance(meta, dict):
            sf = meta.get("stream_finished", meta.get("finished"))
            is_finalize = _bool_value(sf) if sf is not None else True
            req_id_list = meta.get("req_id", [])
            req_id = str(req_id_list[0]) if req_id_list else "0"
            return is_finalize, req_id

        sf = raw.get("stream_finished", raw.get("finished"))
        is_finalize = _bool_value(sf) if sf is not None else True
        return is_finalize, "0"

    def _token2wav(
        self,
        token: torch.Tensor,
        req_id: str,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        finalize: bool = False,
    ) -> torch.Tensor:
        assert self.flow is not None and self.hift is not None

        token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=self.device)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=self.device)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=self.device)

        tts_mel = self.flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            n_timesteps=self.flow_n_timesteps,
        )

        with self._stream_lock:
            state = self._stream_state.get(req_id)
            if state is None:
                state = _StreamState()
                self._stream_state[req_id] = state
            mel_overlap_snap = state.mel_overlap
            hift_cache_mel_snap = state.hift_cache_mel
            hift_cache_source_snap = state.hift_cache_source

        if mel_overlap_snap is not None and self.mel_overlap_len > 0:
            tts_mel = self._fade_in_out(tts_mel, mel_overlap_snap)

        if hift_cache_mel_snap is not None:
            tts_mel = torch.cat([hift_cache_mel_snap, tts_mel], dim=2)
            hift_cache_source = hift_cache_source_snap
        else:
            hift_cache_source = torch.zeros(1, 1, 0, device=self.device, dtype=self.dtype)

        if not finalize:
            new_mel_overlap = None
            if self.mel_overlap_len > 0:
                if tts_mel.shape[-1] > self.mel_overlap_len:
                    new_mel_overlap = tts_mel[:, :, -self.mel_overlap_len :]
                    tts_mel = tts_mel[:, :, : -self.mel_overlap_len]
                else:
                    with self._stream_lock:
                        if req_id in self._stream_state:
                            st = self._stream_state[req_id]
                            st.mel_overlap = tts_mel
                    return torch.zeros(1, 0, device=self.device, dtype=self.dtype)

            tts_speech, tts_source = self.hift.inference(tts_mel, cache_source=hift_cache_source)

            with self._stream_lock:
                if req_id in self._stream_state:
                    st = self._stream_state[req_id]
                    st.mel_overlap = new_mel_overlap
                    st.hift_cache_mel = tts_mel[:, :, -self.mel_cache_len :]
                    st.hift_cache_source = tts_source[:, :, -self.source_cache_len :]

            tts_speech = tts_speech[:, : -self.source_cache_len]
        else:
            tts_speech, _ = self.hift.inference(tts_mel, cache_source=hift_cache_source)
            with self._stream_lock:
                self._stream_state.pop(req_id, None)

        return tts_speech

    def _fade_in_out(self, fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor) -> torch.Tensor:
        if self.mel_window is None or self.mel_overlap_len <= 0:
            return fade_in_mel

        overlap = min(self.mel_overlap_len, fade_in_mel.shape[-1], fade_out_mel.shape[-1])
        if overlap <= 0:
            return fade_in_mel

        device = fade_in_mel.device
        if overlap == self.mel_overlap_len:
            if (
                self._mel_window_tensor is None
                or self._mel_window_tensor.device != device
                or self._mel_window_tensor.dtype != fade_in_mel.dtype
            ):
                self._mel_window_tensor = torch.as_tensor(
                    self.mel_window,
                    device=device,
                    dtype=fade_in_mel.dtype,
                ).view(1, 1, -1)
            window = self._mel_window_tensor
        else:
            window = torch.hamming_window(
                2 * overlap,
                periodic=False,
                device=device,
                dtype=fade_in_mel.dtype,
            ).view(1, 1, -1)

        fade_in_mel[..., :overlap] = (
            fade_in_mel[..., :overlap] * window[..., :overlap]
            + fade_out_mel[..., -overlap:] * window[..., overlap:]
        )
        return fade_in_mel

    # ------------------------------------------------------------------
    # Stubs for LLM_GENERATION execution type
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> list | None:
        return None

    def make_omni_output(self, model_output: Any, **kwargs: Any) -> OmniOutput:
        audio = getattr(self, "_last_audio", None)
        sr = getattr(self, "_last_sample_rate", _SAMPLE_RATE)

        multimodal_outputs: dict[str, Any] = {}
        if audio is not None:
            if isinstance(audio, torch.Tensor):
                audio_flat = audio.squeeze().float().cpu()
            else:
                audio_flat = torch.tensor(audio, dtype=torch.float32).squeeze()
            multimodal_outputs["audio"] = audio_flat
            multimodal_outputs["sample_rate"] = sr
            self._last_audio = None

        text_hs = model_output if isinstance(model_output, torch.Tensor) else torch.zeros(1, 1)
        return OmniOutput(
            text_hidden_states=text_hs,
            multimodal_outputs=multimodal_outputs,
        )

    def on_request_finish(self, req_id: str) -> None:
        with self._stream_lock:
            self._stream_state.pop(req_id, None)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Any) -> None:
        decoder_dir = self._find_decoder_dir()
        if decoder_dir is None:
            logger.warning("GLM-4-Voice decoder directory not found. Decoder will not produce audio.")
            return

        flow_path = decoder_dir / "flow.pt"
        hift_path = decoder_dir / "hift.pt"

        self.flow = self._build_flow_model()
        if flow_path.exists():
            flow_state = torch.load(flow_path, map_location=self.device, weights_only=True)
            missing, unexpected = self.flow.load_state_dict(flow_state, strict=False)
            if missing:
                logger.warning("Flow missing keys (%d): %s", len(missing), missing[:5])
            if unexpected:
                logger.warning("Flow unexpected keys (%d): %s", len(unexpected), unexpected[:5])
            logger.info("Loaded flow weights from %s (%d tensors)", flow_path, len(flow_state))
        else:
            logger.warning("flow.pt not found in %s", decoder_dir)

        self.hift = self._build_hift_model()
        if hift_path.exists():
            hift_state = torch.load(hift_path, map_location=self.device, weights_only=True)
            hift_state = self._remap_hift_state_dict(hift_state)
            missing, unexpected = self.hift.load_state_dict(hift_state, strict=False)
            if missing:
                logger.warning("HiFT missing keys (%d): %s", len(missing), missing[:5])
            if unexpected:
                logger.warning("HiFT unexpected keys (%d): %s", len(unexpected), unexpected[:5])
            logger.info("Loaded HiFi-T weights from %s (%d tensors)", hift_path, len(hift_state))
        else:
            logger.warning("hift.pt not found in %s", decoder_dir)

        self.flow.to(device=self.device, dtype=self.dtype).eval()
        self.hift.to(device=self.device, dtype=self.dtype).eval()

        if self.enable_tf32 and self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            logger.info(
                "GLM-4-Voice decoder TF32 enabled process-wide: "
                "matmul.allow_tf32=%s cudnn.allow_tf32=%s float32_matmul_precision=%s",
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision(),
            )

        if self.remove_weight_norm:
            self._remove_weight_norm_all()
        if self.enable_decoder_compile:
            self._try_torch_compile()

        self.mel_overlap_len = int(self.token_overlap_len / 12.5 * _SAMPLE_RATE / _MEL_HOP_SIZE)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self._mel_window_tensor = None
        self.source_cache_len = int(self.mel_cache_len * _MEL_HOP_SIZE)

        logger.info(
            "GLM-4-Voice decoder ready: mel_overlap=%d, source_cache=%d, flow_steps=%d, dtype=%s",
            self.mel_overlap_len,
            self.source_cache_len,
            self.flow_n_timesteps,
            self.dtype,
        )

    def _remove_weight_norm_all(self) -> None:
        """Fold weight_norm parametrizations into plain weights (inference-equivalent, fewer ops)."""
        count = 0
        for model in [self.flow, self.hift]:
            if model is None:
                continue
            for _name, mod in model.named_modules():
                if hasattr(mod, "parametrizations") and hasattr(mod.parametrizations, "weight"):
                    try:
                        torch.nn.utils.parametrize.remove_parametrizations(mod, "weight")
                        count += 1
                    except Exception:
                        pass
                elif hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
                    try:
                        torch.nn.utils.remove_weight_norm(mod)
                        count += 1
                    except Exception:
                        pass
        if count:
            logger.info("Removed weight_norm from %d modules (inference-equivalent)", count)

    def _try_torch_compile(self) -> None:
        """Apply torch.compile for kernel fusion (inference-equivalent, fewer kernel launches).

        Compiles specific inference methods rather than entire modules, since the
        call path uses .inference() not .forward(). Uses mode="default" (triton
        fusion) instead of "reduce-overhead" (CUDA graphs) because mel sequence
        lengths vary per chunk.
        """
        if not hasattr(torch, "compile"):
            return
        try:
            self.hift.inference = torch.compile(self.hift.inference, mode="default")
            logger.info("Applied torch.compile to HiFTGenerator.inference")
        except Exception as e:
            logger.warning("torch.compile failed for HiFTGenerator: %s", e)
        try:
            assert self.flow is not None and self.flow.decoder is not None
            self.flow.decoder.estimator = torch.compile(
                self.flow.decoder.estimator, mode="default",
            )
            logger.info("Applied torch.compile to flow ConditionalDecoder (estimator)")
        except Exception as e:
            logger.warning("torch.compile failed for flow estimator: %s", e)

    def _find_decoder_dir(self) -> Path | None:
        candidates = [
            Path("glm-4-voice-decoder"),
            Path.home() / ".cache/huggingface/hub/models--THUDM--glm-4-voice-decoder/snapshots",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                if candidate.name == "snapshots":
                    subs = sorted(candidate.iterdir())
                    if subs:
                        return subs[-1]
                return candidate

        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(_GLM4_VOICE_DECODER_REPO, local_files_only=False)
            return Path(path)
        except Exception:
            logger.debug("Could not download decoder from HuggingFace.")

        return None

    def _build_flow_model(self) -> nn.Module:
        from .flow_components import (
            ConditionalCFM,
            ConditionalDecoder,
            ConformerEncoder,
            InterpolateRegulator,
            MaskedDiffWithXvec,
        )

        encoder = ConformerEncoder(
            input_size=512, output_size=512, attention_heads=8,
            linear_units=2048, num_blocks=6, dropout_rate=0.1,
            positional_dropout_rate=0.1, attention_dropout_rate=0.0,
            normalize_before=True, macaron_style=False,
            use_cnn_module=False, cnn_module_kernel=15, causal=False,
        )
        length_regulator = InterpolateRegulator(
            channels=80, sampling_ratios=(1, 1, 1, 1), groups=1,
        )
        estimator = ConditionalDecoder(
            in_channels=320, out_channels=80, channels=(256, 256),
            dropout=0.05, attention_head_dim=64, n_blocks=4,
            num_mid_blocks=12, num_heads=8, act_fn="gelu",
        )
        decoder = ConditionalCFM(
            in_channels=240, n_spks=1, spk_emb_dim=80,
            estimator=estimator, cfm_params={
                "sigma_min": 1e-6, "solver": "euler",
                "t_scheduler": "cosine", "inference_cfg_rate": 0.7,
            },
        )
        return MaskedDiffWithXvec(
            input_size=512, output_size=80, spk_embed_dim=192,
            vocab_size=16384, input_frame_rate=12.5,
            encoder=encoder, length_regulator=length_regulator,
            decoder=decoder,
        )

    def _build_hift_model(self) -> nn.Module:
        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
            HiFTGenerator,
        )

        from .flow_components import ConvRNNF0Predictor

        f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
        return HiFTGenerator(
            in_channels=80, base_channels=512, nb_harmonics=8,
            sampling_rate=22050, upsample_rates=[8, 8],
            upsample_kernel_sizes=[16, 16], f0_predictor=f0_predictor,
        )

    @staticmethod
    def _remap_hift_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        """Remap hift.pt keys: strip 'generator.' prefix, convert old weight_norm keys."""
        if any(k.startswith("generator.") for k in state_dict):
            state_dict = {k.replace("generator.", "", 1): v for k, v in state_dict.items()}

        has_old_wn = any(k.endswith(".weight_g") or k.endswith(".weight_v") for k in state_dict)
        if not has_old_wn:
            return state_dict

        converted: dict[str, Any] = {}
        wn_count = 0
        for k, v in state_dict.items():
            if k.endswith(".weight_g"):
                base = k[: -len(".weight_g")]
                converted[f"{base}.parametrizations.weight.original0"] = v
                wn_count += 1
            elif k.endswith(".weight_v"):
                base = k[: -len(".weight_v")]
                converted[f"{base}.parametrizations.weight.original1"] = v
                wn_count += 1
            else:
                converted[k] = v

        logger.info("Converted %d old-style weight_norm keys to parametrizations format", wn_count)
        return converted


_GLM4_VOICE_DECODER_REPO = "THUDM/glm-4-voice-decoder"
