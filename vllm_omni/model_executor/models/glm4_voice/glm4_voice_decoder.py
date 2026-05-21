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
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_omni.sequence import OmniOutput

logger = logging.getLogger(__name__)

# Output sample rate (matches reference implementation).
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


class GLM4VoiceDecoderForGeneration(nn.Module):
    """Wrapper for CosyVoice flow decoder + HiFi-T vocoder.

    Loads ``flow.pt`` and ``hift.pt`` from the ``glm-4-voice-decoder``
    checkpoint directory via HyperPyYAML config.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32  # flow matching requires fp32

        # Models loaded lazily in load_weights().
        self.flow: nn.Module | None = None
        self.hift: nn.Module | None = None

        # Streaming constants (from reference AudioDecoder).
        self.token_overlap_len = 5
        self.mel_overlap_len: int = 0  # computed after flow loads
        self.mel_window: np.ndarray | None = None
        self.mel_cache_len = 1
        self.source_cache_len: int = 0

        # Per-request streaming state.
        self._stream_lock = threading.Lock()
        self._stream_state: dict[str, _StreamState] = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Decode speech tokens to audio waveform.

        Expects runtime ``info`` dict (via ``kwargs``) with:
        - ``speech_tokens``: list[int] — audio token IDs (0-based, already
          offset-subtracted).
        - ``stream_finished``: bool — whether this is the final chunk.
        - ``_omni_req_id``: str — request identifier for streaming state.
        - ``prompt_token``: optional tensor [1, N] — previous tokens for
          voice cloning context.
        - ``prompt_feat``: optional tensor [1, N, 80] — previous mel for
          voice cloning context.
        - ``embedding``: optional tensor [1, 192] — speaker embedding.
        """
        info = kwargs.get("info", {})
        speech_tokens = info.get("speech_tokens", [])

        if not speech_tokens or self.flow is None or self.hift is None:
            # Nothing to decode; return dummy hidden states.
            return torch.zeros(1, 1, device=self.device, dtype=self.dtype)

        req_id = str(info.get("_omni_req_id", "0"))
        is_finalize = info.get("stream_finished", True)

        # Build token tensor [1, N].
        token = torch.tensor([speech_tokens], dtype=torch.int64, device=self.device)

        # Voice cloning context (optional).
        prompt_token = info.get("prompt_token")
        if prompt_token is None:
            prompt_token = torch.zeros(1, 0, dtype=torch.int64, device=self.device)
        elif not isinstance(prompt_token, torch.Tensor):
            prompt_token = torch.tensor(prompt_token, dtype=torch.int64, device=self.device).unsqueeze(0)
        else:
            prompt_token = prompt_token.to(self.device)

        prompt_feat = info.get("prompt_feat")
        if prompt_feat is None:
            prompt_feat = torch.zeros(1, 0, 80, device=self.device, dtype=self.dtype)
        elif not isinstance(prompt_feat, torch.Tensor):
            prompt_feat = torch.tensor(prompt_feat, device=self.device, dtype=self.dtype)
        else:
            prompt_feat = prompt_feat.to(device=self.device, dtype=self.dtype)

        embedding = info.get("embedding")
        if embedding is None:
            embedding = torch.zeros(1, 192, device=self.device, dtype=self.dtype)
        elif not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, device=self.device, dtype=self.dtype)
        else:
            embedding = embedding.to(device=self.device, dtype=self.dtype)

        # --- Token-to-waveform (ported from AudioDecoder.token2wav) ---
        with torch.no_grad():
            tts_speech = self._token2wav(
                token=token,
                req_id=req_id,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                finalize=is_finalize,
            )

        # Package as hidden states — the actual audio is in OmniOutput.
        # We store audio tensor in a buffer for make_omni_output to pick up.
        self._last_audio = tts_speech
        self._last_sample_rate = _SAMPLE_RATE

        return torch.zeros(1, 1, device=self.device, dtype=self.dtype)

    def _token2wav(
        self,
        token: torch.Tensor,
        req_id: str,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        finalize: bool = False,
    ) -> torch.Tensor:
        """Convert tokens to waveform with streaming mel overlap."""
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
        )

        # Streaming mel overlap and vocoder cache.
        with self._stream_lock:
            state = self._stream_state.get(req_id)
            if state is None:
                state = _StreamState()
                self._stream_state[req_id] = state

        # Mel overlap fade-in/fade-out.
        if state.mel_overlap is not None and self.mel_overlap_len > 0:
            tts_mel = self._fade_in_out(tts_mel, state.mel_overlap)

        # Prepend HiFi-T cache mel.
        if state.hift_cache_mel is not None:
            tts_mel = torch.cat([state.hift_cache_mel, tts_mel], dim=2)
            hift_cache_source = state.hift_cache_source
        else:
            hift_cache_source = torch.zeros(1, 1, 0, device=self.device)

        if not finalize:
            # Keep overlap mel.
            if self.mel_overlap_len > 0:
                state.mel_overlap = tts_mel[:, :, -self.mel_overlap_len :]
                tts_mel = tts_mel[:, :, : -self.mel_overlap_len]

            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

            # Update HiFi-T cache.
            state.hift_cache_mel = tts_mel[:, :, -self.mel_cache_len :]
            state.hift_cache_source = tts_source[:, :, -self.source_cache_len :]
            tts_speech = tts_speech[:, : -self.source_cache_len]
        else:
            tts_speech, _ = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            # Clean up streaming state.
            with self._stream_lock:
                self._stream_state.pop(req_id, None)

        return tts_speech

    def _fade_in_out(self, fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor) -> torch.Tensor:
        """Apply Hamming window crossfade between mel chunks."""
        if self.mel_window is None or self.mel_overlap_len <= 0:
            return fade_in_mel

        device = fade_in_mel.device
        fade_in_mel = fade_in_mel.cpu()
        fade_out_mel = fade_out_mel.cpu()
        window = torch.from_numpy(self.mel_window).float()
        half = self.mel_overlap_len

        fade_in_mel[..., :half] = fade_in_mel[..., :half] * window[:half] + fade_out_mel[..., -half:] * window[half:]
        return fade_in_mel.to(device)

    # ------------------------------------------------------------------
    # Stubs for LLM_GENERATION execution type
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> list | None:
        return None

    def preprocess(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return input_ids, positions, info

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return hidden_states, info

    def make_omni_output(
        self,
        hidden_states: torch.Tensor,
        info: dict[str, Any],
        sampled_token_ids: torch.Tensor | None = None,
    ) -> OmniOutput:
        """Package decoded audio into OmniOutput."""
        audio = getattr(self, "_last_audio", None)
        sr = getattr(self, "_last_sample_rate", _SAMPLE_RATE)

        multimodal_outputs: dict[str, Any] = {}
        if audio is not None:
            # Flatten to 1D float32 numpy.
            if isinstance(audio, torch.Tensor):
                audio_flat = audio.squeeze().float().cpu()
            else:
                audio_flat = torch.tensor(audio, dtype=torch.float32).squeeze()
            multimodal_outputs["audio"] = audio_flat
            multimodal_outputs["sample_rate"] = sr
            self._last_audio = None

        return OmniOutput(
            hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def on_request_finish(self, req_id: str) -> None:
        with self._stream_lock:
            self._stream_state.pop(req_id, None)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Any) -> None:
        """Load flow and vocoder weights from glm-4-voice-decoder checkpoint.

        The decoder checkpoint contains:
        - ``config.yaml``: HyperPyYAML config defining flow + hift modules
        - ``flow.pt``: Flow matching model state dict
        - ``hift.pt``: HiFi-T vocoder state dict
        """
        # Attempt to find decoder directory.  The weights iterator comes from
        # the model loader, but for the decoder stage the model path should
        # point to the decoder checkpoint (configured in deploy YAML).
        decoder_dir = self._find_decoder_dir()
        if decoder_dir is None:
            logger.warning("GLM-4-Voice decoder directory not found. Decoder will not produce audio.")
            return

        config_path = decoder_dir / "config.yaml"
        flow_path = decoder_dir / "flow.pt"
        hift_path = decoder_dir / "hift.pt"

        if not config_path.exists():
            logger.error("Missing config.yaml in %s", decoder_dir)
            return

        # Load config via HyperPyYAML.
        try:
            from hyperpyyaml import load_hyperpyyaml
        except ImportError:
            logger.error("hyperpyyaml is required for GLM-4-Voice decoder. Install with: pip install HyperPyYAML")
            return

        # Add CosyVoice to sys.path so HyperPyYAML can resolve class refs.
        cosyvoice_paths = [
            str(decoder_dir.parent),
            str(decoder_dir),
        ]
        for p in cosyvoice_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

        with open(config_path) as f:
            scratch_configs = load_hyperpyyaml(f)

        self.flow = scratch_configs["flow"]
        self.hift = scratch_configs["hift"]

        if flow_path.exists():
            flow_state = torch.load(flow_path, map_location=self.device, weights_only=True)
            self.flow.load_state_dict(flow_state)
            logger.info("Loaded flow weights from %s", flow_path)

        if hift_path.exists():
            hift_state = torch.load(hift_path, map_location=self.device, weights_only=True)
            self.hift.load_state_dict(hift_state)
            logger.info("Loaded HiFi-T weights from %s", hift_path)

        self.flow.to(device=self.device, dtype=self.dtype)
        self.hift.to(device=self.device, dtype=self.dtype)
        self.flow.eval()
        self.hift.eval()

        # Compute streaming constants.
        input_frame_rate = getattr(self.flow, "input_frame_rate", 12.5)
        self.mel_overlap_len = int(self.token_overlap_len / input_frame_rate * _SAMPLE_RATE / _MEL_HOP_SIZE)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self.source_cache_len = int(self.mel_cache_len * _MEL_HOP_SIZE)

        logger.info(
            "GLM-4-Voice decoder ready: mel_overlap=%d, source_cache=%d",
            self.mel_overlap_len,
            self.source_cache_len,
        )

    def _find_decoder_dir(self) -> Path | None:
        """Locate the decoder checkpoint directory."""
        # Try common locations.
        candidates = [
            Path("glm-4-voice-decoder"),
            Path.home() / ".cache/huggingface/hub/models--THUDM--glm-4-voice-decoder/snapshots",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                # If it's a snapshots dir, find the latest snapshot.
                if candidate.name == "snapshots":
                    subs = sorted(candidate.iterdir())
                    if subs:
                        return subs[-1]
                return candidate

        # Try huggingface_hub download.
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(_GLM4_VOICE_DECODER_REPO, local_files_only=False)
            return Path(path)
        except Exception:
            logger.debug("Could not download decoder from HuggingFace.")

        return None


_GLM4_VOICE_DECODER_REPO = "THUDM/glm-4-voice-decoder"
