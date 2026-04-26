# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocoder loading and mel-to-audio conversion for GLM-TTS.

Supports:
  - HiFT (24kHz) from cosyvoice hifigan checkpoint
  - Vocos2D JIT (32kHz) from TorchScript checkpoint

Extracted from glm_tts_dit_wrapper.py to keep file sizes under 800 lines.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# HiFT vocoder (24kHz)
# ---------------------------------------------------------------------------


class HiFTWrapper:
    """Thin wrapper around CosyVoice HiFTGenerator."""

    def __init__(self, state_dict: dict[str, Any], device: torch.device):
        self.device = device
        self.sample_rate = 24000
        from cosyvoice.hifigan_cosy2.f0_predictor import ConvRNNF0Predictor
        from cosyvoice.hifigan_cosy2.generator import HiFTGenerator

        f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
        self.model = HiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=24000,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ],
            lrelu_slope=0.1,
            audio_limit=0.99,
            f0_predictor=f0_predictor,
        ).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.to(self.device)
        with torch.no_grad():
            audio, _ = self.model.inference(mel)
        return audio


def load_hift(ckpt_path: str, device: torch.device) -> HiFTWrapper:
    """Load HiFT vocoder from checkpoint."""
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    return HiFTWrapper(state_dict, device)


# ---------------------------------------------------------------------------
# Vocos2D JIT vocoder (32kHz)
# ---------------------------------------------------------------------------

_MEL_LOGDIFF = -7.847762537473608


class Vocos2DWrapper:
    """Thin wrapper around Vocos2D TorchScript model."""

    def __init__(self, ckpt_path: str, device: torch.device):
        self.device = device
        self.gen_model = torch.jit.load(ckpt_path, map_location=device)
        self.gen_model.eval()
        self._mel_logdiff = _MEL_LOGDIFF

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.size(-1) == 1:
            mel = torch.cat([mel, mel], dim=-1)
        xs_mel = mel.transpose(-1, -2) + self._mel_logdiff
        xs = self.gen_model(xs_mel.to(self.device))
        if xs.ndim == 2:
            xs = xs.unsqueeze(1)
        return xs


def load_vocos2d_jit(ckpt_path: str, device: torch.device) -> Vocos2DWrapper:
    """Load Vocos2D JIT vocoder from TorchScript checkpoint."""
    return Vocos2DWrapper(ckpt_path, device)


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------


def load_vocoder(
    model_root: str,
    device: torch.device,
    sample_rate: int = 24000,
) -> tuple[Any | None, int]:
    """Try to load the best available vocoder.

    Returns:
        (vocoder, actual_sample_rate) or (None, sample_rate) on failure.
    """
    # Try HiFT first for 24kHz
    if sample_rate == 24000:
        hift_path = os.path.join(model_root, "hift", "hift.pt")
        if os.path.isfile(hift_path):
            try:
                vocoder = load_hift(hift_path, device)
                logger.info("Loaded HiFT vocoder from %s", hift_path)
                return vocoder, 24000
            except Exception as e:
                logger.warning("Failed to load HiFT: %s", e)

    # Try Vocos2D JIT
    vocos_jit_path = os.path.join(model_root, "vocos2d", "generator_jit.ckpt")
    if os.path.isfile(vocos_jit_path):
        try:
            vocoder = load_vocos2d_jit(vocos_jit_path, device)
            logger.info("Loaded Vocos2D JIT vocoder (32kHz)")
            return vocoder, 32000
        except Exception as e:
            logger.warning("Failed to load Vocos2D JIT: %s", e)

    logger.warning("No vocoder available")
    return None, sample_rate


def mel_to_audio(
    vocoder: Any | None,
    mel: torch.Tensor,
) -> torch.Tensor:
    """Convert mel-spectrogram to audio waveform.

    Args:
        vocoder: Vocoder instance (HiFTWrapper, Vocos2DWrapper, or None).
        mel: Mel-spectrogram [B, T, mel_dim].

    Returns:
        Audio waveform [B, 1, samples].
    """
    # Transpose to [B, mel_dim, T] for vocoder
    mel = mel.transpose(1, 2)

    if vocoder is not None:
        if callable(vocoder) and not hasattr(vocoder, "decode"):
            audio = vocoder(mel)
        else:
            audio = vocoder.decode(mel)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
    else:
        logger.warning("No vocoder available, returning silence")
        batch_size = mel.shape[0]
        n_samples = mel.shape[2] * 256
        audio = torch.zeros(batch_size, 1, n_samples, device=mel.device)

    return audio
