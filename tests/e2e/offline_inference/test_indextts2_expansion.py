# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for IndexTTS2 two-stage pipeline.

Stage 0 (GPT AR) → Stage 1 (S2Mel + BigVGAN). Output is 22050 Hz mono WAV.
Reference audio is required for voice cloning.
"""

from __future__ import annotations

import os

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import Omni
from vllm_omni.model_executor.models.indextts2.prompt_utils import (
    build_indextts2_prefill_prompt_ids,
)

MODEL_NAME = "IndexTeam/IndexTTS-2"
STAGE_CONFIG = get_deploy_config_path("indextts2.yaml")

_OMNI_RUNNER_PARAM = (
    MODEL_NAME,
    STAGE_CONFIG,
)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

SAMPLE_RATE = 22050

DEFAULT_SAMPLING = SamplingParams(
    temperature=0.8,
    top_p=0.8,
    top_k=30,
    max_tokens=1500,
    repetition_penalty=10.0,
    stop_token_ids=[8193],
    seed=42,
    detokenize=False,
)


def _audio_from_mm(mm: dict) -> torch.Tensor | None:
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if isinstance(audio, list):
        chunks = [chunk.reshape(-1) for chunk in audio if isinstance(chunk, torch.Tensor) and chunk.numel() > 0]
        return torch.cat(chunks, dim=0) if chunks else None
    return audio if isinstance(audio, torch.Tensor) else None


def _sample_rate_from_mm(mm: dict) -> int:
    sr = mm.get("sr")
    if isinstance(sr, list):
        sr = sr[-1] if sr else None
    if hasattr(sr, "item"):
        return int(sr.item())
    return int(sr) if sr is not None else SAMPLE_RATE


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
    """Provide reference audio for voice cloning.

    Uses a synthetic sine wave when no real audio is available, to avoid
    external network dependencies. Real-model tests should provide
    INDEXTTS2_REF_AUDIO env var pointing to a real clip.
    """
    env_path = os.environ.get("INDEXTTS2_REF_AUDIO")
    if env_path and os.path.isfile(env_path):
        return env_path

    import numpy as np
    import soundfile as sf

    cache_dir = tmp_path_factory.mktemp("indextts2_ref")
    target = cache_dir / "ref_sine.wav"
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(target), sine, sr)
    return str(target)


def _build_request(text: str, ref_audio: str) -> dict:
    return {
        "prompt_token_ids": build_indextts2_prefill_prompt_ids(MODEL_NAME, text),
        "additional_information": {
            "text": [text],
            "voice": [ref_audio],
        },
    }


def _collect_audio(omni: Omni, request: dict) -> tuple[torch.Tensor, int]:
    for omni_out in omni.generate(request, DEFAULT_SAMPLING):
        mm = omni_out.multimodal_output
        assert mm is not None, "Expected multimodal_output"
        audio = _audio_from_mm(mm)
        assert audio is not None, "Expected audio output"
        return audio.cpu(), _sample_rate_from_mm(mm)
    raise AssertionError("No outputs received")


@hardware_test(res={"cuda": "L40"})
def test_indextts2_chinese(omni_runner: OmniRunner, ref_audio_path) -> None:
    """Chinese TTS produces non-empty 22050 Hz audio."""
    req = _build_request("你好，这是一段语音合成测试。", ref_audio_path)
    audio, sr = _collect_audio(omni_runner.omni, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert torch.any(audio != 0).item()


@hardware_test(res={"cuda": "L40"})
def test_indextts2_english(omni_runner: OmniRunner, ref_audio_path) -> None:
    """English TTS produces non-empty audio."""
    req = _build_request("Hello, this is a voice synthesis test.", ref_audio_path)
    audio, sr = _collect_audio(omni_runner.omni, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert torch.any(audio != 0).item()


@hardware_test(res={"cuda": "L40"})
def test_indextts2_deterministic(omni_runner: OmniRunner, ref_audio_path) -> None:
    """Same seed produces identical waveforms."""
    req = _build_request("Reproducible output test.", ref_audio_path)
    audio1, _ = _collect_audio(omni_runner.omni, req)
    audio2, _ = _collect_audio(omni_runner.omni, req)

    assert audio1.shape == audio2.shape
    assert torch.allclose(audio1, audio2, atol=1e-4)
