# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for GLM-4-Voice TTS.

Tests both sync and async_chunk modes, verifying that the model
generates valid audio output from text input.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from tests.helpers.mark import hardware_test

MODEL = "THUDM/glm-4-voice-9b"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_CONFIG = os.path.join(_THIS_DIR, "..", "..", "..", "vllm_omni", "deploy", "glm4_voice.yaml")

ASYNC_CHUNK_MODES = [
    pytest.param(False, id="sync"),
    pytest.param(True, id="async_chunk"),
]


def _get_deploy_config(async_chunk: bool = True) -> str:
    from tests.helpers.stage_config import modify_stage_config

    updates: dict = {}
    if not async_chunk:
        updates["async_chunk"] = False
    updates.setdefault("stages", {}).setdefault(0, {})["enforce_eager"] = True
    updates["stages"].setdefault(1, {})["enforce_eager"] = True

    return modify_stage_config(_DEPLOY_CONFIG, updates=updates)


def _concat_audio(audio_data) -> np.ndarray:
    """Concatenate audio from various output formats."""
    if isinstance(audio_data, torch.Tensor):
        return audio_data.float().cpu().numpy().flatten()
    if isinstance(audio_data, np.ndarray):
        return audio_data.flatten()
    if isinstance(audio_data, list):
        parts = []
        for chunk in audio_data:
            if isinstance(chunk, torch.Tensor):
                parts.append(chunk.float().cpu().numpy().flatten())
            elif isinstance(chunk, np.ndarray):
                parts.append(chunk.flatten())
        return np.concatenate(parts) if parts else np.array([])
    return np.array([])


@pytest.mark.advanced_model
@pytest.mark.tts
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("async_chunk", ASYNC_CHUNK_MODES)
def test_offline_text_to_speech_zh(async_chunk: bool) -> None:
    """Basic Chinese TTS: text input → audio output."""
    from vllm_omni import Omni
    from vllm_omni.model_executor.models.glm4_voice.glm4_voice import (
        build_glm4_voice_prompt,
    )

    synth_text = "今天天气真不错，适合出去散散步。"

    omni = Omni(
        model=MODEL,
        stage_configs_path=_get_deploy_config(async_chunk=async_chunk),
        trust_remote_code=True,
        stage_init_timeout=600,
    )
    try:
        prompt_text = build_glm4_voice_prompt(synth_text)
        outputs = omni.generate([{"prompt": prompt_text}])
    finally:
        omni.close()

    assert outputs, "No outputs returned"
    mm = outputs[0].multimodal_output
    assert mm is not None, "No multimodal output"
    assert "audio" in mm, "No audio in multimodal output"

    audio = _concat_audio(mm["audio"])
    assert audio.size > 0, "Empty audio output"

    sample_rate = mm.get("sample_rate", 22050)
    duration = audio.size / sample_rate
    assert duration > 0.5, f"Audio too short: {duration:.2f}s"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate=sample_rate)
        file_size = os.path.getsize(f.name)
        assert file_size > 1000, f"WAV file too small: {file_size} bytes"
        os.unlink(f.name)


@pytest.mark.advanced_model
@pytest.mark.tts
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_offline_text_to_speech_en() -> None:
    """English TTS test."""
    from vllm_omni import Omni
    from vllm_omni.model_executor.models.glm4_voice.glm4_voice import (
        build_glm4_voice_prompt,
    )

    synth_text = "The weather is nice today, perfect for a walk in the park."

    omni = Omni(
        model=MODEL,
        stage_configs_path=_get_deploy_config(async_chunk=True),
        trust_remote_code=True,
        stage_init_timeout=600,
    )
    try:
        prompt_text = build_glm4_voice_prompt(synth_text)
        outputs = omni.generate([{"prompt": prompt_text}])
    finally:
        omni.close()

    assert outputs, "No outputs returned"
    mm = outputs[0].multimodal_output
    assert mm is not None and "audio" in mm

    audio = _concat_audio(mm["audio"])
    assert audio.size > 0, "Empty audio output"
