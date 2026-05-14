# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E Offline inference tests for GLM-TTS voice cloning.

GLM-TTS is a zero-shot voice cloning model that always requires ref_audio.
There is no text-only / non-clone inference path in the official implementation.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = os.environ.get("GLM_TTS_MODEL_PATH", "zai-org/GLM-TTS")
REF_TEXT = "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"

DEPLOY_CONFIG = get_deploy_config_path("glm_tts.yaml")

REFERENCE_PROMPT_WAV_PATH = get_asset_path("glm_tts/jiayan_zh.wav")

ASYNC_CHUNK_MODES = [
    pytest.param(False, id="sync"),
    pytest.param(True, id="async_chunk"),
]


def _load_ref_audio() -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(REFERENCE_PROMPT_WAV_PATH), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _concat_audio(audio_val) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        tensors = [torch.as_tensor(t).float().reshape(-1) for t in audio_val if t is not None]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).cpu().numpy().astype(np.float32, copy=False)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.float().cpu().numpy().reshape(-1)
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


def _get_deploy_config(*, async_chunk: bool) -> str:
    """Build deploy config with explicit sync/async mode and eager execution."""
    return modify_stage_config(
        DEPLOY_CONFIG,
        updates={
            "async_chunk": async_chunk,
            "stages": {
                0: {
                    "enforce_eager": True,
                    "async_scheduling": bool(async_chunk),
                },
                1: {
                    "enforce_eager": True,
                },
            },
        },
    )


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("async_chunk", ASYNC_CHUNK_MODES)
def test_offline_voice_clone_zh(async_chunk: bool) -> None:
    """
    Test voice cloning offline inference.
    Deploy Setting: glm_tts.yaml with explicit sync/async mode, enforce_eager
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio

    Uses the official jiayan_zh.wav reference audio from the upstream
    GLM-TTS repository (real human speech matching REF_TEXT).
    """
    synth_text = "我捡到一只超可爱的流浪猫。我给这只小猫取了一个名字，叫丁满。"
    prompt_audio = _load_ref_audio()
    from tests.helpers.runtime import OmniRunner

    with OmniRunner(
        MODEL,
        stage_configs_path=_get_deploy_config(async_chunk=async_chunk),
        stage_init_timeout=600,
    ) as omni_runner:
        outputs = omni_runner.omni.generate(
            [
                {
                    "prompt": synth_text,
                    "multi_modal_data": {"audio": prompt_audio},
                    "modalities": ["audio"],
                    "mm_processor_kwargs": {"prompt_text": REF_TEXT},
                }
            ],
            omni_runner.get_default_sampling_params_list(),
        )

        assert outputs, "No outputs returned"
        audio_mm = outputs[0].multimodal_output
        assert "audio" in audio_mm, "No audio output found"
        audio = _concat_audio(audio_mm["audio"])
        assert audio.size > 0, "Generated audio is empty"
