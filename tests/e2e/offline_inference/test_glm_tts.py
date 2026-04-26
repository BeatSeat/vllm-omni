# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Offline inference tests for GLM-TTS model with text input and audio output.

Tests both no-clone (text-only) and voice-clone modes via the offline
OmniRunner / OmniRunnerHandler pipeline.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = os.environ.get("GLM_TTS_MODEL_PATH", "THUDM/GLM-TTS")
REF_AUDIO_URL = "https://raw.githubusercontent.com/zai-org/GLM-TTS/main/examples/prompt/jiayan_zh.wav"
REF_TEXT = "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"

DEPLOY_CONFIG = get_deploy_config_path("glm_tts.yaml")


def _get_deploy_config():
    """Build deploy config with enforce_eager and conservative memory settings."""
    return modify_stage_config(
        DEPLOY_CONFIG,
        updates={
            "async_chunk": False,
            "stages": {
                0: {
                    "enforce_eager": True,
                    "async_scheduling": False,
                    "gpu_memory_utilization": 0.3,
                },
                1: {
                    "enforce_eager": True,
                    "gpu_memory_utilization": 0.3,
                },
            },
        },
    )


tts_runner_params = [
    pytest.param(
        (MODEL, _get_deploy_config()),
        id="glm_tts",
    )
]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_runner_params, indirect=True)
def test_offline_tts_zh(omni_runner, omni_runner_handler) -> None:
    """
    Test basic Chinese TTS offline inference (no voice cloning).
    Deploy Setting: glm_tts.yaml (sync two-stage, enforce_eager)
    Input Modal: text
    Output Modal: audio
    """
    request_config = {
        "input": "你好，这是一个语音合成测试。",
    }
    omni_runner_handler.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_runner_params, indirect=True)
def test_offline_tts_long_text(omni_runner, omni_runner_handler) -> None:
    """
    Test TTS with longer Chinese text (no voice cloning).
    Deploy Setting: glm_tts.yaml (sync two-stage, enforce_eager)
    Input Modal: text
    Output Modal: audio
    """
    request_config = {
        "input": "每次熬煮小米粥时，奶奶习惯加入一小把西洋参片，淡淡的药香融入粥中，格外温暖。",
    }
    omni_runner_handler.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_runner_params, indirect=True)
def test_offline_voice_clone_zh(omni_runner, omni_runner_handler) -> None:
    """
    Test voice cloning offline inference.
    Deploy Setting: glm_tts.yaml (sync two-stage, enforce_eager)
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio

    Uses a public reference audio from the official GLM-TTS repo.
    """
    request_config = {
        "input": "我捡到一只超可爱的流浪猫。我给它取了一个名字，叫丁满。",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    omni_runner_handler.send_audio_speech_request(request_config)
