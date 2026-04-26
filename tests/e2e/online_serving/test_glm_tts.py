# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for GLM-TTS model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
the GLM-TTS two-stage pipeline (AR + DiT).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("GLM_TTS_MODEL_PATH", "THUDM/GLM-TTS")
REF_AUDIO_URL = "https://raw.githubusercontent.com/zai-org/GLM-TTS/main/examples/prompt/jiayan_zh.wav"
REF_TEXT = "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"

DEPLOY_CONFIG = get_deploy_config_path("glm_tts.yaml")

EXTRA_ARGS = [
    "--trust-remote-code",
    "--enforce-eager",
    "--disable-log-stats",
]

tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            server_args=EXTRA_ARGS,
            stage_init_timeout=300,
        ),
        id="glm_tts",
    )
]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_basic_tts_zh(omni_server, openai_client) -> None:
    """
    Test basic Chinese TTS generation.
    Deploy Setting: glm_tts.yaml (sync two-stage)
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": "你好，这是一个语音合成测试。",
        "stream": False,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_basic_tts_long_text(omni_server, openai_client) -> None:
    """
    Test TTS with a longer Chinese text.
    Deploy Setting: glm_tts.yaml (sync two-stage)
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": "每次熬煮小米粥时，奶奶习惯加入一小把西洋参片，淡淡的药香融入粥中，格外温暖。",
        "stream": False,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_zh(omni_server, openai_client) -> None:
    """
    Test voice cloning TTS with Chinese text.
    Deploy Setting: glm_tts.yaml (sync two-stage)
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False

    Uses a public reference audio from the official GLM-TTS repo.
    """
    request_config = {
        "model": omni_server.model,
        "input": "我捡到一只超可爱的流浪猫。我给它取了一个名字，叫丁满。",
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_models_endpoint(omni_server, openai_client) -> None:
    """Test the /v1/models endpoint returns loaded model."""
    models = openai_client.client.models.list()
    assert len(models.data) > 0
