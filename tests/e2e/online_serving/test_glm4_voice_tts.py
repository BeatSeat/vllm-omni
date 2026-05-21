# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online serving tests for GLM-4-Voice TTS.

Tests the ``/v1/audio/speech`` endpoint with GLM-4-Voice-9B,
covering sync and streaming modes with concurrent requests.
"""

from __future__ import annotations

import pytest

from tests.helpers.fixtures.runtime import OmniServerParams
from tests.helpers.markers import hardware_test
from tests.helpers.utils import get_deploy_config_path

MODEL = "THUDM/glm-4-voice-9b"

_tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("glm4_voice.yaml"),
            server_args=["--trust-remote-code"],
        ),
        id="glm4_voice_async_chunk",
    ),
]


@pytest.mark.advanced_model
@pytest.mark.tts
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _tts_server_params, indirect=True)
def test_text_to_speech_sync_zh(omni_server, openai_client) -> None:
    """Sync TTS: Chinese text → WAV audio."""
    request_config = {
        "model": omni_server.model,
        "input": "今天天气真不错，适合出去散散步。",
        "stream": False,
        "timeout": 180.0,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _tts_server_params, indirect=True)
def test_text_to_speech_sync_en(omni_server, openai_client) -> None:
    """Sync TTS: English text → WAV audio."""
    request_config = {
        "model": omni_server.model,
        "input": "The weather is nice today, perfect for a walk.",
        "stream": False,
        "timeout": 180.0,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _tts_server_params, indirect=True)
def test_text_to_speech_streaming(omni_server, openai_client) -> None:
    """Streaming TTS: text → PCM stream."""
    request_config = {
        "model": omni_server.model,
        "input": "今天天气真不错。",
        "stream": True,
        "timeout": 180.0,
        "response_format": "pcm",
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _tts_server_params, indirect=True)
def test_text_to_speech_concurrent(omni_server, openai_client) -> None:
    """Concurrent requests: 3 parallel TTS requests."""
    request_config = {
        "model": omni_server.model,
        "input": "Hello, how are you today?",
        "stream": False,
        "timeout": 300.0,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config, request_num=3)
