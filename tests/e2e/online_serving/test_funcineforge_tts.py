# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for FunCineForge dubbing/TTS model.

Tests verify the /v1/audio/speech endpoint works correctly with
FunCineForge, using official reference audio from the FunCineForge repo.
FunCineForge requires reference audio + ref_text for voice cloning.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "FunAudioLLM/Fun-CineForge"

# Official reference audio from FunCineForge repo
REF_AUDIO_URL = "https://raw.githubusercontent.com/FunAudioLLM/FunCineForge/main/exps/data/ref.wav"

# Clue text matching official demo.jsonl en_monologue_1
REF_TEXT = (
    "A single middle-aged male speaker describes a business or "
    "construction requirement with a practical and matter-of-fact tone."
)


def get_stage_config(name: str = "funcineforge.yaml"):
    """Get the deploy config path for FunCineForge."""
    return get_deploy_config_path(name)


def get_prompt(prompt_type="en"):
    """Official demo texts from FunCineForge repo."""
    prompts = {
        "en": (
            "Every closet on a Carnival cruise ship. To make the numbers work, I needed a lot of cedar, fast and cheap."
        ),
        "zh": "这是一个电影配音模型的测试。",
    }
    return prompts.get(prompt_type, prompts["en"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config(),
            server_args=[
                "--trust-remote-code",
                "--disable-log-stats",
                "--no-async-chunk",
            ],
        ),
        id="funcineforge",
    )
]

tts_async_chunk_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config(),
            server_args=[
                "--trust-remote-code",
                "--disable-log-stats",
            ],
        ),
        id="funcineforge_async_chunk",
    )
]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_en_sync(omni_server, openai_client) -> None:
    """
    Test FunCineForge dubbing with English text via OpenAI API (sync).
    Uses official ref.wav from FunCineForge repo for voice cloning.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_async_chunk_server_params, indirect=True)
def test_funcineforge_dubbing_en_async(omni_server, openai_client) -> None:
    """
    Test FunCineForge dubbing with async_chunk streaming.
    Uses official ref.wav from FunCineForge repo for voice cloning.

    Deploy Setting: funcineforge.yaml with async_chunk: true
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio (streamed)
    Input Setting: stream=True
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": True,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_zh_sync(omni_server, openai_client) -> None:
    """
    Test FunCineForge dubbing with Chinese text via OpenAI API (sync).
    Verifies bilingual (zh_en) model handles Chinese input.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("zh"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)
