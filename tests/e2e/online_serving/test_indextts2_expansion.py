# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online serving tests for IndexTTS2 via /v1/audio/speech endpoint.

Two-stage pipeline: GPT AR → S2Mel + BigVGAN. Output is 22050 Hz mono WAV.
async_chunk=false, so streaming returns the full audio at once.
"""

from __future__ import annotations

import base64
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.tts]

MODEL = "IndexTeam/IndexTTS-2"


@pytest.fixture(scope="session")
def ref_audio_data_url() -> str:
    """Generate a synthetic reference audio clip as base64 data URL.

    Real-model tests should set INDEXTTS2_REF_AUDIO to a file path.
    """
    env_path = os.environ.get("INDEXTTS2_REF_AUDIO")
    if env_path and os.path.isfile(env_path):
        with open(env_path, "rb") as f:
            data = f.read()
        return f"data:audio/wav;base64,{base64.b64encode(data).decode('ascii')}"

    import io

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="wav")
    return f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def get_prompt(prompt_type="text"):
    prompts = {
        "text": "Hello, this is a short voice cloning demo.",
        "chinese": "你好，这是一段简单的语音合成测试。",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("indextts2.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="indextts2",
    )
]


@hardware_test(res={"cuda": "L40"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Test voice_clone mode via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text + reference audio
    Output Modal: audio (22050 Hz, WAV)
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
    }

    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L40"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Test Chinese voice cloning via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text (Chinese) + reference audio
    Output Modal: audio (22050 Hz, WAV)
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("chinese"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
    }

    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L40"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_003(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Test streaming (async_chunk=false fallback) via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text + reference audio
    Output Modal: audio (22050 Hz, PCM stream)
    Input Setting: stream=True
    Datasets: single request
    """
    extra_body = {
        "ref_audio": ref_audio_data_url,
        "stream": True,
    }
    with openai_client.client.audio.speech.with_streaming_response.create(
        model=omni_server.model,
        input=get_prompt(),
        response_format="pcm",
        extra_body=extra_body,
        timeout=300.0,
    ) as response:
        chunks = [chunk for chunk in response.iter_bytes() if chunk]

    audio_bytes = b"".join(chunks)
    assert len(audio_bytes) > 0
    assert len(audio_bytes) % 2 == 0
