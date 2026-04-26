# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.glm_tts.glm_tts import (
    GLMTTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.glm_tts.sampling import sample_ras_one

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTokenizer:
    def encode(self, text: str):
        return list(range(len(text)))


class _FakeTextFrontend:
    def text_normalize(self, text: str | None):
        return text


def test_glm_tts_text_only_prompt_len_includes_boa() -> None:
    assert (
        GLMTTSForConditionalGeneration.estimate_prompt_len_from_text(
            text="hello",
            tokenizer=_FakeTokenizer(),
            text_frontend=_FakeTextFrontend(),
        )
        == 6
    )


def test_glm_tts_voice_clone_prompt_len_matches_prefill_layout() -> None:
    assert (
        GLMTTSForConditionalGeneration.estimate_prompt_len_from_text(
            text="tts",
            prompt_text="ref",
            prompt_speech_token_len=4,
            tokenizer=_FakeTokenizer(),
            text_frontend=_FakeTextFrontend(),
        )
        == 12
    )


def test_glm_tts_ras_fallback_masks_repeated_top_token() -> None:
    sampled = sample_ras_one(
        torch.tensor([10.0, 0.0]),
        decoded_tokens=[0] * 10,
        top_p=1.0,
        top_k=1,
        win_size=10,
        tau_r=0.1,
        temperature=1.0,
        generator=None,
    )

    assert sampled == 1
