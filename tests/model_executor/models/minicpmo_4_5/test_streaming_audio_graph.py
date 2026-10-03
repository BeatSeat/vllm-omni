# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for streaming-audio CUDA graph bucket helpers."""

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder import StreamingAudioChunk
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder_graph import (
    normalize_buckets,
    row_is_steady,
    select_bucket,
    steady_unit_length,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_graph_bucket_helpers() -> None:
    assert select_bucket(0, (500, 1000, 1500)) == 500
    assert select_bucket(1501, (500, 1000, 1500)) is None
    assert normalize_buckets([1000, 500, 500, 0]) == (500, 1000)
    assert steady_unit_length(24) == 10
    chunk = StreamingAudioChunk(
        features=torch.zeros(1, 16, 24), cache=object(), prefix_extra_frames=2, suffix_extra_frames=2
    )
    assert row_is_steady(chunk, expected_frames=24)
    assert not row_is_steady(
        StreamingAudioChunk(
            features=torch.zeros(1, 16, 24), cache=object(), prefix_extra_frames=0, suffix_extra_frames=2
        ),
        expected_frames=24,
    )
