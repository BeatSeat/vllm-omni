# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 native duplex candidate-space sampling."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

Model = MiniCPMO45OmniForConditionalGeneration


def test_batched_rows_match_per_row_reads():
    md = SimpleNamespace(temperature=torch.tensor([0.5, 0.9, 1.3]))
    rows = Model._sampling_metadata_rows(md, "temperature", 4, 0.8)
    assert rows == [Model._sampling_metadata_value(md, "temperature", row, 0.8) for row in range(4)]


def test_candidate_distribution_matches_the_filter():
    logits = torch.randn(1, 4096, generator=torch.Generator().manual_seed(0)) * 4
    filtered = torch.softmax(Model._top_k_top_p_filter(logits.clone(), top_k=20, top_p=0.85), dim=-1)
    probs, indices = Model._duplex_top_k_top_p_candidates(logits.clone(), top_k=20, top_p=0.85)
    dense = torch.zeros_like(filtered)
    dense[0].scatter_(0, indices[0], probs[0])
    support = filtered[0] > 0
    assert torch.equal(support, dense[0] > 0)
    assert torch.allclose(filtered[0][support], dense[0][support], atol=1e-5)

