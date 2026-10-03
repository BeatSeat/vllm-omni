# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused packed SigLIP layers match the eager packed encode."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5 import vision_fused
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    SiglipVisionConfig,
    SiglipVisionTransformer,
)

pytestmark = [pytest.mark.core_model]


def _tower() -> SiglipVisionTransformer:
    torch.manual_seed(0)
    config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=72,
        num_hidden_layers=3,
        num_attention_heads=4,
        image_size=28,
        patch_size=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    vpm = SiglipVisionTransformer(config).eval()
    for module in vpm.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.normal_(module.weight, mean=1.0, std=0.2)
            torch.nn.init.normal_(module.bias, std=0.2)
    return vpm


@pytest.mark.cpu
def test_fused_layers_match_eager_packed_layers() -> None:
    vpm = _tower()
    seq_groups = [(0, 3, 16), (48, 2, 15)]
    hidden = torch.randn((78, 32), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        expected = hidden.clone()
        for layer in vpm.encoder.layers:
            expected = layer.forward_packed(expected, seq_groups)
        expected = vpm.post_layernorm(expected)
        fused = vision_fused.encode_packed_fused(vpm, hidden.clone(), seq_groups)
    torch.testing.assert_close(fused, expected, rtol=1e-5, atol=1e-5)
