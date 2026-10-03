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


@pytest.mark.cpu
def test_prefetched_frames_are_chunked_and_keyed_by_append() -> None:
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.sessions, runtime._prefetched_vision = {}, {}
    runtime.processor = SimpleNamespace(process_image=lambda *args, **kwargs: None)
    runtime.stage_model = runtime.thinker = SimpleNamespace(config=SimpleNamespace(vision_batch_size=2))
    runtime._decode_video_frames_payload = lambda payload: list(payload["video_frames"])
    runtime._preprocess_frames = lambda frames: [f"p{frame}" for frame in frames]
    calls: list[list[str]] = []

    def encode(items: list[str]) -> list[list[str]]:
        calls.append(items)
        return [[f"e{item}"] for item in items]

    runtime._encode_processed_vision_batch = encode
    appends = [
        {"session_id": f"s{i}", "epoch": 0, "seq": 1, "payload": {"video_frames": ["a", "b", "c"][: i + 1]}}
        for i in range(3)
    ]
    runtime.prefetch_vision(appends)
    assert calls == [["pa", "pa", "pb"], ["pa", "pb", "pc"]]  # vision_batch_size appends per tower call
    other_payload = {**appends[1], "payload": {"video_frames": ["x", "y"]}}
    assert runtime.take_prefetched_vision(other_payload) is None  # a changed payload is not reused
    assert runtime.take_prefetched_vision(appends[1]) is None  # and the entry is gone
    assert runtime.take_prefetched_vision({**appends[2], "seq": "2"}) is None  # another append
    assert runtime.frame_kwargs(appends[2], appends[2]["payload"]) == {"encoded_frames": [["epa"], ["epb"], ["epc"]]}
    assert runtime.frame_kwargs(appends[2], appends[2]["payload"]) == {"video_frames": ["a", "b", "c"]}
