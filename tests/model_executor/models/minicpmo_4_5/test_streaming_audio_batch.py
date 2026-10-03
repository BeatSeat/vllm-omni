# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-session batched streaming audio encoder against the per-session path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
)
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder import (
    StreamingAudioChunk,
    StreamingAudioKVCache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

N_MELS = 16
POOL = 5


class _Thinker:
    """The audio half of the thinker: real modules, the real methods under test."""

    get_audio_embedding_streaming = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming
    get_audio_embedding_streaming_batch = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming_batch
    supports_streaming_audio_batch = MiniCPMO45OmniLLMForConditionalGeneration.supports_streaming_audio_batch
    _get_feat_extract_output_lengths = MiniCPMO45OmniLLMForConditionalGeneration._get_feat_extract_output_lengths

    def __init__(self, *, attn_implementation: str = "sdpa", max_positions: int = 40, dtype=torch.float32) -> None:
        torch.manual_seed(0)
        config = WhisperConfig(
            num_mel_bins=N_MELS,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=128,  # the projection takes ffn // 4 == d_model, as in the checkpoint
            max_source_positions=max_positions,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = attn_implementation
        self.apm = MiniCPMWhisperEncoder(config).eval().to(dtype)
        self.audio_projection_layer = MultiModalProjector(in_dim=config.encoder_ffn_dim // 4, out_dim=24).to(dtype)
        self.audio_avg_pooler = nn.AvgPool1d(POOL, stride=POOL)
        self.audio_encoder_layer = -1
        self.audio_past_key_values = None
        self.config = SimpleNamespace(audio_pool_step=POOL, duplex_audio_kv_page_positions=16)


def _mel(frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, N_MELS, frames), generator=generator)


def _sequential(thinker: _Thinker, cache, mel: torch.Tensor, prefix: int, *, use_extra_context: bool = True):
    """One unit through the per-session path, with that session's cache swapped in."""
    thinker.audio_past_key_values = cache
    with torch.no_grad():
        nested = thinker.get_audio_embedding_streaming(
            {"audio_features": mel, "audio_feature_lens": [torch.tensor([mel.shape[-1]])]},
            use_extra_context=use_extra_context,
            prefix_extra_frames=prefix,
            suffix_extra_frames=2,
        )
    embeds = torch.cat([t for row in nested for t in row]) if nested else None
    return embeds, thinker.audio_past_key_values


def _legacy_layer(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer_cache = cache.self_attention_cache.layers[layer]
    return layer_cache.keys, layer_cache.values


# Per round, the mel frames of each session's unit (None: sits the round out).
# 22 frames at chunk 0 and 24 later mirror 1020 ms / 1040 ms at 1/10 scale; the
# odd lengths make conv2 read a padded column and 25 frames give an 11-position
# unit that the pooler cannot divide. max_positions=40 forces a reset per cycle.
_SCHEDULE = [
    [22, 22, 23, 21],
    [24, None, 24, 25],
    [24, 24, None, 24],
    [24, 23, 24, 24],
    [24, 24, 24, None],
    [25, 24, 23, 24],
]


def test_batched_rounds_match_sequential_sessions() -> None:
    thinker = _Thinker(attn_implementation="sdpa")
    sessions = len(_SCHEDULE[0])
    legacy_caches: list[object | None] = [None] * sessions
    batched_caches: list[StreamingAudioKVCache | None] = [None] * sessions
    chunk_index = [0] * sessions
    resets = 0

    for round_index, frames_per_session in enumerate(_SCHEDULE):
        chunks, active, expected = [], [], []
        for session, frames in enumerate(frames_per_session):
            if frames is None:
                continue
            mel = _mel(frames, seed=100 * round_index + session)
            prefix = 0 if chunk_index[session] == 0 else 2
            before = batched_caches[session]
            embeds, legacy_caches[session] = _sequential(thinker, legacy_caches[session], mel, prefix)
            expected.append(embeds)
            chunks.append(
                StreamingAudioChunk(features=mel, cache=before, prefix_extra_frames=prefix, suffix_extra_frames=2)
            )
            active.append(session)
            chunk_index[session] += 1

        idle = {s: (batched_caches[s], batched_caches[s].length if batched_caches[s] else 0) for s in range(sessions)}
        outputs, caches = thinker.get_audio_embedding_streaming_batch(chunks)

        for session, output, cache, reference in zip(active, outputs, caches, expected, strict=True):
            assert output is not None and reference is not None
            torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)
            if cache is not batched_caches[session] and batched_caches[session] is not None:
                resets += 1
            batched_caches[session] = cache
            legacy_length = legacy_caches[session].self_attention_cache.get_seq_length()
            assert cache.length == legacy_length
            for layer in range(len(thinker.apm.layers)):
                keys, values = _legacy_layer(legacy_caches[session], layer)
                torch.testing.assert_close(cache.head_view(cache.keys(layer)[: cache.length]), keys)
                torch.testing.assert_close(cache.head_view(cache.values(layer)[: cache.length]), values)
        for session in set(range(sessions)) - set(active):
            # Not in the batch: the same cache object, the same committed length.
            cache, length = idle[session]
            assert batched_caches[session] is cache
            assert (cache.length if cache is not None else 0) == length

    assert resets >= 2, "the schedule must cross the max_source_positions reset"
