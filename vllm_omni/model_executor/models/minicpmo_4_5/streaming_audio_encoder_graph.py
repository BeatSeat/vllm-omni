# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs of MiniCPM-o's streaming Whisper encoder for the steady unit shape."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from .streaming_audio_encoder import (
    DEFAULT_KV_PAGE_POSITIONS,
    StreamingAudioChunk,
    StreamingAudioKVCache,
    _attend,
    _extra_context_trim,
    encode_streaming_audio_batch,
)

if TYPE_CHECKING:
    from torch.cuda import CUDAGraph

_NEG_INF = float("-inf")

# stage0.py builds every unit but a session's first with these trims.
STEADY_PREFIX_EXTRA_FRAMES = 2
STEADY_SUFFIX_EXTRA_FRAMES = 2

DEFAULT_GRAPH_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
# Shared storage scales with the largest bucket only. 900 stays under ~3 GiB
# at batch 32 on MiniCPM-o 4.5 (bf16); history past it falls back to eager.
DEFAULT_GRAPH_CACHE_BUCKETS: tuple[int, ...] = (250, 500, 750, 900)


def batch_sizes_for_sessions(max_sessions: int) -> tuple[int, ...]:
    """Powers of two below ``max_sessions``, plus ``max_sessions`` itself."""
    top = max(1, int(max_sessions))
    sizes = [1 << i for i in range(top.bit_length()) if (1 << i) < top]
    return normalize_buckets([*sizes, top])


def select_bucket(value: int, buckets: Sequence[int]) -> int | None:
    return min((bucket for bucket in buckets if value <= bucket), default=None)


def normalize_buckets(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(v) for v in values if int(v) > 0}))


def steady_unit_length(unit_frames: int) -> int:
    conv_length = (int(unit_frames) - 1) // 2 + 1
    trim = _extra_context_trim(STEADY_PREFIX_EXTRA_FRAMES) + _extra_context_trim(STEADY_SUFFIX_EXTRA_FRAMES)
    return conv_length - trim


def steady_pooled_length(unit_frames: int, unit_length: int, pool_step: int) -> int:
    nominal_cap = ((unit_frames - 1) // 2 + 1 - pool_step) // pool_step + 1
    return min(nominal_cap, unit_length // pool_step)


def row_is_steady(chunk: StreamingAudioChunk, *, expected_frames: int) -> bool:
    if chunk.cache is None or not chunk.use_extra_context:
        return False
    if int(chunk.prefix_extra_frames) != STEADY_PREFIX_EXTRA_FRAMES:
        return False
    if int(chunk.suffix_extra_frames) != STEADY_SUFFIX_EXTRA_FRAMES:
        return False
    frames = int(chunk.features.shape[-1])
    if frames != expected_frames:
        return False
    return chunk.feature_length is None or int(chunk.feature_length) == frames


@dataclass(slots=True)
class _Group:
    index: int
    cache_in: StreamingAudioKVCache
    cache_out: StreamingAudioKVCache
    past: int


class StreamingAudioGraphEncoder:
    """Pre-captured CUDA graphs of the streaming encoder's steady unit shape."""

    def __init__(
        self,
        encoder: nn.Module,
        projection: nn.Module,
        pooler: nn.Module,
        *,
        unit_frames: int,
        pool_step: int,
        batch_sizes: Sequence[int] = DEFAULT_GRAPH_BATCH_SIZES,
        cache_buckets: Sequence[int] = DEFAULT_GRAPH_CACHE_BUCKETS,
        page_positions: int = DEFAULT_KV_PAGE_POSITIONS,
        pinned_h2d: bool = False,
    ) -> None:
        weight = encoder.conv1.weight
        if weight.device.type != "cuda":
            raise ValueError("StreamingAudioGraphEncoder requires a CUDA encoder")
        self.encoder = encoder
        self.projection = projection
        self.pooler = pooler
        self.device = weight.device
        self.dtype = weight.dtype
        self.pool_step = int(pool_step)
        self.page_positions = int(page_positions)
        self.pinned_h2d = bool(pinned_h2d)
        self.num_layers = len(encoder.layers)
        self.embed_dim = int(encoder.config.d_model)
        self.num_heads = int(encoder.config.encoder_attention_heads)
        self.num_mels = int(encoder.config.num_mel_bins)
        self.max_positions = int(encoder.embed_positions.weight.shape[0])
        self.implementation = getattr(encoder.config, "_attn_implementation", None) or "eager"
        self.unit_frames = int(unit_frames)
        self.unit_length = steady_unit_length(self.unit_frames)
        if self.unit_length <= 0:
            raise ValueError(f"unit_frames={unit_frames} is too small for the steady prefix/suffix trim")
        self.pooled_length = steady_pooled_length(self.unit_frames, self.unit_length, self.pool_step)
        if self.pooled_length <= 0:
            raise ValueError(
                f"unit_frames={unit_frames} pool_step={pool_step} leaves no pooled output for the steady unit"
            )
        self.batch_sizes = normalize_buckets(batch_sizes)
        self.cache_buckets = normalize_buckets(cache_buckets)
        if not self.batch_sizes or not self.cache_buckets:
            raise ValueError("StreamingAudioGraphEncoder needs at least one batch size and one cache bucket")
        self.max_batch = max(self.batch_sizes)
        self.max_cache_bucket = max(self.cache_buckets)
        self.max_total = self.max_cache_bucket + self.unit_length
        self._graphs: dict[tuple[int, int], CUDAGraph] = {}
        self._pooled: dict[tuple[int, int], torch.Tensor] = {}
        self._mel_storage: torch.Tensor | None = None
        self._cache_storage: torch.Tensor | None = None
        self._mask_storage: torch.Tensor | None = None
        self._position_ids_storage: torch.Tensor | None = None

    def _allocate_storage(self) -> None:
        self._mel_storage = torch.zeros(
            (self.max_batch, self.num_mels, self.unit_frames), dtype=self.dtype, device=self.device
        )
        self._cache_storage = torch.zeros(
            (self.num_layers, 2, self.max_batch, self.max_total, self.embed_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._mask_storage = torch.zeros(
            (self.max_batch, 1, self.unit_length, self.max_total), dtype=self.dtype, device=self.device
        )
        self._position_ids_storage = torch.zeros(
            (self.max_batch, self.unit_length), dtype=torch.long, device=self.device
        )

    def _views(self, batch: int, cache_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._cache_storage is not None
        total = cache_len + self.unit_length
        return (
            self._mel_storage[:batch],
            self._cache_storage[:, :, :batch, :total, :],
            self._mask_storage[:batch, :, :, :total],
            self._position_ids_storage[:batch],
        )

    def capture(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot capture the streaming audio encoder graph mid-capture")
        from vllm.platforms import current_platform

        self._allocate_storage()
        pool = current_platform.get_global_graph_pool()
        for batch in self.batch_sizes:
            for cache_len in self.cache_buckets:
                self._capture_one(batch, cache_len, pool)

    def _capture_one(self, batch: int, cache_len: int, pool: object) -> None:
        key = (batch, cache_len)
        mel, cache, mask, position_ids = self._views(batch, cache_len)

        def forward():
            return self._forward(mel, cache, mask, position_ids, cache_len)

        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                warmup = forward()
        current_stream.wait_stream(warmup_stream)
        del warmup

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=pool):
            pooled = forward()
        self._graphs[key] = graph
        self._pooled[key] = pooled

    def _forward(
        self,
        mel: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_len: int,
    ) -> torch.Tensor:
        encoder = self.encoder
        batch = mel.shape[0]
        hidden = nn.functional.gelu(encoder.conv1(mel))
        hidden = nn.functional.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
        prefix = _extra_context_trim(STEADY_PREFIX_EXTRA_FRAMES)
        hidden = hidden[:, prefix : prefix + self.unit_length, :]
        hidden = hidden + encoder.embed_positions(position_ids)
        head_dim = self.embed_dim // self.num_heads
        for layer_index, layer in enumerate(encoder.layers):
            attention = layer.self_attn
            residual = hidden
            normed = layer.self_attn_layer_norm(hidden)
            query = attention.q_proj(normed) * attention.scaling
            key = attention.k_proj(normed)
            value = attention.v_proj(normed)
            cache[layer_index, 0, :, cache_len : cache_len + self.unit_length, :].copy_(key)
            cache[layer_index, 1, :, cache_len : cache_len + self.unit_length, :].copy_(value)
            q = query.view(batch, self.unit_length, self.num_heads, head_dim).transpose(1, 2)
            k_all = cache[layer_index, 0].view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
            v_all = cache[layer_index, 1].view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
            attended = _attend(q, k_all, v_all, mask, self.implementation)
            attended = attended.transpose(1, 2).reshape(batch, self.unit_length, self.embed_dim)
            hidden = residual + attention.out_proj(attended)
            residual = hidden
            hidden = layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden)))
            hidden = residual + layer.fc2(hidden)
        hidden = encoder.layer_norm(hidden)
        embeds = self.projection(hidden)
        return self.pooler(embeds.transpose(1, 2)).transpose(1, 2)

    def encode(
        self,
        chunks: Sequence[StreamingAudioChunk],
    ) -> tuple[list[torch.Tensor | None], list[StreamingAudioKVCache | None]]:
        outputs: list[torch.Tensor | None] = [None] * len(chunks)
        caches: list[StreamingAudioKVCache | None] = [chunk.cache for chunk in chunks]
        steady: list[_Group] = []
        eager_indices: list[int] = []
        for index, chunk in enumerate(chunks):
            if not row_is_steady(chunk, expected_frames=self.unit_frames):
                eager_indices.append(index)
                continue
            cache_in = chunk.cache
            assert cache_in is not None
            reset = cache_in.length + self.unit_length >= self.max_positions
            past = 0 if reset else cache_in.length
            if past > self.max_cache_bucket:
                eager_indices.append(index)
                continue
            cache_out = (
                StreamingAudioKVCache(
                    num_layers=self.num_layers,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    max_positions=self.max_positions,
                    page_positions=self.page_positions,
                )
                if reset
                else cache_in
            )
            steady.append(_Group(index=index, cache_in=cache_in, cache_out=cache_out, past=past))

        for start in range(0, len(steady), self.max_batch):
            self._replay(steady[start : start + self.max_batch], chunks, outputs, caches)

        if eager_indices:
            sub_outputs, sub_caches = encode_streaming_audio_batch(
                self.encoder,
                self.projection,
                self.pooler,
                [chunks[i] for i in eager_indices],
                pool_step=self.pool_step,
                page_positions=self.page_positions,
            )
            for local, index in enumerate(eager_indices):
                outputs[index] = sub_outputs[local]
                caches[index] = sub_caches[local]
        return outputs, caches

    def _stage_mel(self, dst: torch.Tensor, feature: torch.Tensor) -> None:
        if self.pinned_h2d and feature.device.type == "cpu":
            dst.copy_(feature.to(dtype=self.dtype).pin_memory(), non_blocking=True)
        else:
            dst.copy_(feature.to(device=self.device, dtype=self.dtype))

    @torch.inference_mode()
    def _replay(
        self,
        group: list[_Group],
        chunks: Sequence[StreamingAudioChunk],
        outputs: list[torch.Tensor | None],
        caches: list[StreamingAudioKVCache | None],
    ) -> None:
        batch = select_bucket(len(group), self.batch_sizes)
        cache_len = select_bucket(max((row.past for row in group), default=0), self.cache_buckets)
        assert batch is not None and cache_len is not None
        key = (batch, cache_len)
        mel, cache, mask, position_ids = self._views(batch, cache_len)
        mask.fill_(0.0)
        if len(group) < batch:
            mask[len(group) :, :, :, :cache_len].fill_(_NEG_INF)
            position_ids[len(group) :].copy_(torch.arange(self.unit_length, device=self.device))
        for slot, row in enumerate(group):
            feature = chunks[row.index].features
            if feature.ndim == 3:
                feature = feature[0]
            self._stage_mel(mel[slot], feature)
            pad = cache_len - row.past
            if pad:
                mask[slot, :, :, :pad].fill_(_NEG_INF)
            if row.past:
                cache[:, :, slot, pad:cache_len, :].copy_(row.cache_in.history(row.past))
            position_ids[slot].copy_(torch.arange(row.past, row.past + self.unit_length, device=self.device))
        self._graphs[key].replay()
        pooled = self._pooled[key]
        for slot, row in enumerate(group):
            outputs[row.index] = pooled[slot, : self.pooled_length].clone()
            row.cache_out.reserve(row.past + self.unit_length, dtype=self.dtype, device=self.device)
            row.cache_out.commit(row.past, cache[:, :, slot, cache_len : cache_len + self.unit_length, :])
            row.cache_out.length = row.past + self.unit_length
            caches[row.index] = row.cache_out


__all__ = [
    "DEFAULT_GRAPH_BATCH_SIZES",
    "DEFAULT_GRAPH_CACHE_BUCKETS",
    "STEADY_PREFIX_EXTRA_FRAMES",
    "STEADY_SUFFIX_EXTRA_FRAMES",
    "StreamingAudioGraphEncoder",
    "normalize_buckets",
    "row_is_steady",
    "select_bucket",
    "steady_pooled_length",
    "steady_unit_length",
]
