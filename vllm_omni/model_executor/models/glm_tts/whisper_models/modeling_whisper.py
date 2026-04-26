# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WhisperVQEncoder — inference-only, encoder + VQ codebook.

Simplified from the upstream CogAudio/GLM-TTS modeling_whisper.py.
Uses Flash Attention (varlen) for efficient attention computation,
following the same pattern as qwen3_tts and ming_flash_omni encoders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from vllm_omni.diffusion.attention.backends.utils.fa import HAS_FLASH_ATTN, flash_attn_varlen_func
from vllm_omni.model_executor.models.whisper_utils import Conv1d, Linear, sinusoids

from .configuration_whisper import WhisperVQConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class QuantizedBaseModelOutput(BaseModelOutput):
    quantized_token_ids: torch.LongTensor | None = None


def vector_quantize(inputs: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Nearest-neighbour codebook lookup."""
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook**2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)
    distances = torch.addmm(
        codebook_sqr + inputs_sqr,
        inputs_flatten,
        codebook.t(),
        alpha=-2.0,
        beta=1.0,
    )
    _, indices_flatten = torch.min(distances, dim=1)
    codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
    codes = codes_flatten.view_as(inputs)
    return codes, indices_flatten, distances


# ---------------------------------------------------------------------------
# Attention (Flash Attn varlen with manual fallback)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head attention using Flash Attention varlen for packed sequences."""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head

        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.use_flash_attn = HAS_FLASH_ATTN

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if self.use_flash_attn and q.dtype in (torch.float16, torch.bfloat16):
            out = self._flash_attn(q, k, v, cu_seqlens)
        else:
            out = self._manual_attn(q, k, v, cu_seqlens)

        return self.out(out)

    def _flash_attn(self, q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor) -> Tensor:
        n_ctx = q.shape[0]
        q = q.view(n_ctx, self.n_head, self.head_dim)
        k = k.view(n_ctx, self.n_head, self.head_dim)
        v = v.view(n_ctx, self.n_head, self.head_dim)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0)
        return out.reshape(n_ctx, self.n_state)

    def _manual_attn(self, q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor) -> Tensor:
        """SDPA fallback for non-fp16/bf16 dtypes or when Flash Attn is unavailable."""
        n_ctx = q.shape[0]
        scale = self.head_dim**-0.5

        q = q.view(n_ctx, self.n_head, self.head_dim)
        k = k.view(n_ctx, self.n_head, self.head_dim)
        v = v.view(n_ctx, self.n_head, self.head_dim)

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        batch_size = len(seqlens)
        max_seqlen = max(seqlens)

        # Pad to batch format for matmul
        q_pad = torch.zeros(batch_size, max_seqlen, self.n_head, self.head_dim, dtype=q.dtype, device=q.device)
        k_pad = torch.zeros_like(q_pad)
        v_pad = torch.zeros_like(q_pad)

        for i in range(batch_size):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            sl = seqlens[i]
            q_pad[i, :sl] = q[s:e]
            k_pad[i, :sl] = k[s:e]
            v_pad[i, :sl] = v[s:e]

        # [B, n_head, T, head_dim]
        q_pad = q_pad.transpose(1, 2)
        k_pad = k_pad.transpose(1, 2)
        v_pad = v_pad.transpose(1, 2)

        # Padding mask
        attn_mask = torch.arange(max_seqlen, device=q.device)[None, :] < torch.tensor(seqlens, device=q.device)[:, None]
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        attn_mask = attn_mask.masked_fill(~attn_mask, -torch.finfo(q.dtype).max)

        attn_scores = torch.matmul(q_pad, k_pad.transpose(-2, -1)) * scale
        attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v_pad)

        # [B, T, n_state]
        context = context.transpose(1, 2).contiguous().view(batch_size, max_seqlen, self.n_state)

        # Unpad back to packed format
        out = torch.cat([context[i, : seqlens[i]] for i in range(batch_size)], dim=0)
        return out


# ---------------------------------------------------------------------------
# Encoder Layer (pre-norm transformer block)
# ---------------------------------------------------------------------------


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        n_mlp = n_state * 4
        self.attn_ln = nn.LayerNorm(n_state)
        self.mlp_ln = nn.LayerNorm(n_state)
        self.attn = MultiHeadAttention(n_state, n_head)
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        x = x + self.attn(self.attn_ln(x), cu_seqlens=cu_seqlens)
        x = x + self.mlp(self.mlp_ln(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class WhisperVQEncoder(nn.Module):
    """Whisper encoder with VQ codebook and pooling.

    Inference-only: uses Flash Attention (varlen) for packed-sequence
    attention. Training helpers are not included.
    """

    def __init__(self, config: WhisperVQConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        n_head = config.encoder_attention_heads
        max_source_positions = config.max_source_positions

        # Conv stem
        self.conv1 = Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        # Positional embedding (frozen sinusoidal)
        self.register_buffer("embed_positions", sinusoids(max_source_positions, embed_dim))

        # Transformer layers
        n_layers = config.quantize_position if config.quantize_encoder_only else config.encoder_layers
        self.layers = nn.ModuleList([ResidualAttentionBlock(embed_dim, n_head) for _ in range(n_layers)])

        # Final layer norm (only when using full encoder)
        self.layer_norm = nn.LayerNorm(embed_dim) if not config.quantize_encoder_only else None

        # Pooling
        self.pooling_layer = None
        if config.pooling_kernel_size is not None:
            if config.pooling_type == "max":
                self.pooling_layer = nn.MaxPool1d(kernel_size=config.pooling_kernel_size)
            elif config.pooling_type == "avg":
                self.pooling_layer = nn.AvgPool1d(kernel_size=config.pooling_kernel_size)

        # VQ codebook
        self.codebook = None
        self.embed_positions2 = None
        if config.quantize_vocab_size is not None:
            self.codebook = nn.Embedding(config.quantize_vocab_size, embed_dim)
            pos2_len = max_source_positions
            if config.pooling_kernel_size is not None:
                pos2_len = math.ceil(max_source_positions / config.pooling_kernel_size)
            self.embed_positions2 = nn.Embedding(pos2_len, embed_dim)

    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def forward(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> QuantizedBaseModelOutput:
        """
        Args:
            input_features: [B, n_mels, T] mel spectrogram
            attention_mask: [B, T] binary mask (1 = valid, 0 = pad)
        """
        batch_size, _, raw_seq_len = input_features.shape
        conv_stride = self.conv1.stride[0] * self.conv2.stride[0]

        # Conv stem
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)  # [B, T', D]
        seq_len = hidden_states.shape[1]

        # Add positional embedding
        hidden_states = hidden_states + self.embed_positions[:seq_len].to(hidden_states.dtype)

        # Build cu_seqlens for packed sequences
        if attention_mask is not None:
            attention_mask = attention_mask[:, ::conv_stride]
            seq_lengths = attention_mask.sum(dim=1).int()  # [B]
        else:
            seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=hidden_states.device)

        cu_seqlens = F.pad(seq_lengths.cumsum(0), (1, 0)).int()

        # Pack sequences (remove padding)
        if attention_mask is not None and not attention_mask.all():
            # Build packed tensor from valid positions
            packed = torch.cat([hidden_states[i, : seq_lengths[i]] for i in range(batch_size)], dim=0)
        else:
            packed = hidden_states.reshape(-1, hidden_states.shape[-1])

        quantized_token_ids = None

        for idx, layer in enumerate(self.layers):
            packed = layer(packed, cu_seqlens=cu_seqlens)

            # Pooling after pooling_position
            if idx + 1 == self.config.pooling_position and self.config.pooling_kernel_size is not None:
                packed, cu_seqlens, seq_lengths = self._apply_pooling(packed, cu_seqlens, seq_lengths)

            # VQ after quantize_position
            if idx + 1 == self.config.quantize_position and self.config.quantize_vocab_size is not None:
                packed, quantized_token_ids = self._apply_vq(packed, cu_seqlens, seq_lengths)

        if self.layer_norm is not None:
            packed = self.layer_norm(packed)

        # Unpack back to padded batch format
        hidden_states = self._unpack(packed, cu_seqlens, seq_lengths, batch_size)

        return QuantizedBaseModelOutput(
            last_hidden_state=hidden_states,
            quantized_token_ids=quantized_token_ids,
        )

    def _apply_pooling(self, packed: Tensor, cu_seqlens: Tensor, seq_lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply pooling to packed sequences."""
        k = self.config.pooling_kernel_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        segments = packed.split(seqlens, dim=0)

        pooled_segments = []
        new_lengths = []
        for seg in segments:
            # [T, D] -> [1, D, T]
            seg_t = seg.unsqueeze(0).permute(0, 2, 1)
            if seg_t.shape[-1] % k != 0:
                pad_size = k - seg_t.shape[-1] % k
                seg_t = F.pad(seg_t, (0, pad_size))
            pooled = self.pooling_layer(seg_t).permute(0, 2, 1).squeeze(0)  # [T', D]
            pooled_segments.append(pooled)
            new_lengths.append(pooled.shape[0])

        new_packed = torch.cat(pooled_segments, dim=0)
        new_seq_lengths = torch.tensor(new_lengths, dtype=torch.int32, device=packed.device)
        new_cu_seqlens = F.pad(new_seq_lengths.cumsum(0), (1, 0)).int()
        return new_packed, new_cu_seqlens, new_seq_lengths

    def _apply_vq(self, packed: Tensor, cu_seqlens: Tensor, seq_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Apply vector quantization to packed sequences."""
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        segments = packed.split(seqlens, dim=0)

        quantized_segments = []
        token_id_segments = []
        for seg in segments:
            # seg: [T, D]
            seg_3d = seg.unsqueeze(0)  # [1, T, D]
            quantized, indices, _ = vector_quantize(seg_3d, self.codebook.weight)
            quantized_segments.append(quantized.squeeze(0))
            token_id_segments.append(indices.reshape(1, -1))

        packed = torch.cat(quantized_segments, dim=0)
        quantized_token_ids = torch.cat(token_id_segments, dim=0)

        # Add post-VQ positional embedding
        for i, seg_len in enumerate(seqlens):
            start = cu_seqlens[i]
            packed[start : start + seg_len] = packed[start : start + seg_len] + self.embed_positions2.weight[:seg_len]

        return packed, quantized_token_ids

    def _unpack(self, packed: Tensor, cu_seqlens: Tensor, seq_lengths: Tensor, batch_size: int) -> Tensor:
        """Unpack packed sequences back to padded batch format."""
        max_len = seq_lengths.max().item()
        embed_dim = packed.shape[-1]
        output = torch.zeros(batch_size, max_len, embed_dim, dtype=packed.dtype, device=packed.device)
        for i in range(batch_size):
            sl = seq_lengths[i].item()
            output[i, :sl] = packed[cu_seqlens[i] : cu_seqlens[i + 1]]
        return output
