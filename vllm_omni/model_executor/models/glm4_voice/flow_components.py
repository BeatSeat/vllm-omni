# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained CosyVoice v1 flow-matching components for GLM-4-Voice.

Reimplements the full CosyVoice v1 flow stack (Conformer encoder,
InterpolateRegulator, matcha UNet, CFM ODE solver) so that
GLM-4-Voice decoder can load ``flow.pt`` / ``hift.pt`` without
HyperPyYAML, matcha, or external CosyVoice packages.

Attribute names must exactly match the reference CosyVoice v1 code
so that ``load_state_dict`` works against the original checkpoints.
"""

from __future__ import annotations

import math
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

# ---------------------------------------------------------------------------
# Section A: Utility functions
# ---------------------------------------------------------------------------


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Bool mask where True = padding position."""
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool mask (True=attend) to additive bias (0=attend, -inf=ignore)."""
    return torch.where(mask, torch.tensor(0.0, dtype=dtype), torch.tensor(float("-inf"), dtype=dtype))


# ---------------------------------------------------------------------------
# Section B: Positional encoding + subsampling
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000, reverse: bool = False):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(
        self, offset: Union[int, torch.Tensor], size: int, apply_dropout: bool = True
    ) -> torch.Tensor:
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        else:
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size).to(offset.device)
            flag = index > 0
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])
        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding — does NOT add pos_emb into x."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class LinearNoSubsampling(nn.Module):
    """Linear projection + positional encoding (no subsampling)."""

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc: nn.Module):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


# ---------------------------------------------------------------------------
# Section C: Attention, FFN, ConvModule
# ---------------------------------------------------------------------------


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention (wenet / ESPnet style)."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        n_batch = value.size(0)
        if mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:, :, :, : scores.size(-1)]
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.forward_qkv(query, key, value)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention with relative positional encoding."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class PositionwiseFeedForward(nn.Module):
    """Position-wise FFN."""

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class ConvolutionModule(nn.Module):
    """Conformer convolution module."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, 1, 0, bias=bias)

        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding, groups=channels, bias=bias)

        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, 1, 0, bias=bias)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:
                x = F.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                x = torch.cat((cache, x), dim=2)
            new_cache = x[:, :, -self.lorder :]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)

        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)
        return x.transpose(1, 2), new_cache


# ---------------------------------------------------------------------------
# Section D: Conformer encoder
# ---------------------------------------------------------------------------


class ConformerEncoderLayer(nn.Module):
    """Single Conformer encoder layer (macaron FFN + MHSA + Conv + FFN)."""

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module | None = None,
        feed_forward_macaron: nn.Module | None = None,
        conv_module: nn.Module | None = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)
            self.norm_final = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # macaron FFN
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # MHSA
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Conv module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        # FFN
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class ConformerEncoder(nn.Module):
    """Conformer encoder (CosyVoice v1 style).

    For GLM-4-Voice, the config.yaml uses ``BlockConformerEncoder`` which
    is equivalent to ``ConformerEncoder`` with ``input_layer="linear"``
    and ``input_size == output_size``. The ``block_size`` parameter only
    affects training/streaming and is ignored for offline inference.
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
    ):
        super().__init__()
        self._output_size = output_size

        pos_enc = RelPositionalEncoding(output_size, positional_dropout_rate)
        self.embed = LinearNoSubsampling(input_size, output_size, dropout_rate, pos_enc)
        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

        activation = nn.SiLU()

        encoder_selfattn_layer_args = (attention_heads, output_size, attention_dropout_rate, key_bias)
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, causal)

        self.encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None,
                    ConvolutionModule(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks

        # Full attention mask (non-streaming)
        chunk_masks = masks.expand(-1, T, -1)  # (B, T, T)

        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks


# ---------------------------------------------------------------------------
# Section E: Matcha-TTS components (for the UNet estimator)
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(x.dtype)


class Block1D(nn.Module):
    """Conv1d → GroupNorm → Mish."""

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(8, dim_out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(nn.Module):
    """Residual block with time-conditioning."""

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out)
        self.block2 = Block1D(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class TimestepEmbedding(nn.Module):
    """MLP for timestep conditioning: Linear → act → Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        else:
            self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor, condition: Any = None) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Downsample1D(nn.Module):
    """Strided Conv1d downsampler (factor 2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """ConvTranspose1d upsampler (factor 2)."""

    def __init__(self, dim: int, use_conv_transpose: bool = False):
        super().__init__()
        self.use_conv_transpose = use_conv_transpose
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
        else:
            self.conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv_transpose and self.conv is not None:
            return self.conv(x)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return x


class GELU(nn.Module):
    """GELU activation wrapper (matches diffusers GELU with .proj attribute)."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states), approximate=self.approximate)


class FeedForwardMatcha(nn.Module):
    """Matcha/diffusers-style FeedForward for BasicTransformerBlock."""

    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, dropout: float = 0.0, activation_fn: str = "gelu"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == "geglu":
            act_fn = GELU(dim, inner_dim)  # simplified — GEGLU not needed for GLM-4-Voice
        else:
            act_fn = GELU(dim, inner_dim)

        self.net = nn.ModuleList([act_fn, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AttentionMatcha(nn.Module):
    """Simplified multi-head attention matching matcha/diffusers state_dict keys."""

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        h = self.heads

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.view(batch_size, seq_len, h, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, h, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, h, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim)

        for module in self.to_out:
            out = module(out)
        return out


class BasicTransformerBlock(nn.Module):
    """Matcha/diffusers BasicTransformerBlock (self-attention only, no cross-attention)."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = AttentionMatcha(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForwardMatcha(dim=dim, dropout=dropout, activation_fn=activation_fn)
        self._chunk_size = None
        self._chunk_dim = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Self-attention
        norm_hidden = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden, attention_mask=attention_mask)
        hidden_states = attn_output + hidden_states

        # FFN
        norm_hidden = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden)
        hidden_states = ff_output + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Section F: UNet decoder + Conditional Flow Matching
# ---------------------------------------------------------------------------


class ConditionalDecoder(nn.Module):
    """UNet-style 1D conditional decoder for flow matching (matcha architecture)."""

    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 80,
        channels: tuple[int, ...] = (256, 256),
        dropout: float = 0.05,
        attention_head_dim: int = 64,
        n_blocks: int = 1,
        num_mid_blocks: int = 2,
        num_heads: int = 4,
        act_fn: str = "gelu",
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels=in_channels, time_embed_dim=time_embed_dim, act_fn="silu")

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # Down
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # Mid
        for _ in range(num_mid_blocks):
            resnet = ResnetBlock1D(dim=channels[-1], dim_out=channels[-1], time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=channels[-1],
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # Up
        channels_rev = channels[::-1] + (channels[0],)
        for i in range(len(channels_rev) - 1):
            input_channel = channels_rev[i] * 2
            output_channel = channels_rev[i + 1]
            is_last = i == len(channels_rev) - 2
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = Upsample1D(output_channel, use_conv_transpose=True) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels_rev[-1], channels_rev[-1])
        self.final_proj = nn.Conv1d(channels_rev[-1], self.out_channels, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> torch.Tensor:
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_down.bool().expand(-1, x.size(1), -1).contiguous()
            attn_mask = mask_to_bias(attn_mask, x.dtype).unsqueeze(1)
            for block in transformer_blocks:
                x = block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_mid.bool().expand(-1, x.size(1), -1).contiguous()
            attn_mask = mask_to_bias(attn_mask, x.dtype).unsqueeze(1)
            for block in transformer_blocks:
                x = block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, : skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = mask_up.bool().expand(-1, x.size(1), -1).contiguous()
            attn_mask = mask_to_bias(attn_mask, x.dtype).unsqueeze(1)
            for block in transformer_blocks:
                x = block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


class ConditionalCFM(nn.Module):
    """Conditional Flow Matching ODE solver (CosyVoice v1 style).

    No trainable parameters except the estimator. Implements classifier-free
    guidance with batched CFG (batch of 2: conditional + unconditional).
    """

    def __init__(
        self,
        in_channels: int = 240,
        cfm_params: dict | None = None,
        n_spks: int = 1,
        spk_emb_dim: int = 80,
        estimator: nn.Module | None = None,
    ):
        super().__init__()
        self.n_feats = in_channels
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.estimator = estimator

        params = cfm_params or {}
        self.sigma_min = params.get("sigma_min", 1e-6)
        self.solver = params.get("solver", "euler")
        self.t_scheduler = params.get("t_scheduler", "cosine")
        self.inference_cfg_rate = params.get("inference_cfg_rate", 0.7)

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> torch.Tensor:
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        sol = []

        # Batched CFG: batch of 2 [conditional, unconditional]
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=mu.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=mu.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=mu.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=mu.dtype)
        spks_in = torch.zeros([2, self.spk_emb_dim], device=x.device, dtype=mu.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=mu.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            if spks is not None:
                spks_in[0] = spks
            if cond is not None:
                cond_in[0] = cond

            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()


# ---------------------------------------------------------------------------
# Section G: Flow top-level + F0 predictor + InterpolateRegulator
# ---------------------------------------------------------------------------


class InterpolateRegulator(nn.Module):
    """Length regulator via linear interpolation + Conv1d stack."""

    def __init__(self, channels: int = 80, sampling_ratios: tuple = (1, 1, 1, 1), out_channels: int | None = None, groups: int = 1):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor, ylens: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode="linear")
        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask, ylens

    def inference(self, x1: torch.Tensor, x2: torch.Tensor, mel_len1: int, mel_len2: int, input_frame_rate: float = 50.0) -> Tuple[torch.Tensor, int]:
        if x2.shape[1] > 40:
            x2_head = F.interpolate(x2[:, :20].transpose(1, 2).contiguous(), size=int(20 / input_frame_rate * 22050 / 256), mode="linear")
            x2_mid = F.interpolate(
                x2[:, 20:-20].transpose(1, 2).contiguous(),
                size=mel_len2 - int(20 / input_frame_rate * 22050 / 256) * 2,
                mode="linear",
            )
            x2_tail = F.interpolate(x2[:, -20:].transpose(1, 2).contiguous(), size=int(20 / input_frame_rate * 22050 / 256), mode="linear")
            x2 = torch.concat([x2_head, x2_mid, x2_tail], dim=2)
        else:
            x2 = F.interpolate(x2.transpose(1, 2).contiguous(), size=mel_len2, mode="linear")
        if x1.shape[1] != 0:
            x1 = F.interpolate(x1.transpose(1, 2).contiguous(), size=mel_len1, mode="linear")
            x = torch.concat([x1, x2], dim=2)
        else:
            x = x2
        out = self.model(x).transpose(1, 2).contiguous()
        return out, mel_len1 + mel_len2


class ConvRNNF0Predictor(nn.Module):
    """Non-causal F0 predictor for HiFTGenerator (CosyVoice v1 style)."""

    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.num_class = num_class
        try:
            from torch.nn.utils.parametrizations import weight_norm
        except ImportError:
            from torch.nn.utils import weight_norm

        self.condnet = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class MaskedDiffWithXvec(nn.Module):
    """CosyVoice v1 flow model: embedding → encoder → regulator → CFM decoder."""

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 16384,
        input_frame_rate: float = 12.5,
        only_mask_loss: bool = True,
        encoder: nn.Module | None = None,
        length_regulator: nn.Module | None = None,
        decoder: nn.Module | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(encoder.output_size() if encoder else input_size, output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    @torch.inference_mode()
    def inference(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        finalize: bool = True,
    ) -> torch.Tensor:
        assert token.shape[0] == 1

        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len

        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2], device=token.device))).to(h)
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
        )

        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float()
