# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker).
"""Shared Snake/SnakeBeta activations for speech decoders.

Used by: Qwen3-TTS, IndexTTS2 BigVGAN, and other vocoder models.
Supports Triton-fused forward on CUDA and eager fallback on all devices.
"""

import torch
from torch import nn
from torch.nn import Parameter
from vllm.logger import init_logger

logger = init_logger(__name__)


class Snake(nn.Module):
    """Sine-based periodic activation: Snake(x) := x + 1/a * sin^2(x*a).

    Shape: (B, C, T) -> (B, C, T)
    Reference: https://huggingface.co/papers/2006.08195
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        hidden_states = hidden_states + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )
        return hidden_states


class SnakeBeta(nn.Module):
    """Modified Snake with separate magnitude parameter: x + 1/b * sin^2(x*a).

    Shape: (B, C, T) -> (B, C, T)
    Reference: https://huggingface.co/papers/2006.08195
    """

    _triton_kernel = None  # None = untried, False = unavailable, callable = ready
    _TRITON_MAX_BLOCK_T = 4096

    @staticmethod
    def _init_triton():
        if SnakeBeta._triton_kernel is not None:
            return SnakeBeta._triton_kernel is not False
        try:
            import triton
            import triton.language as tl
        except ImportError:
            SnakeBeta._triton_kernel = False
            return False

        @triton.jit
        def _kernel(  # noqa: N803
            x_ptr,
            alpha_ptr,
            inv_beta_ptr,
            out_ptr,
            stride_b,
            stride_c,
            t_len,
            block_t: tl.constexpr,
        ):
            """Fused SnakeBeta using precomputed alpha and 1/(beta+eps)."""
            bid = tl.program_id(0)
            cid = tl.program_id(1)
            t_off = tl.program_id(2) * block_t + tl.arange(0, block_t)
            mask = t_off < t_len

            x = tl.load(x_ptr + bid * stride_b + cid * stride_c + t_off, mask=mask, other=0.0)
            ea = tl.load(alpha_ptr + cid)
            ib = tl.load(inv_beta_ptr + cid)
            x_float = x.to(tl.float32)
            sin_val = tl.sin(x_float * ea)
            result = x + (ib * sin_val * sin_val).to(x.dtype)

            tl.store(out_ptr + bid * stride_b + cid * stride_c + t_off, result, mask=mask)

        SnakeBeta._triton_kernel = _kernel
        return True

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9
        self.register_buffer("_alpha_cache", None, persistent=False)
        self.register_buffer("_inv_beta_cache", None, persistent=False)

    def precompute_exp_cache(self):
        with torch.no_grad():
            if self.alpha_logscale:
                self._alpha_cache = torch.exp(self.alpha).contiguous()
                self._inv_beta_cache = (1.0 / (torch.exp(self.beta) + self.no_div_by_zero)).contiguous()
            else:
                self._alpha_cache = self.alpha.contiguous()
                self._inv_beta_cache = (1.0 / (self.beta + self.no_div_by_zero)).contiguous()

    @property
    def _cached(self):
        return self._alpha_cache is not None

    def forward(self, hidden_states):
        if hidden_states.is_cuda and not torch.is_grad_enabled() and self._init_triton():
            try:
                return self._triton_forward(hidden_states)
            except Exception:
                logger.warning("Triton SnakeBeta failed, falling back to eager", exc_info=True)
                SnakeBeta._triton_kernel = False
        return self._eager_forward(hidden_states)

    def _eager_forward(self, hidden_states):
        if self._cached:
            alpha = self._alpha_cache.unsqueeze(0).unsqueeze(-1)
            inv_beta = self._inv_beta_cache.unsqueeze(0).unsqueeze(-1)
        else:
            alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
            beta = self.beta.unsqueeze(0).unsqueeze(-1)
            if self.alpha_logscale:
                alpha = torch.exp(alpha)
                beta = torch.exp(beta)
            inv_beta = 1.0 / (beta + self.no_div_by_zero)
        hidden_states = hidden_states + inv_beta * torch.pow(torch.sin(hidden_states * alpha), 2)
        return hidden_states

    def _triton_forward(self, x):
        import triton

        if not self._cached:
            self.precompute_exp_cache()

        x = x.contiguous()
        B, C, T = x.shape
        out = torch.empty_like(x)
        block_t = min(triton.next_power_of_2(T), self._TRITON_MAX_BLOCK_T)
        self._triton_kernel[(B, C, triton.cdiv(T, block_t))](
            x,
            self._alpha_cache,
            self._inv_beta_cache,
            out,
            x.stride(0),
            x.stride(1),
            t_len=T,
            block_t=block_t,
        )
        return out
