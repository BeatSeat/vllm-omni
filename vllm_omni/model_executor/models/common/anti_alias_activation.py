# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/junjun3518/alias-free-torch (Apache-2.0)
# NPU/XPU compatibility from Qwen2.5-Omni.
"""Anti-aliased activation wrapper for BigVGAN-style vocoders.

Provides UpSample1d / DownSample1d with Kaiser-windowed sinc filters and
TorchActivation1d that composes upsample -> activation -> downsample.
Supports CUDA, NPU, and XPU devices.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.platforms import current_platform as current_omni_platform


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """1D Kaiser-windowed sinc filter. Returns shape (1, 1, kernel_size)."""
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    # NPU/XPU: kaiser_window must be created on CPU then moved
    if current_omni_platform.is_npu():
        kaiser_window = torch.kaiser_window(
            kernel_size, beta=beta, periodic=False, dtype=torch.float32, device="cpu"
        ).to("npu")
    elif current_omni_platform.is_xpu():
        kaiser_window = torch.kaiser_window(
            kernel_size, beta=beta, periodic=False, dtype=torch.float32, device="cpu"
        ).to("xpu")
    else:
        kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter
    normalized_filter /= normalized_filter.sum()
    return normalized_filter.view(1, 1, kernel_size)


def _replication_pad_1d(hidden_states: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    """Manual replicate padding — workaround for NPU where F.pad(mode='replicate') is limited."""
    if pad_left == 0 and pad_right == 0:
        return hidden_states
    segments = []
    if pad_left > 0:
        segments.append(hidden_states[..., :1].expand(*hidden_states.shape[:-1], pad_left))
    segments.append(hidden_states)
    if pad_right > 0:
        segments.append(hidden_states[..., -1:].expand(*hidden_states.shape[:-1], pad_right))
    return torch.cat(segments, dim=-1)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filt = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filt, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        if current_omni_platform.is_npu():
            input_dtype = hidden_states.dtype
            hidden_states = _replication_pad_1d(hidden_states.to(self.filter.dtype), self.pad, self.pad)
            hidden_states = self.ratio * F.conv_transpose1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(input_dtype)
        else:
            orig_dtype = hidden_states.dtype
            hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate").to(self.filter.dtype)
            hidden_states = self.ratio * F.conv_transpose1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(orig_dtype)
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]
        return hidden_states


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filt, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        if current_omni_platform.is_npu():
            input_dtype = hidden_states.dtype
            hidden_states = _replication_pad_1d(hidden_states.to(self.filter.dtype), self.pad_left, self.pad_right)
            out = F.conv1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(input_dtype)
        else:
            orig_dtype = hidden_states.dtype
            hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate").to(
                self.filter.dtype
            )
            out = F.conv1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(orig_dtype)
        return out


class TorchActivation1d(nn.Module):
    """Anti-aliased activation: upsample -> activation -> downsample."""

    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise TypeError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)
        return hidden_states
