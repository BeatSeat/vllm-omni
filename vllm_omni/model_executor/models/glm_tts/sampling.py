# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS per-request scalar accessors.

The core RAS and nucleus sampling logic lives in
``vllm_omni.model_executor.models.common.tts_sampling`` and is called from
``GLMTTSForConditionalGeneration.sample()``.
"""

from __future__ import annotations

import torch

__all__ = [
    "req_float",
]


def req_float(param: torch.Tensor | None, req_idx: int, default: float) -> float:
    if param is None or param.numel() == 0:
        return default
    index = min(req_idx, int(param.numel()) - 1)
    return float(param.reshape(-1)[index].item())
