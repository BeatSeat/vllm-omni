# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS custom sampling: top-k, nucleus, and RAS (Repetition-Aware Sampling).

Standalone sampling methods used by GLMTTSForConditionalGeneration.sample().
Extracted to keep the main model file under the 800-line cap.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from vllm.logger import init_logger

__all__ = [
    "log_sampling_debug",
    "nucleus_sample_one",
    "req_float",
    "req_scalar",
    "sample_ras_one",
    "sample_topk_one",
    "sampling_debug_bounds",
    "sampling_debug_enabled",
]

logger = init_logger(__name__)

# Debug env vars
_DEBUG_SAMPLING_ENV = "GLM_TTS_DEBUG_SAMPLING"
_DEBUG_SAMPLING_START_ENV = "GLM_TTS_DEBUG_SAMPLING_START"
_DEBUG_SAMPLING_END_ENV = "GLM_TTS_DEBUG_SAMPLING_END"
_DEBUG_SAMPLING_MAX_STEPS = 12


def sampling_debug_enabled() -> bool:
    """Check if debug logging for GLM-TTS sampling is enabled via env."""
    value = os.environ.get(_DEBUG_SAMPLING_ENV, "")
    return value.lower() in {"1", "true", "yes", "on", "debug"}


def sampling_debug_bounds() -> tuple[int | None, int | None]:
    """Parse optional step range bounds from env."""

    def _parse(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, raw)
            return None

    return _parse(_DEBUG_SAMPLING_START_ENV), _parse(_DEBUG_SAMPLING_END_ENV)


def req_scalar(param: torch.Tensor | None, req_idx: int, default: int) -> int:
    """Read a per-request int scalar from a flat tensor."""
    if param is None or param.numel() == 0:
        return default
    index = min(req_idx, int(param.numel()) - 1)
    return int(param.reshape(-1)[index].item())


def req_float(param: torch.Tensor | None, req_idx: int, default: float) -> float:
    """Read a per-request float scalar from a flat tensor."""
    if param is None or param.numel() == 0:
        return default
    index = min(req_idx, int(param.numel()) - 1)
    return float(param.reshape(-1)[index].item())


def multinomial_sample(
    weights: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.multinomial(weights, 1, replacement=True, generator=generator).reshape(())


def sample_topk_one(
    weighted_scores: torch.Tensor,
    *,
    top_k: int,
    eoa_token_id: int,
    ignore_eos: bool,
    generator: torch.Generator | None,
) -> int:
    """Sample one token using top-k filtering."""
    top_k = max(1, min(int(top_k), int(weighted_scores.shape[-1])))
    while True:
        prob, indices = weighted_scores.softmax(dim=-1).topk(top_k)
        sampled_index = multinomial_sample(prob, generator=generator)
        top_id = int(indices[int(sampled_index.item())].item())
        if (not ignore_eos) or top_id != eoa_token_id:
            return top_id


def nucleus_sample_one(
    weighted_scores: torch.Tensor,
    *,
    top_p: float,
    top_k: int,
    temperature: float,
    generator: torch.Generator | None,
) -> int:
    """Sample one token using nucleus (top-p) filtering."""
    scaled_scores = weighted_scores / max(float(temperature), 1e-5)
    probs = scaled_scores.softmax(dim=0)
    sorted_prob, sorted_idx = probs.sort(descending=True, stable=True)
    kept_probs: list[torch.Tensor] = []
    kept_indices: list[torch.Tensor] = []
    cum_prob = 0.0
    max_keep = len(sorted_idx) if top_k <= 0 else min(int(top_k), len(sorted_idx))
    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(kept_probs) < max_keep:
            cum_prob += float(sorted_prob[i].item())
            kept_probs.append(sorted_prob[i])
            kept_indices.append(sorted_idx[i])
        else:
            break

    if not kept_probs:
        return int(sorted_idx[0].item())

    sample_probs = torch.stack(kept_probs)
    sample_idx = multinomial_sample(sample_probs, generator=generator)
    sample_indices = torch.stack(kept_indices)
    return int(sample_indices[int(sample_idx.item())].item())


def sample_ras_one(
    weighted_scores: torch.Tensor,
    decoded_tokens: Sequence[int],
    *,
    top_p: float,
    top_k: int,
    win_size: int,
    tau_r: float,
    temperature: float,
    generator: torch.Generator | None,
) -> int:
    """Repetition-Aware Sampling: nucleus + masked fallback on repetition."""
    top_id = nucleus_sample_one(
        weighted_scores,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        generator=generator,
    )
    if win_size > 0 and decoded_tokens:
        recent = torch.as_tensor(
            list(decoded_tokens[-win_size:]),
            device=weighted_scores.device,
            dtype=torch.long,
        )
        rep_num = int((recent == top_id).sum().item())
        if rep_num >= win_size * tau_r:
            weighted_scores = weighted_scores.clone()
            weighted_scores[top_id] = float("-inf")
            fallback_probs = weighted_scores.softmax(dim=0)
            top_id = int(multinomial_sample(fallback_probs, generator=generator).item())
    return top_id


def log_sampling_debug(
    *,
    req_idx: int,
    weighted_scores: torch.Tensor,
    decoded_tokens: Sequence[int],
    sampled_id: int,
    sample_method: str,
    eoa_token_id: int,
) -> None:
    """Conditional debug logging for GLM-TTS sampling steps."""
    if not sampling_debug_enabled():
        return

    step = len(decoded_tokens)
    debug_start, debug_end = sampling_debug_bounds()
    if debug_start is None and debug_end is None:
        if step >= _DEBUG_SAMPLING_MAX_STEPS and sampled_id != eoa_token_id:
            return
    elif sampled_id != eoa_token_id:
        if debug_start is not None and step < debug_start:
            return
        if debug_end is not None and step > debug_end:
            return

    probs = weighted_scores.softmax(dim=0)
    top_n = min(10, int(probs.shape[0]))
    top_probs, top_ids = probs.topk(top_n)
    eoa_prob = float(probs[eoa_token_id].item())
    eoa_score = float(weighted_scores[eoa_token_id].item())
    sampled_prob = float(probs[sampled_id].item())
    sampled_score = float(weighted_scores[sampled_id].item())
    top_items = ", ".join(
        f"{int(tok)}:{float(prob):.4f}" for tok, prob in zip(top_ids.tolist(), top_probs.tolist(), strict=False)
    )
    tail = list(decoded_tokens[-8:]) if decoded_tokens else []
    logger.info(
        "GLM-TTS sampling debug: req=%d step=%d method=%s sampled_id=%d sampled_prob=%.6f "
        "sampled_score=%.4f eoa_id=%d eoa_prob=%.6f eoa_score=%.4f decoded_tail=%s top10=[%s]",
        req_idx,
        step,
        sample_method,
        sampled_id,
        sampled_prob,
        sampled_score,
        eoa_token_id,
        eoa_prob,
        eoa_score,
        tail,
        top_items,
    )
