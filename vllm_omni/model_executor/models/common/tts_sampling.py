# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared TTS sampling primitives: nucleus (top-p/top-k) and RAS.

Used by CosyVoice3 and GLM-TTS (and any future TTS model with RAS).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def multinomial_sample(
    probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one sample from a categorical distribution."""
    return torch.multinomial(probs, 1, replacement=True, generator=generator).reshape(())


def nucleus_sample_one(
    weighted_scores: torch.Tensor,
    *,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token using nucleus (top-p + top-k) filtering.

    ``weighted_scores`` should be log-softmax-ed logits (for RAS callers)
    or raw logits (softmax is applied internally).
    """
    probs = weighted_scores.softmax(dim=0)
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
    return int(torch.stack(kept_indices)[int(sample_idx.item())].item())


def ras_sample_one(
    weighted_scores: torch.Tensor,
    decoded_tokens: Sequence[int],
    *,
    top_p: float,
    top_k: int,
    win_size: int,
    tau_r: float,
    generator: torch.Generator | None = None,
) -> int:
    """Repetition-Aware Sampling following GLM-TTS/CosyVoice.

    If the nucleus-sampled token appears too often in the recent window,
    upstream samples once from the original full distribution instead of
    masking the repeated token.  This matters for GLM-TTS because masking the
    dominant repeated token can artificially raise EOA probability right after
    the minimum length guard is lifted.
    """
    def _random_sample_one() -> int:
        return int(multinomial_sample(weighted_scores.softmax(dim=0), generator=generator).item())

    top_id = nucleus_sample_one(
        weighted_scores,
        top_p=top_p,
        top_k=top_k,
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
            top_id = _random_sample_one()
    return top_id
