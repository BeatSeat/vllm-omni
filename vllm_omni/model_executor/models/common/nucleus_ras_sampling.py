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

    Vectorized via ``torch.cumsum`` — no Python loop over the vocabulary.
    """
    probs = weighted_scores.softmax(dim=0)
    sorted_prob, sorted_idx = probs.sort(descending=True, stable=True)

    # Apply top-k truncation
    if top_k > 0:
        sorted_prob = sorted_prob[: int(top_k)]
        sorted_idx = sorted_idx[: int(top_k)]

    # Apply top-p (nucleus) filtering: keep the smallest prefix whose
    # cumulative probability exceeds top_p, always including the first token.
    # Use float32 for cumulative sum to match the original Python float64
    # accumulation loop.  Low-precision (fp16/bf16) cumsum over ~32K vocab
    # causes rounding drift that shifts the top-p boundary, changing which
    # tokens are kept and ultimately producing different AR sequences.
    sorted_prob_f32 = sorted_prob.float()
    cum_prob = sorted_prob_f32.cumsum(dim=0)
    # Mask: include tokens where cumsum *before* this token is still < top_p
    mask = (cum_prob - sorted_prob_f32) < top_p
    if not mask.any():
        # Fallback: always keep at least the top token
        mask[0] = True

    kept_probs = sorted_prob[mask]
    kept_indices = sorted_idx[mask]

    sample_pos = multinomial_sample(kept_probs, generator=generator)
    return int(kept_indices[int(sample_pos.item())].item())


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
