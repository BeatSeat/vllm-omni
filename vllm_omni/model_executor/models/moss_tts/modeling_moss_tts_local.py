# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Local depth transformer for MossTTSRealtime.

A small (4-layer) Qwen3-style decoder that generates the rvq=16 RVQ codebook
codes for one audio frame, autoregressively over codebooks. It runs inside the
talker's per-step ``make_omni_output``, independent from vLLM's main scheduler.

The transformer body is shared with Qwen3-TTS and Qwen3-Omni via
``common.qwen3_code_predictor.CodePredictorBaseModel`` (re-prefill, no KV cache,
HF-compatible numerics). MossTTSRealtime differs in two ways:

  * codebook 0 is generated here from ``backbone_last_hidden`` (the other models
    receive it from the talker's main LM head), so we run one extra step and own
    all ``rvq`` LM heads;
  * sampling adds top-p and a windowed repetition penalty on top of
    temperature + top-k, matching upstream ``MossTTSRealtimeInference.generate``.

Re-prefilling the (<=rvq) frame each step is numerically identical to the
previous KV-cache loop -- causal attention, only the last position is read.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
    MossTTSLocalTransformerConfig,
)

logger = init_logger(__name__)
_MAX_LOCAL_CUDA_GRAPHS = 16


class MossTTSRealtimeLocalTransformer(nn.Module):
    """Per-frame depth transformer. Mirrors upstream ``...LocalTransformer``.

    State per audio frame:
      - The first input token uses ``backbone_last_hidden_state`` as the
        embedding (codebook 0's "input" is the backbone hidden, not a token).
      - Subsequent tokens (1..rvq-1) embed via ``model.codec_embedding[idx-1]``.

    Outputs:
      - One logit row per codebook position, projected through
        ``local_lm_heads[codebook_idx]`` (passed in by the talker).
    """

    def __init__(self, cfg: MossTTSLocalTransformerConfig) -> None:
        super().__init__()
        self.config = cfg
        # Shared body. Its codec_embedding holds rvq-1 embeddings -- upstream's
        # embed_tokens for codebooks 1..rvq-1 (codebook 0 uses the backbone
        # hidden). embedding_dim == hidden_size, so no projection is needed.
        self.model = CodePredictorBaseModel(
            cfg,
            embedding_dim=cfg.hidden_size,
            use_parallel_embedding=False,
            prefix="model",
        )
        self.register_buffer("_pos_ids", torch.arange(cfg.rvq, dtype=torch.long).unsqueeze(0), persistent=False)
        self._cuda_graphs: dict[
            tuple[Any, ...],
            tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._uncapturable_keys: set[tuple[Any, ...]] = set()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Delegate ``model.*`` tensors to ``CodePredictorBaseModel.load_weights``.

        The shared body fuses ``q/k/v_proj`` into ``qkv_proj`` and
        ``gate/up_proj`` into ``gate_up_proj``; its loader re-packs the HF
        shards and raises on incomplete or missing fused parameters.  Going
        through it (instead of ``default_weight_loader`` per tensor) keeps
        those guarantees for the MossTTSRealtime checkpoint too.
        """
        model_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], tensor))
        loaded = self.model.load_weights(iter(model_weights))
        return {f"model.{n}" for n in loaded}

    def _populate_rep_mask(
        self,
        rep_mask: torch.Tensor,
        *,
        history_tensors: Sequence[torch.Tensor | None] | None,
        history_per_codebook: list[list[int]] | None,
    ) -> bool:
        rvq = self.config.rvq
        vocab_size = self.config.audio_vocab_size
        device = rep_mask.device
        has_any = False
        if history_tensors is not None:
            for b, t in enumerate(history_tensors):
                if b >= rep_mask.shape[0] or not isinstance(t, torch.Tensor) or t.numel() == 0:
                    continue
                tail = t if (t.device == device and t.dtype == torch.long) else t.to(device=device, dtype=torch.long)
                if tail.dim() == 1 and tail.numel() % rvq == 0:
                    tail = tail.view(-1, rvq)
                if tail.dim() == 2 and tail.shape[1] == rvq:
                    rep_mask[b].scatter_(1, tail.t().clamp(0, vocab_size - 1), True)
                    has_any = True
        elif history_per_codebook is not None:
            for step in range(min(rvq, len(history_per_codebook))):
                hist = history_per_codebook[step]
                if hist:
                    hist_t = torch.tensor(hist, dtype=torch.long, device=device).clamp_(0, vocab_size - 1)
                    rep_mask[:, step, :].scatter_(1, hist_t.unsqueeze(0).expand(rep_mask.shape[0], -1), True)
                    has_any = True
        return has_any

    def _run_loop(
        self,
        embeds: torch.Tensor,
        pos_ids_full: torch.Tensor,
        rep_mask: torch.Tensor | None,
        codes: torch.Tensor,
        lm_heads: nn.ModuleList,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        repetition_penalty: float,
    ) -> None:
        rvq = self.config.rvq
        hidden_size = self.config.hidden_size
        B = embeds.shape[0]
        codec_embeds = self.model.codec_embedding
        apply_rep = rep_mask is not None and repetition_penalty != 1.0

        for step in range(rvq):
            seq_len = step + 1
            hidden = self.model(embeds[:, :seq_len, :], pos_ids_full[:, :seq_len])
            logits = lm_heads[step](hidden[:, step, :]).float()

            if apply_rep and rep_mask is not None:
                step_mask = rep_mask[:, step, :]
                penalized = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
                logits = torch.where(step_mask, penalized, logits)

            codes[:, step] = _sample_token(logits, temperature, top_k, top_p, do_sample)

            if step + 1 < rvq:
                embeds[:, step + 1, :] = codec_embeds[step](codes[:, step].view(B, 1)).view(B, hidden_size)

    @torch.no_grad()
    def generate_frame(
        self,
        backbone_last_hidden: torch.Tensor,  # (B, H)
        lm_heads: nn.ModuleList,  # ModuleList of rvq Linear(H -> audio_vocab_size)
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        history_per_codebook: list[list[int]] | None = None,
        history_tensors: Sequence[torch.Tensor | None] | None = None,
        use_cuda_graph: bool = False,
    ) -> torch.Tensor:
        """Generate one audio frame (rvq codebook tokens) for batch B.

        Returns a ``(B, rvq)`` LongTensor.

        ``history_tensors`` provides per-request ``(W, rvq)`` GPU tensors of
        recently emitted token ids, avoiding host-device synchronization across
        concurrent requests. ``history_per_codebook`` is retained for direct
        single-request callers.
        """
        device = backbone_last_hidden.device
        B = backbone_last_hidden.shape[0]
        rvq = self.config.rvq
        hidden_size = self.config.hidden_size
        vocab_size = self.config.audio_vocab_size

        can_graph = (
            use_cuda_graph
            and device.type == "cuda"
            and B > 0
            and not torch.cuda.is_current_stream_capturing()
        )
        if can_graph:
            key = (
                B,
                device.index,
                backbone_last_hidden.dtype,
                id(lm_heads),
                float(temperature),
                int(top_k),
                float(top_p),
                bool(do_sample),
                float(repetition_penalty),
            )
            if key not in self._cuda_graphs and key not in self._uncapturable_keys:
                if len(self._cuda_graphs) < _MAX_LOCAL_CUDA_GRAPHS:
                    try:
                        static_embeds = backbone_last_hidden.new_zeros((B, rvq, hidden_size))
                        static_pos_ids = (
                            torch.arange(rvq, device=device, dtype=torch.long)
                            .unsqueeze(0)
                            .expand(B, -1)
                            .contiguous()
                        )
                        static_rep_mask = torch.zeros((B, rvq, vocab_size), dtype=torch.bool, device=device)
                        static_codes = backbone_last_hidden.new_zeros((B, rvq), dtype=torch.long)

                        capture_stream = torch.cuda.Stream(device=device)
                        main_stream = torch.cuda.current_stream(device=device)
                        capture_stream.wait_stream(main_stream)
                        with torch.cuda.stream(capture_stream):
                            for _ in range(2):
                                self._run_loop(
                                    static_embeds,
                                    static_pos_ids,
                                    static_rep_mask,
                                    static_codes,
                                    lm_heads,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p,
                                    do_sample=do_sample,
                                    repetition_penalty=repetition_penalty,
                                )
                        main_stream.wait_stream(capture_stream)

                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=capture_stream):
                            self._run_loop(
                                static_embeds,
                                static_pos_ids,
                                static_rep_mask,
                                static_codes,
                                lm_heads,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                do_sample=do_sample,
                                repetition_penalty=repetition_penalty,
                            )
                        main_stream.wait_stream(capture_stream)
                        self._cuda_graphs[key] = (
                            graph,
                            static_embeds,
                            static_pos_ids,
                            static_rep_mask,
                            static_codes,
                        )
                    except Exception:
                        logger.debug(
                            "MossTTSRealtimeLocalTransformer CUDA graph capture failed for key=%s; using eager loop",
                            key,
                            exc_info=True,
                        )
                        self._uncapturable_keys.add(key)

            cached = self._cuda_graphs.get(key)
            if cached is not None:
                graph, static_embeds, _static_pos_ids, static_rep_mask, static_codes = cached
                static_embeds[:, 0, :].copy_(backbone_last_hidden)
                if repetition_penalty != 1.0:
                    static_rep_mask.zero_()
                    self._populate_rep_mask(
                        static_rep_mask,
                        history_tensors=history_tensors,
                        history_per_codebook=history_per_codebook,
                    )
                graph.replay()
                return static_codes.clone()

        embeds = backbone_last_hidden.new_zeros((B, rvq, hidden_size))
        embeds[:, 0, :] = backbone_last_hidden.to(embeds.dtype)

        pos_ids_buf = getattr(self, "_pos_ids", None)
        if isinstance(pos_ids_buf, torch.Tensor) and pos_ids_buf.device == device and pos_ids_buf.shape[-1] == rvq:
            pos_ids_full = pos_ids_buf.expand(B, -1)
        else:
            pos_ids_full = torch.arange(rvq, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        rep_mask: torch.Tensor | None = None
        if repetition_penalty != 1.0 and (history_tensors is not None or history_per_codebook is not None):
            candidate_mask = torch.zeros((B, rvq, vocab_size), dtype=torch.bool, device=device)
            if self._populate_rep_mask(
                candidate_mask,
                history_tensors=history_tensors,
                history_per_codebook=history_per_codebook,
            ):
                rep_mask = candidate_mask

        codes = backbone_last_hidden.new_zeros((B, rvq), dtype=torch.long)
        self._run_loop(
            embeds,
            pos_ids_full,
            rep_mask,
            codes,
            lm_heads,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        return codes


def _normalize_generators(
    generators: Sequence[torch.Generator | None] | None,
    batch_size: int,
) -> list[torch.Generator | None] | None:
    """Validate per-row generators against the batch they will sample for.

    Mirrors ``Qwen3CodePredictor._normalize_generators``: a length mismatch is
    an error instead of a silent per-row fallback to the global RNG, which would
    quietly break reproducibility of seeded requests.
    """
    if generators is None:
        return None
    row_generators = list(generators)
    if len(row_generators) != batch_size:
        raise ValueError(f"Expected {batch_size} per-row generators, but got {len(row_generators)}.")
    return row_generators


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    generator: torch.Generator | None = None,
    generators: Sequence[torch.Generator | None] | None = None,
) -> torch.Tensor:
    """Top-k + top-p sampling for the upstream inference branch.

    Nucleus filtering and multinomial sampling operate only on the retained
    top-k candidates before the result is mapped back to the original
    vocabulary. This preserves the categorical distribution when the top-k
    boundary has no ties, but it is not seed/bit equivalent to multinomial over
    a full-vocabulary tensor because random-number mapping depends on width.

    When ``generators`` is provided (per-row), each row is sampled with its
    own generator via separate ``multinomial`` calls. This keeps the rest of
    the forward batched while making seeded requests reproducible per-row.
    ``generators`` must hold exactly one entry per batch row; a mismatch raises
    ``ValueError`` rather than silently falling back to the global RNG.
    """
    if not do_sample or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-6)
    compact_indices = None
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        top_vals, top_indices = torch.topk(logits, top_k, dim=-1, sorted=True)
        # topk is already descending: keep the following nucleus filter,
        # softmax, and multinomial at width k instead of the full vocab.
        logits = top_vals
        compact_indices = top_indices

    if 0.0 < top_p < 1.0:
        sorted_indices = None
        if compact_indices is None:
            logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Drop tail beyond top_p (keep at least one token).
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        logits = logits.masked_fill(drop, float("-inf"))
        if sorted_indices is not None:
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_indices, logits)

    probs = F.softmax(logits, dim=-1)
    B = int(probs.shape[0])
    row_generators = _normalize_generators(generators, B)
    if row_generators is not None and any(gen is not None for gen in row_generators):
        rows = []
        for row in range(B):
            row_probs = probs[row : row + 1]
            flat = row_probs.reshape(-1, row_probs.shape[-1])
            sampled = torch.multinomial(flat, num_samples=1, generator=row_generators[row])
            rows.append(sampled)
        sampled = torch.cat(rows, dim=0).reshape(probs.shape[:-1])
    else:
        flat = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat, num_samples=1, generator=generator).reshape(probs.shape[:-1])

    if compact_indices is not None:
        return compact_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return sampled


__all__ = ["MossTTSRealtimeLocalTransformer"]
