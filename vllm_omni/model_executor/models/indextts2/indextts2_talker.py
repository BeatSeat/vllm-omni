# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS2 Stage 0: GPT-2 AR Talker with vLLM-native PagedAttention.

Predicts mel codes autoregressively and collects hidden_states as latent
for Stage 1 (S2Mel + BigVGAN).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt2 import GPT2Block
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_indextts2 import IndexTTS2Config
from .gpt.conformer_encoder import ConformerEncoder
from .gpt.embeddings import LearnedPositionEmbeddings
from .gpt.perceiver import PerceiverResampler
from .preprocess_utils import (
    compute_fbank,
    load_campplus,
    load_qwen_emotion,
    load_reference_audio,
    load_semantic_codec,
    load_wav2vec2,
    resolve_model_file,
    wav2vec_extract,
)

logger = init_logger(__name__)


def _find_most_similar_cosine(query: torch.Tensor, matrix: torch.Tensor) -> int:
    sims = F.cosine_similarity(query.float(), matrix.float(), dim=1)
    return int(torch.argmax(sims).item())


class IndexTTS2TalkerForConditionalGeneration(nn.Module):
    """vLLM-native GPT-2 AR talker for IndexTTS2.

    Stage 0 of the two-stage pipeline. Predicts mel codes (8194 vocab)
    and accumulates hidden_states as latent for Stage 1 S2Mel decoder.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: IndexTTS2Config = vllm_config.model_config.hf_config  # type: ignore[assignment]
        gpt_cfg = self.config.gpt

        self.model_dim = gpt_cfg["model_dim"]
        self.num_layers = gpt_cfg["layers"]
        self.num_heads = gpt_cfg["heads"]
        self.max_mel_tokens = gpt_cfg["max_mel_tokens"]
        self.max_text_tokens = gpt_cfg["max_text_tokens"]
        self.number_mel_codes = gpt_cfg["number_mel_codes"]
        self.start_mel_token = gpt_cfg["start_mel_token"]
        self.stop_mel_token = gpt_cfg["stop_mel_token"]
        self.number_text_tokens = gpt_cfg.get("number_text_tokens", 12000)
        self.condition_num_latent = gpt_cfg.get("condition_num_latent", 32)

        # --- Flags for vLLM-Omni framework ---
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.requires_raw_input_tokens = True
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("codes", "mel"),
            ("hidden_states", "latent"),
            ("meta", "mel_start_offset"),
            ("meta", "latent_acc"),
        }

        # --- GPT-2 transformer (vLLM-native with PagedAttention) ---
        # Build a GPT2Config-like object for GPT2Block
        from transformers import GPT2Config as HFGpt2Config

        max_seq_len = self.max_mel_tokens + self.max_text_tokens + self.condition_num_latent + 4
        gpt2_config = HFGpt2Config(
            vocab_size=self.number_mel_codes,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            n_embd=self.model_dim,
            n_layer=self.num_layers,
            n_head=self.num_heads,
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
        )
        # Store for GPT2Block construction
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.start_layer, self.end_layer, self.h = make_layers(
            gpt2_config.n_layer,
            lambda prefix: GPT2Block(gpt2_config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(self.model_dim, eps=gpt2_config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.model_dim
        )

        # --- Embeddings ---
        self.text_embedding = nn.Embedding(self.number_text_tokens + 1, self.model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, self.model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(self.max_text_tokens + 2, self.model_dim)
        self.mel_pos_embedding = LearnedPositionEmbeddings(self.max_mel_tokens + 2 + 1, self.model_dim)
        self.speed_emb = nn.Embedding(2, self.model_dim)
        self.emo_layer = nn.Linear(self.model_dim, self.model_dim)
        self.emovec_layer = nn.Linear(1024, self.model_dim)

        # --- Mel head (logits) ---
        if get_pp_group().is_last_rank:
            self.mel_head = ParallelLMHead(
                self.number_mel_codes,
                self.model_dim,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "mel_head"),
            )
        else:
            self.mel_head = None  # type: ignore[assignment]
        self.logits_processor = LogitsProcessor(self.number_mel_codes)

        # --- Conditioning modules (speaker / emotion) ---
        cond_cfg = gpt_cfg.get("condition_module", {})
        emo_cond_cfg = gpt_cfg.get("emo_condition_module", {})

        self.conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=cond_cfg.get("output_size", 512),
            linear_units=cond_cfg.get("linear_units", 2048),
            attention_heads=cond_cfg.get("attention_heads", 8),
            num_blocks=cond_cfg.get("num_blocks", 6),
            input_layer=cond_cfg.get("input_layer", "conv2d2"),
        )
        self.perceiver_encoder = PerceiverResampler(
            self.model_dim,
            dim_context=cond_cfg.get("output_size", 512),
            ff_mult=cond_cfg.get("perceiver_mult", 2),
            heads=cond_cfg.get("attention_heads", 8),
            num_latents=self.condition_num_latent,
        )

        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=emo_cond_cfg.get("output_size", 512),
            linear_units=emo_cond_cfg.get("linear_units", 1024),
            attention_heads=emo_cond_cfg.get("attention_heads", 4),
            num_blocks=emo_cond_cfg.get("num_blocks", 4),
            input_layer=emo_cond_cfg.get("input_layer", "conv2d2"),
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=emo_cond_cfg.get("output_size", 512),
            ff_mult=emo_cond_cfg.get("perceiver_mult", 2),
            heads=emo_cond_cfg.get("attention_heads", 4),
            num_latents=1,
        )

        # Padding masks for perceiver cross-attention
        self.cond_mask_pad = nn.ConstantPad1d((self.condition_num_latent, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

        # --- Lazy-loaded external models ---
        self._w2v_stat: torch.Tensor | None = None
        self._emo_matrix: torch.Tensor | None = None
        self._spk_matrix: torch.Tensor | None = None
        self._text_tokenizer: Any = None

        # Initialize embeddings per GPT-2 convention
        for emb in [self.text_embedding, self.mel_embedding]:
            emb.weight.data.normal_(mean=0.0, std=0.02)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        self._decode_step = 0
        self._decode_t0: float = 0.0

    # ------------------------------------------------------------------
    # vLLM required hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.mel_embedding(input_ids)

    def _extract_mel_offsets(self, positions: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
        """Build per-token mel_start_offset from model_intermediate_buffer."""
        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        if not info_dicts:
            if self._decode_step <= 3:
                logger.warning(
                    "[OFFSET] no buffer available at step %d, kwargs keys=%s",
                    self._decode_step,
                    list(kwargs.keys()),
                )
            return torch.zeros_like(positions)
        offsets = []
        for info in info_dicts:
            if isinstance(info, dict):
                meta = info.get("meta", {})
                val = meta.get("mel_start_offset", 0)
                offsets.append(int(val) if not isinstance(val, int) else val)
            else:
                offsets.append(0)
        if self._decode_step <= 3:
            logger.info("[OFFSET] step=%d offsets=%s n_dicts=%d", self._decode_step, offsets, len(info_dicts))
        return torch.tensor(offsets, dtype=positions.dtype, device=positions.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """AR transformer forward.

        During prefill: inputs_embeds is set by preprocess(), input_ids ignored.
        During decode: inputs_embeds is mel_embedding(input_ids) + mel_pos.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                # Fallback: decode without preprocess (should not happen normally)
                logger.warning("[FORWARD] inputs_embeds=None — fallback decode path")
                mel_start_offset = self._extract_mel_offsets(positions, kwargs)
                mel_pos = positions - mel_start_offset
                mel_pos = torch.clamp(mel_pos, min=0)
                inputs_embeds = self.mel_embedding(input_ids) + self.mel_pos_embedding.emb(mel_pos)
            hidden_states = inputs_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.h[self.start_layer : self.end_layer]:
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None or self.mel_head is None:
            return None
        logits = self.logits_processor(self.mel_head, hidden_states)

        step = self._decode_step
        if logits is not None and (step <= 5 or step % 200 == 0):
            flat = logits.view(-1, logits.shape[-1]) if logits.ndim >= 2 else logits.unsqueeze(0)
            first_row = flat[0]
            stop_logit = first_row[self.stop_mel_token].item()
            top5_vals, top5_ids = torch.topk(first_row, min(5, first_row.numel()))
            logger.info(
                "[LOGITS] step=%d shape=%s stop(%d)=%.3f top5_ids=%s top5_vals=%s",
                step,
                logits.shape,
                self.stop_mel_token,
                stop_logit,
                top5_ids.tolist(),
                top5_vals.tolist(),
            )
        return logits

    # ------------------------------------------------------------------
    # Omni multimodal output plumbing
    # ------------------------------------------------------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Collect mel_codes and hidden_states from intermediate buffer."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        mel_codes_list: list[torch.Tensor] = []
        latent_list: list[torch.Tensor] = []
        s_ref = None
        ref_mel = None
        style = None

        source_mel_len = 0
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes = info.get("codes", {})
            mc = codes.get("mel")
            if isinstance(mc, torch.Tensor) and mc.numel() > 0:
                source_mel_len = max(source_mel_len, int(mc.shape[0]))
                # The output processor accumulates multimodal payloads across
                # decode steps.  Preprocess stores cumulative mel codes for the
                # next decode position, so emit only the newest token here.
                mel_codes_list.append(mc[-1:].contiguous())

            # Prefer accumulated latent from postprocess (meta.latent_acc)
            meta = info.get("meta", {})
            lat_acc = meta.get("latent_acc")
            if isinstance(lat_acc, torch.Tensor) and lat_acc.numel() > 0:
                latent_list.append(lat_acc[-1:].contiguous())
            else:
                hs = info.get("hidden_states", {})
                lat = hs.get("latent")
                if isinstance(lat, torch.Tensor) and lat.numel() > 0:
                    latent_list.append(lat[-1:].contiguous())

            if "S_ref" in meta and s_ref is None:
                s_ref = meta["S_ref"]
                logger.info(
                    "[OMNI_OUTPUT] grabbed S_ref=%s from meta",
                    s_ref.shape if isinstance(s_ref, torch.Tensor) else type(s_ref),
                )
            if "ref_mel" in meta and ref_mel is None:
                ref_mel = meta["ref_mel"]
                logger.info(
                    "[OMNI_OUTPUT] grabbed ref_mel=%s from meta",
                    ref_mel.shape if isinstance(ref_mel, torch.Tensor) else type(ref_mel),
                )
            if "style" in meta and style is None:
                style = meta["style"]
                logger.info(
                    "[OMNI_OUTPUT] grabbed style=%s from meta",
                    style.shape if isinstance(style, torch.Tensor) else type(style),
                )

        if not mel_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        mel_codes = torch.cat(mel_codes_list, dim=0)
        latent = torch.cat(latent_list, dim=0) if latent_list else hidden

        logger.info(
            "[OMNI_OUTPUT] mel_codes=%s latent=%s s_ref=%s ref_mel=%s style=%s",
            mel_codes.shape,
            latent.shape if isinstance(latent, torch.Tensor) else "none",
            s_ref.shape if isinstance(s_ref, torch.Tensor) else "none",
            ref_mel.shape if isinstance(ref_mel, torch.Tensor) else "none",
            style.shape if isinstance(style, torch.Tensor) else "none",
        )

        mm: OmniPayload = {
            "codes": {"mel": mel_codes},
            "hidden_states": {"latent": latent},
        }
        # Only emit meta on first decode step (the cumulative source has <=1
        # element). OutputProcessor accumulates multimodal_outputs across steps;
        # emitting static meta every step causes batch-dim inflation.
        emit_meta = source_mel_len <= 1
        if emit_meta:
            meta_out: dict[str, Any] = {}
            if s_ref is not None:
                meta_out["S_ref"] = s_ref
            if ref_mel is not None:
                meta_out["ref_mel"] = ref_mel
            if style is not None:
                meta_out["style"] = style
            if meta_out:
                mm["meta"] = meta_out
            logger.info(
                "[OMNI_OUTPUT] EMIT meta (first step): S_ref=%s ref_mel=%s style=%s",
                s_ref.shape if isinstance(s_ref, torch.Tensor) else None,
                ref_mel.shape if isinstance(ref_mel, torch.Tensor) else None,
                style.shape if isinstance(style, torch.Tensor) else None,
            )
        elif mel_codes.shape[0] % 50 == 0:
            logger.info(
                "[OMNI_OUTPUT] step=%d — meta SKIPPED (only emitted on first step)",
                mel_codes.shape[0],
            )

        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

    # ------------------------------------------------------------------
    # preprocess / postprocess
    # ------------------------------------------------------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build prompt embeddings for prefill; compute mel embeddings for decode.

        Prefill layout:
            [conds(32) + emo_vec(1) + duration(2)] [text_emb + text_pos] [start_mel + mel_pos(0)]
        Decode: mel_embedding(token) + mel_pos_embedding(step).
        """
        span_len = int(input_ids.shape[0])
        meta = info_dict.get("meta", {})

        # --- Decode path: span_len==1 and mel_start_offset already set ---
        if span_len == 1 and isinstance(meta, dict) and meta.get("mel_start_offset") is not None:
            return self._preprocess_decode(input_ids, info_dict, meta)

        # --- Prefill path ---
        return self._preprocess_prefill(input_ids, input_embeds, info_dict, span_len)

    def _preprocess_decode(
        self,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
        meta: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Lightweight decode step: embed token + accumulate codes."""
        device = input_ids.device

        # Accumulate the sampled token from previous step
        existing_codes = info_dict.get("codes", {}).get("mel")
        if isinstance(existing_codes, torch.Tensor) and existing_codes.numel() > 0:
            existing_codes = existing_codes.to(device)
            new_codes = torch.cat([existing_codes, input_ids.detach()])
        else:
            new_codes = input_ids.detach().clone()

        # mel_pos = number of accumulated codes (start_mel was pos 0 in prefill)
        mel_pos = torch.tensor([new_codes.shape[0]], device=device, dtype=torch.long)
        embeds = self.mel_embedding(input_ids) + self.mel_pos_embedding.emb(mel_pos)

        self._decode_step += 1
        if self._decode_step <= 5 or self._decode_step % 200 == 0:
            logger.info(
                "[DECODE_PP] step=%d mel_pos=%d codes_len=%d token=%s",
                self._decode_step,
                int(mel_pos.item()),
                new_codes.shape[0],
                input_ids.tolist(),
            )

        return input_ids, embeds, {"codes": {"mel": new_codes}}

    def _preprocess_prefill(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        info_dict: dict[str, Any],
        span_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Full conditioning pipeline — runs once during prefill."""
        t_start = time.perf_counter()
        logger.info("[PREFILL_PP] start, span_len=%d, device=%s", span_len, input_ids.device)

        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        device = input_ids.device

        text_list = info_dict.get("text")
        if not isinstance(text_list, list) or not text_list or not text_list[0]:
            raise ValueError("Missing additional_information.text for IndexTTS2 talker.")
        text = text_list[0]

        voice_list = info_dict.get("voice")
        voice_path = voice_list[0] if isinstance(voice_list, list) and voice_list else None
        if voice_path is None:
            raise ValueError("IndexTTS2 requires voice (reference audio path) in additional_information.")

        emo_audio_list = info_dict.get("emo_audio")
        emo_audio_path = emo_audio_list[0] if isinstance(emo_audio_list, list) and emo_audio_list else None

        emo_text_list = info_dict.get("emo_text")
        emo_text = emo_text_list[0] if isinstance(emo_text_list, list) and emo_text_list else None

        use_emo_text_list = info_dict.get("use_emo_text")
        use_emo_text = (
            bool(use_emo_text_list[0]) if isinstance(use_emo_text_list, list) and use_emo_text_list else False
        )

        emo_vector_list = info_dict.get("emo_vector")
        emo_vector = emo_vector_list[0] if isinstance(emo_vector_list, list) and emo_vector_list else None

        emo_alpha_list = info_dict.get("emo_alpha")
        emo_alpha = float(emo_alpha_list[0]) if isinstance(emo_alpha_list, list) and emo_alpha_list else 1.0

        use_random_list = info_dict.get("use_random")
        use_random = bool(use_random_list[0]) if isinstance(use_random_list, list) and use_random_list else False

        # --- Load audio and extract features ---
        wav_16k, wav_22k = load_reference_audio(voice_path, device)

        w2v_model, w2v_proc = load_wav2vec2(self.model_path, device)
        self._ensure_w2v_stat_loaded(device)
        spk_cond_emb = wav2vec_extract(wav_16k, w2v_model, w2v_proc, device, self._w2v_stat)

        semantic_codec = load_semantic_codec(self.model_path, self.config.semantic_codec, device)
        logger.info("[PREFILL_PP] spk_cond_emb=%s", spk_cond_emb.shape)
        with torch.no_grad():
            _, s_ref = semantic_codec.quantize(spk_cond_emb)  # [B, T, 1024] quantized embeddings
        logger.info("[PREFILL_PP] S_ref=%s after quantize", s_ref.shape)

        campplus = load_campplus(self.model_path, device)
        fbank = compute_fbank(wav_16k, device)
        logger.info("[PREFILL_PP] fbank=%s", fbank.shape)
        with torch.no_grad():
            style = campplus(fbank)  # [1, 192]
        logger.info("[PREFILL_PP] style=%s", style.shape)

        ref_mel = self._compute_mel_22k(wav_22k, device)  # [1, 80, T_ref]
        logger.info("[PREFILL_PP] ref_mel=%s", ref_mel.shape)

        model_dtype = next(self.conditioning_encoder.parameters()).dtype
        spk_cond_emb = spk_cond_emb.to(device=device, dtype=model_dtype)
        spk_lens = torch.tensor([spk_cond_emb.shape[1]], device=device, dtype=torch.long)
        speech_cond, mask = self.conditioning_encoder(spk_cond_emb, spk_lens)
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        conds = self.perceiver_encoder(speech_cond, conds_mask)  # [1, 32, D]

        emo_vec = self._compute_emotion_vector(
            wav_16k=wav_16k,
            emo_audio_path=emo_audio_path,
            main_text=text,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            emo_vector=emo_vector,
            emo_alpha=emo_alpha,
            use_random=use_random,
            style=style,
            spk_cond_emb=spk_cond_emb,
            w2v_model=w2v_model,
            w2v_proc=w2v_proc,
            device=device,
        )

        speed_zero = torch.zeros(1, device=device, dtype=torch.long)
        speed_one = torch.ones(1, device=device, dtype=torch.long)
        duration_emb = self.speed_emb(speed_zero)
        duration_emb_half = self.speed_emb(speed_one)

        conds_with_emo = conds + emo_vec.unsqueeze(1)  # [1, 32, D]
        conds_prefix = torch.cat(
            [conds_with_emo, duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)],
            dim=1,
        )  # [1, 34, D]

        text_tokens = self._tokenize_text(text, device)  # [1, L]
        text_emb = self.text_embedding(text_tokens) + self.text_pos_embedding(text_tokens)

        start_mel = self.mel_embedding(torch.tensor([[self.start_mel_token]], device=device, dtype=torch.long))
        start_mel_pos = self.mel_pos_embedding.get_fixed_embedding(0, device)
        start_mel = start_mel + start_mel_pos  # [1, 1, D]

        inputs_embeds = torch.cat([conds_prefix, text_emb, start_mel], dim=1).squeeze(0)
        mel_start_offset = int(conds_prefix.shape[1]) + int(text_emb.shape[1])

        if int(inputs_embeds.shape[0]) < span_len:
            pad_n = span_len - int(inputs_embeds.shape[0])
            pad_emb = torch.zeros(pad_n, self.model_dim, device=device, dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, pad_emb], dim=0)
        elif int(inputs_embeds.shape[0]) > span_len:
            inputs_embeds = inputs_embeds[:span_len]

        input_ids_out = torch.full_like(input_ids, self.start_mel_token)

        logger.info(
            "[PREFILL_PP] info_update shapes — S_ref=%s, ref_mel=%s, style=%s",
            s_ref.shape,
            ref_mel.shape,
            style.shape,
        )
        info_update: dict[str, Any] = {
            "meta": {
                "mel_start_offset": mel_start_offset,
                "S_ref": s_ref.cpu().contiguous(),
                "ref_mel": ref_mel.cpu().contiguous(),
                "style": style.cpu().contiguous(),
                "latent_acc": torch.zeros(0, self.model_dim, dtype=inputs_embeds.dtype, device=device),
            },
            "codes": {"mel": torch.zeros(0, dtype=torch.long, device=device)},
            "hidden_states": {"latent": torch.zeros(0, self.model_dim, dtype=inputs_embeds.dtype, device=device)},
        }

        t_elapsed = time.perf_counter() - t_start
        logger.info(
            "[PREFILL_PP] done in %.2fs — mel_start_offset=%d, inputs_embeds=%s, text_tokens=%d",
            t_elapsed,
            mel_start_offset,
            inputs_embeds.shape,
            int(text_emb.shape[1]),
        )
        self._decode_step = 0
        self._decode_t0 = time.perf_counter()

        return input_ids_out, inputs_embeds, info_update

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Accumulate per-step hidden states into meta.latent_acc for Stage 1.

        Only accumulate decode steps (single token). Prefill hidden states
        (multiple tokens) belong to conditioning/text, not mel generation.
        """
        if hidden_states.shape[0] != 1:
            return {}

        meta = kwargs.get("meta", {})
        existing_latent = meta.get("latent_acc")

        hs = hidden_states.detach()

        if isinstance(existing_latent, torch.Tensor) and existing_latent.numel() > 0:
            existing_latent = existing_latent.to(hs.device)
            new_latent = torch.cat([existing_latent, hs], dim=0)
        else:
            new_latent = hs

        step = new_latent.shape[0]
        if step <= 3 or step % 50 == 0:
            logger.info(
                "[POSTPROCESS] latent_acc step=%d shape=%s hs=%s",
                step,
                new_latent.shape,
                hs.shape,
            )

        return {"meta": {"latent_acc": new_latent}}

    def _compute_mel_22k(self, wav_22k: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute 80-band mel spectrogram at 22.05kHz for Stage 1."""
        s2mel_cfg = self.config.s2mel.get("preprocess_params", {})
        sr = s2mel_cfg.get("sr", 22050)
        spect = s2mel_cfg.get("spect_params", {})
        n_fft = spect.get("n_fft", 1024)
        hop_length = spect.get("hop_length", 256)
        win_length = spect.get("win_length", 1024)
        n_mels = spect.get("n_mels", 80)

        wav = wav_22k.float().cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        from .s2mel.modules.audio import mel_spectrogram as mel_fn

        mel = mel_fn(wav, n_fft, n_mels, sr, hop_length, win_length, 0, None)
        return mel.to(device=device)  # [1, 80, T]

    # ------------------------------------------------------------------
    # Emotion conditioning
    # ------------------------------------------------------------------

    def _compute_emotion_vector(
        self,
        *,
        wav_16k: torch.Tensor,
        emo_audio_path: str | None,
        main_text: str,
        use_emo_text: bool,
        emo_text: str | None,
        emo_vector: Any,
        emo_alpha: float,
        use_random: bool,
        style: torch.Tensor,
        spk_cond_emb: torch.Tensor,
        w2v_model: Any,
        w2v_proc: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """Full emotion control logic aligned with official infer_v2.py. Returns [1, D]."""
        # --- Mutual exclusion: emo_vector or emo_text clears emo_audio ---
        if use_emo_text or emo_vector is not None:
            emo_audio_path = None

        # --- Path A: emo_text → QwenEmotion CausalLM → emo_vector ---
        if use_emo_text and emo_vector is None:
            if emo_text is None:
                emo_text = main_text
            emo_vector = self._predict_emotion_from_text(emo_text, device)

        # --- Apply normalize + emo_alpha scaling to emo_vector ---
        if emo_vector is not None:
            if isinstance(emo_vector, torch.Tensor):
                emo_vector = emo_vector.tolist()
            elif isinstance(emo_vector, np.ndarray):
                emo_vector = emo_vector.tolist()
            emo_vector = self._normalize_emo_vec(emo_vector, apply_bias=True)
            alpha_scale = max(0.0, min(1.0, emo_alpha))
            if alpha_scale != 1.0:
                emo_vector = [int(x * alpha_scale * 10000) / 10000 for x in emo_vector]

        # --- Path B: emo_vector (8-dim) → spk_matrix cosine lookup ---
        emovec_mat = None
        if emo_vector is not None:
            vec = torch.tensor(emo_vector, device=device, dtype=torch.float32)
            emovec_mat = self._compute_emo_from_distribution(vec, style, device, use_random=use_random)

        # --- Audio-based emotion with alpha blending ---
        if emo_audio_path is None:
            effective_alpha = 1.0
            emo_wav_16k = wav_16k
        else:
            effective_alpha = emo_alpha
            emo_wav_16k, _ = load_reference_audio(emo_audio_path, device)

        self._ensure_w2v_stat_loaded(device)
        emo_cond_emb = wav2vec_extract(emo_wav_16k, w2v_model, w2v_proc, device, self._w2v_stat)
        emovec = self._merge_emovec(spk_cond_emb, emo_cond_emb, effective_alpha)

        # --- Overlay emo_vector onto audio-based emovec ---
        if emovec_mat is not None:
            weight_sum = torch.tensor(emo_vector, device=device, dtype=torch.float32).sum()
            emovec = emovec_mat + (1.0 - weight_sum) * emovec

        return emovec

    _MELANCHOLIC_WORDS = frozenset(
        {
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
    )

    _CN_KEY_TO_EN = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    _DESIRED_ORDER = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]

    _EMO_BIAS = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]

    @staticmethod
    def _normalize_emo_vec(emo_vector: list[float], apply_bias: bool = True) -> list[float]:
        """Apply biased dampening and cap total sum at 0.8 (aligned with official)."""
        if apply_bias:
            bias = IndexTTS2TalkerForConditionalGeneration._EMO_BIAS
            emo_vector = [v * b for v, b in zip(emo_vector, bias)]
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale = 0.8 / emo_sum
            emo_vector = [v * scale for v in emo_vector]
        return emo_vector

    def _predict_emotion_from_text(self, text: str, device: torch.device) -> list[float] | None:
        """Use QwenEmotion CausalLM to predict 8-dim emotion vector from text (aligned with official)."""
        model, tokenizer = load_qwen_emotion(self.model_path, device)
        if model is None or tokenizer is None:
            return None

        messages = [
            {"role": "system", "content": "文本情感分类"},
            {"role": "user", "content": text},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content_str = tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = {
                m.group(1): float(m.group(2)) for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content_str)
            }

        text_lower = text.lower()
        if any(w in text_lower for w in self._MELANCHOLIC_WORDS):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)

        max_score, min_score = 1.2, 0.0
        emo_vector = []
        for cn_key in self._DESIRED_ORDER:
            val = content.get(cn_key, 0.0)
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0
            emo_vector.append(max(min_score, min(max_score, val)))

        if all(v <= 0.0 for v in emo_vector):
            emo_vector[7] = 1.0  # default calm

        return emo_vector

    def _compute_emo_from_distribution(
        self,
        emo_dist: torch.Tensor,
        style: torch.Tensor,
        device: torch.device,
        use_random: bool = False,
    ) -> torch.Tensor:
        """From 8-dim emotion distribution + CAMPPlus style, build emovec_mat via spk_matrix cosine lookup."""
        self._ensure_emo_matrix_loaded(device)
        self._ensure_spk_matrix_loaded(device)
        if self._emo_matrix is None or self._spk_matrix is None:
            return torch.zeros(1, self.model_dim, device=device)

        weight_vector = emo_dist.to(device=device, dtype=torch.float32)
        if weight_vector.ndim == 2:
            weight_vector = weight_vector.squeeze(0)  # [8]

        if use_random:
            indices = [int(torch.randint(0, sub.shape[0], (1,), device=device).item()) for sub in self._spk_matrix]
        else:
            indices = [_find_most_similar_cosine(style, sub.to(device)) for sub in self._spk_matrix]
        emo_rows = torch.cat(
            [sub.to(device)[idx].unsqueeze(0) for idx, sub in zip(indices, self._emo_matrix)], dim=0
        )  # [8, 1024]

        emovec_mat = (weight_vector.unsqueeze(1) * emo_rows).sum(0).unsqueeze(0)
        if emovec_mat.shape[-1] == self.emovec_layer.in_features:
            emovec_mat = self.emovec_layer(emovec_mat.to(dtype=self.emovec_layer.weight.dtype))
        elif emovec_mat.shape[-1] != self.model_dim:
            raise ValueError(
                f"Unexpected IndexTTS2 emotion matrix dim {emovec_mat.shape[-1]}; "
                f"expected {self.emovec_layer.in_features} or {self.model_dim}."
            )
        return self.emo_layer(emovec_mat.to(dtype=self.emo_layer.weight.dtype))  # [1, D]

    def _compute_audio_emo_vec(self, cond_emb: torch.Tensor) -> torch.Tensor:
        """Conformer → Perceiver → emovec_layer → emo_layer. Input: [1, T, 1024], output: [1, D]."""
        emo_dtype = next(self.emo_conditioning_encoder.parameters()).dtype
        cond_emb = cond_emb.to(device=cond_emb.device, dtype=emo_dtype)
        cond_lens = torch.tensor([cond_emb.shape[1]], device=cond_emb.device, dtype=torch.long)
        emo_cond, emo_mask = self.emo_conditioning_encoder(cond_emb, cond_lens)
        emo_conds_mask = self.emo_cond_mask_pad(emo_mask.squeeze(1))
        emo_percept = self.emo_perceiver_encoder(emo_cond, emo_conds_mask)  # [1, 1, 1024]
        emo_vec_syn = self.emovec_layer(emo_percept.squeeze(1))  # [1, D]
        return self.emo_layer(emo_vec_syn)  # [1, D]

    def _merge_emovec(
        self,
        spk_cond_emb: torch.Tensor,
        emo_cond_emb: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Alpha-blend speaker and emotion audio vectors: base + alpha * (emo - base)."""
        base_vec = self._compute_audio_emo_vec(spk_cond_emb)
        emo_vec = self._compute_audio_emo_vec(emo_cond_emb)
        return base_vec + alpha * (emo_vec - base_vec)  # [1, D]

    # ------------------------------------------------------------------
    # Text tokenizer
    # ------------------------------------------------------------------

    def _tokenize_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Tokenize text using the BPE tokenizer, add start/stop tokens."""
        if self._text_tokenizer is None:
            from .tokenizer import IndexTTS2Tokenizer

            bpe_path = resolve_model_file(self.model_path, "bpe.model")
            if bpe_path is None:
                raise FileNotFoundError(f"BPE model not found in {self.model_path}")
            self._text_tokenizer = IndexTTS2Tokenizer(bpe_path, model_dir=self.model_path)

        token_ids = self._text_tokenizer.encode(text, add_special_tokens=False)

        # Wrap with start/stop text tokens
        start_text = 0
        stop_text = 1
        token_ids = [start_text] + token_ids + [stop_text]
        return torch.tensor([token_ids], device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # Lazy loading helpers
    # ------------------------------------------------------------------

    def _ensure_w2v_stat_loaded(self, device: torch.device) -> None:
        if self._w2v_stat is not None:
            return
        stat_path = resolve_model_file(self.model_path, self.config.w2v_stat)
        if stat_path is not None:
            raw = torch.load(stat_path, map_location="cpu", weights_only=True)
            if isinstance(raw, dict):
                self._w2v_stat = torch.stack([raw["mean"], raw["var"]])  # [2, 1024]
            else:
                self._w2v_stat = raw
        else:
            logger.warning("Wav2Vec2-BERT stat file not found in %s", self.model_path)

    def _ensure_emo_matrix_loaded(self, device: torch.device) -> None:
        if self._emo_matrix is not None:
            return
        path = resolve_model_file(self.model_path, self.config.emo_matrix)
        if path is not None:
            raw = torch.load(path, map_location="cpu", weights_only=True)
            self._emo_matrix = torch.split(raw, list(self.config.emo_num))
        else:
            logger.warning("Emotion matrix not found in %s", self.model_path)

    def _ensure_spk_matrix_loaded(self, device: torch.device) -> None:
        if self._spk_matrix is not None:
            return
        path = resolve_model_file(self.model_path, self.config.spk_matrix)
        if path is not None:
            raw = torch.load(path, map_location="cpu", weights_only=True)
            self._spk_matrix = torch.split(raw, list(self.config.emo_num))
        else:
            logger.warning("Speaker matrix not found in %s", self.model_path)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from IndexTTS2 checkpoint (gpt.pth).

        Bypasses vllm's default .pt iterator because the model directory
        contains raw-tensor files (feat1.pt, feat2.pt) that are not state
        dicts and would crash ``pt_weights_iterator``.

        Weight mapping from gpt.pth checkpoint → vLLM model params:
          gpt.h.{i}.*                → h.{i}.*  (strip ``gpt.`` prefix)
          gpt.ln_f.*                 → ln_f.*
          final_norm.*               → ln_f.*   (alias)
          text_head.*                → (skipped)
          <everything else>          → identity
        """
        _ = weights  # don't iterate — raw .pt files crash the default iterator

        ckpt_path = resolve_model_file(self.model_path, self.config.gpt_checkpoint)
        if ckpt_path is None:
            raise FileNotFoundError(f"IndexTTS2 GPT checkpoint {self.config.gpt_checkpoint!r} not found")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        prefix_map = {
            "gpt.h.": "h.",
            "gpt.ln_f.": "ln_f.",
            "final_norm.": "ln_f.",
            "text_head.": "_skip_text_head.",
        }

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in state.items():
            # Skip attention masks
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            # Skip wpe (null position embeddings)
            if ".wpe." in name or ".wte." in name:
                continue

            # Remap checkpoint name to model name
            mapped_name = name
            for old_prefix, new_prefix in prefix_map.items():
                if name.startswith(old_prefix):
                    mapped_name = new_prefix + name[len(old_prefix) :]
                    break

            # Skip unmapped / intentionally skipped weights
            if mapped_name.startswith("_skip_"):
                continue

            if is_pp_missing_parameter(mapped_name, self):
                continue

            if mapped_name not in params_dict:
                logger.debug("Skipping unrecognized weight: %s → %s", name, mapped_name)
                continue

            param = params_dict[mapped_name]

            # GPT-2 Conv1D → Linear transpose
            for conv1d_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_name not in mapped_name:
                    continue
                if not mapped_name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped_name)

        logger.info(
            "Loaded %d weights for IndexTTS2TalkerForConditionalGeneration",
            len(loaded_params),
        )
        required_prefixes = [
            "text_embedding.",
            "mel_embedding.",
            "text_pos_embedding.",
            "mel_pos_embedding.",
            "speed_emb.",
            "emo_layer.",
            "emovec_layer.",
            "conditioning_encoder.",
            "perceiver_encoder.",
            "emo_conditioning_encoder.",
            "emo_perceiver_encoder.",
            "h.",
            "ln_f.",
        ]
        if self.mel_head is not None:
            required_prefixes.append("mel_head.")
        missing_prefixes = [
            prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in loaded_params)
        ]
        if missing_prefixes:
            raise RuntimeError(
                "IndexTTS2 GPT checkpoint did not load required parameter groups: "
                + ", ".join(missing_prefixes)
            )

        # Ensure all sub-modules on correct device (some nn.Parameter created
        # with torch.Tensor() end up on CPU even when model is on CUDA).
        target_device = next((p.device for p in self.h.parameters()), torch.device("cpu"))
        for mod in [
            self.conditioning_encoder,
            self.emo_conditioning_encoder,
            self.perceiver_encoder,
            self.emo_perceiver_encoder,
        ]:
            mod.to(target_device)

        return loaded_params
