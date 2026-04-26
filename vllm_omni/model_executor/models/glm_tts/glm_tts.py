# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS AR Model (Stage 0): Text → Speech Tokens.

Based on Llama architecture, generates speech token sequences from input text.
Analogous to Fish Speech Slow AR and Qwen3-TTS Talker models.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_glm_tts import GLMTTSConfig
from .sampling import (
    log_sampling_debug,
    req_float,
    req_scalar,
    sample_ras_one,
    sample_topk_one,
)
from .text_frontend import GLMTTSTextFrontend
from .voice_clone import (
    extract_prompt_feat,
    extract_prompt_speech_token,
    extract_spk_embedding,
    load_voice_clone_frontend,
)

logger = init_logger(__name__)


def estimate_prompt_len_from_text(
    *,
    text: str,
    tokenizer: Any,
    text_frontend: GLMTTSTextFrontend | None = None,
    prompt_text: str | None = None,
    prompt_speech_token_len: int = 0,
) -> int:
    """Estimate the placeholder length required by model-side prefill."""
    frontend = text_frontend or GLMTTSTextFrontend()

    def _normalize(value: str | None, *, add_trailing_space: bool = False) -> str:
        if value is None:
            return ""
        normalized = frontend.text_normalize(value)
        normalized = (normalized or value).strip()
        if add_trailing_space and normalized:
            normalized = f"{normalized} "
        return normalized

    def _token_len(value: str) -> int:
        return int(len(tokenizer.encode(value)))

    text_len = _token_len(_normalize(text))
    if prompt_text is None or prompt_speech_token_len <= 0:
        return max(1, text_len + 1)  # [Text | BOA]

    prompt_text_len = _token_len(_normalize(prompt_text, add_trailing_space=True))
    return max(1, prompt_text_len + text_len + 1 + int(prompt_speech_token_len))


class GLMTTSForConditionalGeneration(nn.Module):
    """vLLM model for GLM-TTS.

    Handles both stages via model_stage branching:
      - ``glm_tts`` (Stage 0): Text → Speech tokens (LLM AR, Llama backbone).
      - ``glm_tts_dit`` (Stage 1): Speech tokens → Audio (DiT flow-matching + vocoder).

    Attributes:
        have_multimodal_outputs: Signals scheduler to collect multimodal outputs.
        has_preprocess: Model has preprocess hook for input preparation (stage 0 only).
        has_postprocess: Model has postprocess hook for hidden state caching (stage 0 only).
    """

    prefer_model_sampler = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.model_stage = getattr(vllm_config.model_config, "model_stage", "glm_tts")

        # Load configuration
        config: GLMTTSConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config

        # ---- Stage 1 (DiT): delegate to GLMTTSDiTForGeneration ----
        if self.model_stage == "glm_tts_dit":
            from .glm_tts_dit_wrapper import GLMTTSDiTForGeneration

            self._dit_gen = GLMTTSDiTForGeneration(vllm_config=vllm_config, prefix=prefix)
            # Expose the DiT as self.model for weight loading routing
            self.model = self._dit_gen
            self.have_multimodal_outputs = True
            self.enable_update_additional_information = True
            # DiT stage does not use preprocess/postprocess/sample
            self.has_preprocess = False
            self.has_postprocess = False
            # Prevent vLLM DefaultModelLoader from loading the ~7GB Llama
            # safetensors weights. DiT loads flow/flow.pt in load_weights().
            self.allow_patterns_overrides = ["flow.pt"]
            self.fall_back_to_pt_during_load = False
            return
        self._sample_method = str(getattr(config, "sample_method", "ras")).lower()
        if self._sample_method not in {"ras", "topk"}:
            raise ValueError(f"Unsupported GLM-TTS sample_method={self._sample_method!r}; expected 'ras' or 'topk'.")

        # Resolve model_root (repo root) from model_path.
        # With model_subdir=llm in stage config, model_path points to {root}/llm/.
        # model_root is the parent directory containing sibling resources:
        #   vq32k-phoneme-tokenizer/, ckpt/, frontend/, flow/, vocos/
        model_root = self.model_path
        if os.path.basename(model_root.rstrip("/\\")) == "llm":
            parent = os.path.dirname(model_root.rstrip("/\\"))
            if os.path.isdir(os.path.join(parent, "vq32k-phoneme-tokenizer")):
                model_root = parent
        self._model_root = model_root

        # Load tokenizer for special token ID resolution.
        # Prefer the tokenizer path set by tokenizer_subdir in stage config
        # (resolved by _resolve_model_tokenizer_paths in stage_init_utils).
        # Fall back to vq32k-phoneme-tokenizer/ under model_root.
        tokenizer_path = getattr(vllm_config.model_config, "tokenizer", None)
        if tokenizer_path and os.path.isdir(tokenizer_path):
            pass  # Use the tokenizer path from engine_args
        else:
            tokenizer_path = os.path.join(model_root, "vq32k-phoneme-tokenizer")
            if not os.path.exists(tokenizer_path):
                tokenizer_path = self.model_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        special_ids = self._get_special_token_ids()
        self._ats = special_ids["ats"]
        self._ate = special_ids["ate"]
        self._boa = special_ids["boa"]
        self._eoa = special_ids["eoa"]
        self._pad = special_ids["pad"]
        self._bos = self._tokenizer.bos_token_id or self._pad

        logger.info(
            "GLM-TTS token IDs: ATS=%d, ATE=%d, BOA=%d, EOA=%d, PAD=%d, vocab_size=%d",
            self._ats,
            self._ate,
            self._boa,
            self._eoa,
            self._pad,
            config.vocab_size,
        )

        # Assert: special token sanity
        assert self._ate - self._ats == 32767, f"Audio token range should be 32768, got {self._ate - self._ats + 1}"
        assert self._ats < self._ate, f"ATS={self._ats} should be < ATE={self._ate}"
        assert self._boa < self._ats, f"BOA={self._boa} should be < ATS={self._ats} (BOA is text token)"

        # Validate vocab_size covers all special tokens
        max_token = max(self._ats, self._ate, self._boa, self._eoa, self._pad)
        if max_token >= config.vocab_size:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must be > max token ID ({max_token}). "
                f"Check model's config.json has correct vocab_size."
            )

        # Update config with dynamic token IDs so vLLM uses correct eos_token
        # This enables proper stop detection when EOA is sampled
        config.eos_token_id = self._eoa
        config.eoa_token_id = self._eoa
        config.audio_token_start = self._ats
        config.audio_token_end = self._ate
        config.boa_token_id = self._boa
        config.pad_token_id = self._pad
        config.bos_token_id = self._bos

        # Model flags for AR scheduler
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.gpu_resident_buffer_keys: set[str] = {"last_hidden"}

        # Voice cloning frontend models (lazy-loaded)
        self._speech_tokenizer = None
        self._campplus_session = None
        self._voice_clone_frontend_loaded = False
        self._text_frontend: GLMTTSTextFrontend | None = None

        # Llama transformer backbone
        self.model = LlamaModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # LM head for speech token prediction
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Logit mask: only allow audio tokens [ATS, ATE] plus EOA
        vocab = int(config.vocab_size)
        audio_mask = torch.zeros((vocab,), dtype=torch.bool)
        audio_mask[self._ats : self._ate + 1] = True
        audio_mask[self._eoa] = True
        self.register_buffer("_audio_allowed_mask", audio_mask, persistent=False)

    def _get_special_token_ids(self) -> dict[str, int]:
        """Get special token IDs from tokenizer dynamically.

        Based on GLM-TTS original: glmtts_inference.py:72-103
        """
        tokenizer = self._tokenizer
        special_tokens = {
            "ats": "<|audio_0|>",
            "ate": "<|audio_32767|>",
            "boa": "<|begin_of_audio|>",
            "eoa": "<|user|>",
            "pad": "<|endoftext|>",
        }

        result = {}
        for key, token_str in special_tokens.items():
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Token '{key}' ({token_str}) should encode to single ID, got: {token_ids}")
            result[key] = token_ids[0]

        return result

    def _get_tokenizer(self):
        """Return cached tokenizer (loaded in __init__)."""
        return self._tokenizer

    def _get_text_frontend(self) -> GLMTTSTextFrontend:
        if self._text_frontend is None:
            self._text_frontend = GLMTTSTextFrontend()
        return self._text_frontend

    def _normalize_text_for_tokens(
        self,
        text: str | None,
        *,
        add_trailing_space: bool = False,
    ) -> str:
        if text is None:
            return ""
        normalized = self._get_text_frontend().text_normalize(text)
        normalized = (normalized or text).strip()
        if add_trailing_space and normalized:
            normalized = f"{normalized} "
        return normalized

    estimate_prompt_len_from_text = staticmethod(estimate_prompt_len_from_text)

    def _encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        token_ids = self._get_tokenizer().encode(text)
        return torch.tensor([token_ids], device=device, dtype=torch.long)

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.embed_input_ids(input_ids, **kwargs)
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.forward(input_ids, positions, **kwargs)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.compute_logits(hidden_states, sampling_metadata)
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None

        # Mask out non-audio tokens
        logits = logits.masked_fill(~self._audio_allowed_mask, float("-inf"))

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if logits is None or logits.numel() == 0:
            return None

        # --- GLM-TTS min_len/max_len constraint (per-request) ---
        # Based on official GLM-TTS inference logic (glmtts.py):
        # - min_len = text_token_len * min_token_text_ratio (default 2.0)
        # - max_len = text_token_len * max_token_text_ratio (default 20.0)
        # Before min_len: mask EOA to prevent premature termination
        # At max_len: force EOA by masking all other tokens
        min_ratio = float(getattr(self.config, "min_token_text_ratio", 2.0))
        max_ratio = float(getattr(self.config, "max_token_text_ratio", 20.0))
        text_token_lens = self._consume_pending_text_token_lens(int(logits.shape[0]))

        logits = logits.clone()  # Always clone to avoid modifying original
        num_reqs = int(logits.shape[0])
        for req_idx in range(num_reqs):
            text_token_len = text_token_lens[req_idx]
            min_len = int(text_token_len * min_ratio)
            max_len = int(text_token_len * max_ratio)
            current_step = (
                len(sampling_metadata.output_token_ids[req_idx])
                if req_idx < len(sampling_metadata.output_token_ids)
                else 0
            )
            if current_step < min_len:
                logits[req_idx, self._eoa] = float("-inf")
            elif current_step >= max_len:
                mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
                mask[self._eoa] = False
                logits[req_idx].masked_fill_(mask, float("-inf"))

        sampler = getattr(self, "_glm_tts_sampler", None)
        if sampler is None:
            sampler = Sampler()
            self._glm_tts_sampler = sampler

        if sampling_metadata.max_num_logprobs is not None:
            return sampler(logits=logits, sampling_metadata=sampling_metadata)

        logits = logits.to(torch.float32)
        sampling_for_processors = replace(sampling_metadata, no_penalties=True)
        logits = sampler.apply_logits_processors(
            logits,
            sampling_for_processors,
            predict_bonus_token=False,
        )

        sampled_ids: list[int] = []
        sample_method = self._sample_method
        ras_top_p = float(getattr(self.config, "ras_top_p", 0.8))
        ras_top_k = int(getattr(self.config, "ras_top_k", 25))
        ras_win_size = int(getattr(self.config, "ras_win_size", 10))
        ras_tau_r = float(getattr(self.config, "ras_tau_r", 0.1))
        default_top_k = int(getattr(self.config, "sampling_top_k", 25))

        for req_idx in range(int(logits.shape[0])):
            row_logits = logits[req_idx]
            generator = sampling_metadata.generators.get(req_idx)
            decoded_tokens = (
                sampling_metadata.output_token_ids[req_idx] if req_idx < len(sampling_metadata.output_token_ids) else []
            )
            weighted_scores = torch.log_softmax(row_logits, dim=0)
            temperature = max(
                req_float(sampling_metadata.temperature, req_idx, 1.0),
                1e-5,
            )

            if sample_method == "ras":
                top_p = req_float(sampling_metadata.top_p, req_idx, ras_top_p)
                top_k = req_scalar(sampling_metadata.top_k, req_idx, ras_top_k)
                sampled_id = sample_ras_one(
                    weighted_scores,
                    decoded_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    win_size=ras_win_size,
                    tau_r=ras_tau_r,
                    temperature=temperature,
                    generator=generator,
                )
                sampled_ids.append(sampled_id)
                log_sampling_debug(
                    req_idx=req_idx,
                    weighted_scores=weighted_scores,
                    decoded_tokens=decoded_tokens,
                    sampled_id=sampled_id,
                    sample_method=sample_method,
                    eoa_token_id=self._eoa,
                )
                continue

            top_k = req_scalar(sampling_metadata.top_k, req_idx, default_top_k)
            sampled_id = sample_topk_one(
                weighted_scores / temperature,
                top_k=top_k,
                eoa_token_id=self._eoa,
                ignore_eos=False,
                generator=generator,
            )
            sampled_ids.append(sampled_id)
            log_sampling_debug(
                req_idx=req_idx,
                weighted_scores=weighted_scores,
                decoded_tokens=decoded_tokens,
                sampled_id=sampled_id,
                sample_method=sample_method,
                eoa_token_id=self._eoa,
            )

        sampled = torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Package hidden states, speech tokens, and voice clone data into OmniOutput.

        Streaming contract: **delta**.  Each decode step emits exactly one
        speech token (or a prefill placeholder).  The engine's output
        processor concatenates per-step deltas into the final tensor.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        speech_tokens_list: list[torch.Tensor] = []
        multimodal_extras: dict[str, Any] = {}
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            tokens = info.get("speech_tokens")
            if isinstance(tokens, torch.Tensor) and tokens.numel() > 0:
                speech_tokens_list.append(tokens)
            # Propagate voice cloning features from preprocess info_update.
            # IMPORTANT: Only emit once — the output processor accumulates
            # every decode step and concatenates tensors, which would corrupt
            # constant data (e.g. embedding [192] → [N*192]).
            if not info.get("_voice_clone_emitted"):
                for key in ("prompt_token", "prompt_feat", "embedding"):
                    val = info.get(key)
                    if val is not None and key not in multimodal_extras:
                        multimodal_extras[key] = val
                if any(info.get(k) is not None for k in ("prompt_token", "prompt_feat", "embedding")):
                    info["_voice_clone_emitted"] = True

        if not speech_tokens_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs=multimodal_extras)

        speech_tokens = torch.cat(speech_tokens_list, dim=0)
        span_len = int(speech_tokens.shape[0])
        hidden = hidden[:span_len]
        multimodal_extras["speech_tokens"] = speech_tokens
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_extras,
        )

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prepare inputs for GLM-TTS AR model.

        Handles:
        - Text tokenization
        - Prompt construction with special tokens
        - Voice cloning prompt integration
        """
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        logger.debug("preprocess: span_len=%d, input_ids.shape=%s", span_len, input_ids.shape)
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        # Get input text (scalar, not list-wrapped — matches Fish Speech pattern)
        text = info_dict.get("text")
        if not text or not isinstance(text, str):
            raise ValueError("Missing additional_information.text for GLM-TTS AR model.")
        device = input_ids.device

        # First invocation may carry a single placeholder token even though we
        # still need to build the full GLM-TTS prompt embeddings on model side.
        prompt_embeds_cpu = info_dict.get("glm_tts_prompt_embeds")
        is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2
        is_prefill = bool(is_first_prefill or span_len > 1)

        if is_prefill:
            if is_first_prefill:
                prompt_embeds, text_token_len = self._build_prompt_embeds(
                    text=text,
                    info_dict=info_dict,
                    device=device,
                )
                self._record_pending_text_token_len(text_token_len)
                prompt_embeds_cpu = prompt_embeds.detach().to("cpu").contiguous()

                info_update: dict[str, Any] = {
                    "glm_tts_prompt_embeds": prompt_embeds_cpu,
                    "glm_tts_prefill_offset": 0,
                    "glm_tts_text_token_len": text_token_len,
                }

                # Extract and propagate voice cloning features for DiT stage.
                # Source: raw ref_audio_wav (online serving) or pre-extracted (offline).
                ref_audio_wav = info_dict.get("ref_audio_wav")
                ref_audio_sr = info_dict.get("ref_audio_sr")
                has_ref_audio = isinstance(ref_audio_wav, torch.Tensor)

                if has_ref_audio:
                    sr = int(ref_audio_sr or 24000)
                    prompt_speech_token = self._extract_prompt_speech_token(ref_audio_wav, sr)
                    if prompt_speech_token is not None:
                        # Convert list[int] → Tensor for msgspec serialization
                        info_update["prompt_token"] = torch.tensor(prompt_speech_token, dtype=torch.long)
                    prompt_feat = self._extract_prompt_feat(ref_audio_wav, sr)
                    if prompt_feat is not None:
                        info_update["prompt_feat"] = prompt_feat.cpu()
                    embedding = self._extract_spk_embedding(ref_audio_wav, sr)
                    if embedding is not None:
                        # Convert list[float] → Tensor for msgspec serialization
                        info_update["embedding"] = torch.tensor(embedding, dtype=torch.float32)
                else:
                    for key in ("prompt_token", "prompt_feat", "embedding"):
                        val = info_dict.get(key)
                        if val is not None:
                            # Ensure all values are Tensors for serialization
                            if isinstance(val, list):
                                val = torch.tensor(val)
                            info_update[key] = val

                if prompt_embeds.shape[0] > span_len:
                    prompt_embeds = prompt_embeds[:span_len]
                elif prompt_embeds.shape[0] < span_len:
                    pad_embed = self.embed_input_ids(
                        torch.tensor([[self._pad]], device=device, dtype=torch.long)
                    ).squeeze(0)
                    pad_n = span_len - prompt_embeds.shape[0]
                    pad_rows = pad_embed.expand(pad_n, -1)
                    prompt_embeds = torch.cat([prompt_embeds, pad_rows], dim=0)

                info_update["glm_tts_prefill_offset"] = int(span_len)
            else:
                offset = int(info_dict.get("glm_tts_prefill_offset", 0) or 0)
                text_token_len = int(info_dict.get("glm_tts_text_token_len", 10) or 10)
                self._record_pending_text_token_len(text_token_len)
                s = max(0, min(offset, int(prompt_embeds_cpu.shape[0])))
                e = max(0, min(offset + span_len, int(prompt_embeds_cpu.shape[0])))
                take = prompt_embeds_cpu[s:e]

                if int(take.shape[0]) < span_len:
                    pad_embed = self.embed_input_ids(
                        torch.tensor([[self._pad]], device=device, dtype=torch.long)
                    ).squeeze(0)
                    pad_n = span_len - int(take.shape[0])
                    pad_rows = pad_embed.to("cpu").expand(pad_n, -1)
                    take = torch.cat([take, pad_rows], dim=0)

                prompt_embeds = take.to(device=device, dtype=torch.bfloat16)
                info_update = {
                    "glm_tts_prefill_offset": int(offset + span_len),
                    "glm_tts_text_token_len": text_token_len,
                }

            input_ids_out = input_ids.clone()
            input_ids_out[:] = self._pad

            placeholder = torch.full((prompt_embeds.shape[0], 1), -1, device=device, dtype=torch.long)
            info_update["speech_tokens"] = placeholder

            return input_ids_out, prompt_embeds, info_update

        # Decode: span_len == 1
        # Standard autoregressive decode - use input_ids directly
        text_token_len = int(info_dict.get("glm_tts_text_token_len", 10) or 10)
        self._record_pending_text_token_len(text_token_len)
        inputs_embeds_out = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long))
        inputs_embeds_out = inputs_embeds_out.reshape(1, -1).to(dtype=torch.bfloat16)

        # Convert sampled token to speech token (relative to ATS)
        # -1 = invalid/EOA, valid range = [0, ATE-ATS]
        sampled_token = int(input_ids[0].item())
        if self._ats <= sampled_token <= self._ate:
            speech_token = sampled_token - self._ats
            if not (0 <= speech_token <= 32767):
                raise ValueError(f"speech_token={speech_token} out of range [0, 32767]")
        else:
            # EOA or other non-audio token → mark as invalid (-1)
            speech_token = -1
            logger.debug("GLM-TTS decode: non-audio token %d (EOA=%d)", sampled_token, self._eoa)
        speech_tokens = torch.tensor([[speech_token]], device=device, dtype=torch.long)

        info_update = {"speech_tokens": speech_tokens}
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        """Cache last hidden state for next decode step."""
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach()
        return {"last_hidden": last}

    def _build_prompt_embeds(
        self,
        *,
        text: str,
        info_dict: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        """Build prompt embeddings for GLM-TTS.

        Text-only mode:     [Text | BOA]
        Voice clone mode:   [PromptText | Text | BOA | PromptSpeechTokens + ATS]

        Voice cloning follows the Fish Speech pattern: the serving layer passes
        ref_audio_wav + ref_audio_sr + ref_text, and this method extracts
        prompt_speech_token, prompt_feat, and embedding on the model side.
        """
        normalized_text = self._normalize_text_for_tokens(text)
        text_ids = self._encode_text(normalized_text, device)

        # Check for voice cloning inputs
        # Two sources: pre-extracted features (offline) or raw audio (online serving)
        prompt_text = None
        prompt_speech_token = None

        # Source 1: Pre-extracted features (offline inference)
        prompt_text_list = info_dict.get("prompt_text")
        prompt_speech_token_list = info_dict.get("prompt_speech_token")
        if (
            isinstance(prompt_text_list, list)
            and prompt_text_list
            and isinstance(prompt_speech_token_list, list)
            and prompt_speech_token_list
        ):
            prompt_text = prompt_text_list[0]
            prompt_speech_token = prompt_speech_token_list[0]

        # Source 2: Raw audio from serving layer (like Fish Speech)
        ref_audio_wav = info_dict.get("ref_audio_wav")
        ref_audio_sr = info_dict.get("ref_audio_sr")
        ref_text = info_dict.get("ref_text")
        if ref_audio_wav is not None and prompt_speech_token is None:
            if isinstance(ref_audio_wav, torch.Tensor):
                prompt_text = ref_text or ""
                prompt_speech_token = self._extract_prompt_speech_token(ref_audio_wav, int(ref_audio_sr or 24000))

        has_voice_clone = prompt_text is not None and prompt_speech_token is not None

        logger.info(
            "_build_prompt_embeds: has_voice_clone=%s, prompt_text=%s, prompt_speech_token_len=%s, normalized_text=%s",
            has_voice_clone,
            prompt_text[:50] if prompt_text else None,
            len(prompt_speech_token) if prompt_speech_token else None,
            normalized_text[:50] if normalized_text else None,
        )
        # Build sequence
        boa_tensor = torch.tensor([[self._boa]], device=device, dtype=torch.long)

        if has_voice_clone:
            # Voice cloning mode: [PromptText | Text | BOA | PromptSpeechTokens + ATS]
            normalized_prompt_text = self._normalize_text_for_tokens(
                prompt_text,
                add_trailing_space=True,
            )
            prompt_text_ids = self._encode_text(normalized_prompt_text, device)

            prompt_spk_ids = torch.tensor([prompt_speech_token], device=device, dtype=torch.long) + self._ats

            input_ids = torch.cat([prompt_text_ids, text_ids, boa_tensor, prompt_spk_ids], dim=1).to(torch.long)
        else:
            # Text-only mode: [Text | BOA]
            input_ids = torch.cat([text_ids, boa_tensor], dim=1).to(torch.long)

        # Embed
        prompt_embeds = self.embed_input_ids(input_ids.squeeze(0))
        return prompt_embeds.to(dtype=torch.bfloat16), int(text_ids.shape[1])

    def _record_pending_text_token_len(self, text_token_len: int) -> None:
        # GPUARModelRunner calls preprocess once per request in batch order
        # immediately before sample(), so this keeps length constraints aligned
        # with logits rows without a model-wide scalar.
        pending = getattr(self, "_pending_text_token_lens", None)
        if not isinstance(pending, list):
            pending = []
        pending.append(max(1, int(text_token_len)))
        self._pending_text_token_lens = pending

    def _consume_pending_text_token_lens(self, num_reqs: int) -> list[int]:
        pending = getattr(self, "_pending_text_token_lens", None)
        if not isinstance(pending, list) or not pending:
            return [10] * num_reqs
        text_token_lens = [max(1, int(v)) for v in pending[-num_reqs:]]
        if len(text_token_lens) < num_reqs:
            text_token_lens.extend([text_token_lens[-1]] * (num_reqs - len(text_token_lens)))
        self._pending_text_token_lens = []
        return text_token_lens

    def _ensure_voice_clone_frontend(self) -> None:
        """Lazy-load voice cloning frontend models."""
        if self._speech_tokenizer is not None:
            return
        device = next(self.model.parameters()).device
        self._speech_tokenizer, self._campplus_session = load_voice_clone_frontend(
            self._model_root,
            device,
            speech_tokenizer_cache=self._speech_tokenizer,
            campplus_cache=self._campplus_session,
        )
        self._voice_clone_frontend_loaded = True

    def _extract_prompt_speech_token(self, ref_audio_wav: torch.Tensor, ref_audio_sr: int) -> list[int] | None:
        self._ensure_voice_clone_frontend()
        return extract_prompt_speech_token(ref_audio_wav, ref_audio_sr, self._speech_tokenizer)

    def _extract_spk_embedding(self, ref_audio_wav: torch.Tensor, ref_audio_sr: int) -> list[float] | None:
        self._ensure_voice_clone_frontend()
        return extract_spk_embedding(ref_audio_wav, ref_audio_sr, self._campplus_session)

    def _extract_prompt_feat(self, ref_audio_wav: torch.Tensor, ref_audio_sr: int) -> torch.Tensor | None:
        self._ensure_voice_clone_frontend()
        if not self._voice_clone_frontend_loaded:
            return None
        device = next(self.model.parameters()).device
        return extract_prompt_feat(ref_audio_wav, ref_audio_sr, device)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.

        Stage 0 (glm_tts): HuggingFace Llama-format checkpoint.
        Stage 1 (glm_tts_dit): DiT flow.pt + vocoder.
        """
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.load_weights(weights)
        # vLLM uses merged projections; HF Llama has separate weights
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip rotary embedding buffers
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Map GLM-TTS weight names to vLLM Llama format
            if name.startswith("llama."):
                name = name[len("llama.") :]
            elif name == "llama_embedding.weight":
                name = "model.embed_tokens.weight"

            # Handle stacked/merged parameters (qkv_proj, gate_up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip if parameter doesn't exist (e.g., bias for some models)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Non-stacked parameters: load directly
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        # Handle tie_word_embeddings: copy embed_tokens to lm_head if not loaded
        lm_head_key = "lm_head.weight"
        embed_key = "model.embed_tokens.weight"
        if lm_head_key not in loaded_params and embed_key in loaded_params:
            if lm_head_key in params_dict and embed_key in params_dict:
                lm_head_param = params_dict[lm_head_key]
                embed_param = params_dict[embed_key]
                weight_loader = getattr(lm_head_param, "weight_loader", default_weight_loader)
                weight_loader(lm_head_param, embed_param.data)
                loaded_params.add(lm_head_key)
                logger.info("Tied lm_head.weight to embed_tokens.weight")

        logger.info("Loaded %d weights for GLMTTSForConditionalGeneration", len(loaded_params))
        return loaded_params
