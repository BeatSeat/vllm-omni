# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4-Voice-9B: ChatGLM4-based AR model for speech dialogue.

Stage 0 (AR): ChatGLM4 9B backbone generating interleaved text + audio
tokens.  Audio tokens are identified by ``token_id >= audio_offset``
where ``audio_offset = tokenizer('<|audio_0|>').input_ids``.

Stage 1 (Decoder): Delegated to ``GLM4VoiceDecoderForGeneration`` in
``glm4_voice_decoder.py`` (CosyVoice flow matching + HiFi-T vocoder).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.chatglm import ChatGLMModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.glm4_voice.glm4_voice_decoder import (
    GLM4VoiceDecoderForGeneration,
)
from vllm_omni.sequence import OmniOutput

logger = logging.getLogger(__name__)

# Default system prompt instructing interleaved text+audio generation.
_SYSTEM_PROMPT = (
    "User will provide you with a text instruction. "
    "Do it step by step. First, think about the instruction and respond "
    "in a interleaved manner, with 13 text token followed by 26 audio tokens."
)

# Default HF repo for the 9B model and decoder.
_GLM4_VOICE_MODEL_REPO = "THUDM/glm-4-voice-9b"
_GLM4_VOICE_DECODER_REPO = "THUDM/glm-4-voice-decoder"


def _resolve_special_token_ids(
    tokenizer: Any,
) -> tuple[int, int]:
    """Resolve audio_offset and end_token_id from the tokenizer."""
    audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    end_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
    if audio_offset is None or end_token_id is None:
        raise ValueError(
            "Failed to resolve GLM-4-Voice special tokens from tokenizer. "
            "Ensure THUDM/glm-4-voice-9b tokenizer is used."
        )
    return int(audio_offset), int(end_token_id)


def build_glm4_voice_prompt(text: str, system_prompt: str | None = None) -> str:
    """Build the chat-template prompt for GLM-4-Voice TTS inference."""
    sys = system_prompt or _SYSTEM_PROMPT
    return f"<|system|>\n{sys}<|user|>\n{text}<|assistant|>streaming_transcription\n"


class GLM4VoiceForConditionalGeneration(nn.Module):
    """Two-stage model: AR (ChatGLM4 9B) + Decoder (CosyVoice flow).

    The same class handles both stages via ``model_stage`` branching:
    - ``"glm4_voice_ar"``: ChatGLM4 backbone for autoregressive generation.
    - ``"glm4_voice_decoder"``: Delegates to ``GLM4VoiceDecoderForGeneration``.
    """

    supports_multimodal = False
    supports_multimodal_raw_input_only = False
    requires_raw_input_tokens = False
    prefer_model_sampler = False
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model_stage: str = getattr(vllm_config.model_config, "model_stage", "glm4_voice_ar")

        if self.model_stage == "glm4_voice_decoder":
            self.decoder = GLM4VoiceDecoderForGeneration(vllm_config=vllm_config, prefix=prefix)
            return

        # --- AR stage init ---
        # ChatGLM4 backbone.  The upstream vLLM ChatGLMModel expects
        # ``config.num_layers`` (mapped via ``attribute_map``).  Ensure
        # the custom GLM4VoiceConfig exposes required fields.
        self._patch_config_for_chatglm()
        self.model = ChatGLMModel(vllm_config=vllm_config, prefix=f"{prefix}transformer")
        self.logits_processor = LogitsProcessor(self.config.vocab_size, logit_scale=1.0)
        self.sampler = get_sampler()

        # Special token IDs — resolved lazily from tokenizer in
        # ``_ensure_token_ids`` to avoid requiring tokenizer at init.
        self._audio_offset: int | None = None
        self._end_token_id: int | None = None
        self._tokenizer_resolved = False

        # Per-request state: accumulates audio tokens across decode steps.
        self._speech_tokens: dict[str, list[int]] = {}

    def _patch_config_for_chatglm(self) -> None:
        """Ensure config has fields ChatGLMModel expects."""
        cfg = self.config
        if not hasattr(cfg, "num_layers"):
            cfg.num_layers = getattr(cfg, "num_hidden_layers", 40)
        if not hasattr(cfg, "padded_vocab_size"):
            cfg.padded_vocab_size = getattr(cfg, "vocab_size", 168960)
        if not hasattr(cfg, "multi_query_attention"):
            cfg.multi_query_attention = True
        if not hasattr(cfg, "multi_query_group_num"):
            cfg.multi_query_group_num = getattr(cfg, "num_key_value_heads", 2)
        if not hasattr(cfg, "add_bias_linear"):
            cfg.add_bias_linear = False
        if not hasattr(cfg, "add_qkv_bias"):
            cfg.add_qkv_bias = True
        if not hasattr(cfg, "seq_length"):
            cfg.seq_length = getattr(cfg, "max_sequence_length", 8192)
        if not hasattr(cfg, "post_layer_norm"):
            cfg.post_layer_norm = True
        if not hasattr(cfg, "apply_residual_connection_post_layernorm"):
            cfg.apply_residual_connection_post_layernorm = False
        if not hasattr(cfg, "original_rope"):
            cfg.original_rope = True
        if not hasattr(cfg, "rope_ratio"):
            cfg.rope_ratio = 500

    def _ensure_token_ids(self) -> None:
        """Lazily resolve special token IDs from tokenizer."""
        if self._tokenizer_resolved:
            return
        try:
            tokenizer = AutoTokenizer.from_pretrained(_GLM4_VOICE_MODEL_REPO, trust_remote_code=True)
            self._audio_offset, self._end_token_id = _resolve_special_token_ids(tokenizer)
            logger.info(
                "GLM-4-Voice token IDs: audio_offset=%d, end_token=%d",
                self._audio_offset,
                self._end_token_id,
            )
        except Exception:
            logger.warning("Could not resolve GLM-4-Voice tokenizer; using defaults.")
            self._audio_offset = 151552
            self._end_token_id = 151336
        self._tokenizer_resolved = True

    @property
    def audio_offset(self) -> int:
        self._ensure_token_ids()
        assert self._audio_offset is not None
        return self._audio_offset

    @property
    def end_token_id(self) -> int:
        self._ensure_token_ids()
        assert self._end_token_id is not None
        return self._end_token_id

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if self.model_stage == "glm4_voice_decoder":
            return self.decoder.forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        if self.model_stage == "glm4_voice_decoder":
            return self.decoder.compute_logits(hidden_states, sampling_metadata)
        logits = self.logits_processor(self.model.output_layer, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> list | None:
        if self.model_stage == "glm4_voice_decoder":
            return self.decoder.sample(logits, sampling_metadata)
        return self.sampler(logits, sampling_metadata)

    # ------------------------------------------------------------------
    # Omni hooks: preprocess / postprocess / make_omni_output
    # ------------------------------------------------------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Track generated audio tokens during decode steps."""
        if self.model_stage == "glm4_voice_decoder":
            return self.decoder.preprocess(input_ids, positions, info, **kwargs)

        is_prefill = info.get("is_prefill", False)
        req_key = str(info.get("_omni_req_id", "0"))

        if is_prefill:
            self._speech_tokens[req_key] = []
        else:
            sampled = info.get("sampled_token_id")
            if sampled is not None:
                token_id = int(sampled)
                if token_id >= self.audio_offset:
                    audio_token = token_id - self.audio_offset
                    self._speech_tokens.setdefault(req_key, []).append(audio_token)

        return input_ids, positions, info

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Cache last hidden state for next decode step."""
        if self.model_stage == "glm4_voice_decoder":
            return self.decoder.postprocess(hidden_states, info, **kwargs)

        req_key = str(info.get("_omni_req_id", "0"))
        info["speech_tokens"] = list(self._speech_tokens.get(req_key, []))

        return hidden_states, info

    def make_omni_output(
        self,
        hidden_states: torch.Tensor,
        info: dict[str, Any],
        sampled_token_ids: torch.Tensor | None = None,
    ) -> OmniOutput:
        """Package AR output for the decoder stage."""
        if self.model_stage == "glm4_voice_decoder":
            return self.decoder.make_omni_output(hidden_states, info, sampled_token_ids)

        req_key = str(info.get("_omni_req_id", "0"))
        speech_tokens = list(self._speech_tokens.get(req_key, []))

        # Extract the last generated token and check if it's audio.
        last_speech_token = -1
        if sampled_token_ids is not None and sampled_token_ids.numel() > 0:
            last_token = int(sampled_token_ids[-1].item())
            if last_token >= self.audio_offset:
                last_speech_token = last_token - self.audio_offset

        multimodal_outputs: dict[str, Any] = {}
        if speech_tokens:
            multimodal_outputs["speech_tokens"] = speech_tokens
        multimodal_outputs["last_speech_token"] = last_speech_token

        return OmniOutput(
            hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def on_request_finish(self, req_id: str) -> None:
        """Clean up per-request state."""
        self._speech_tokens.pop(req_id, None)
        if self.model_stage == "glm4_voice_decoder" and hasattr(self.decoder, "on_request_finish"):
            self.decoder.on_request_finish(req_id)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Any) -> None:
        if self.model_stage == "glm4_voice_decoder":
            self.decoder.load_weights(weights)
            return

        # AR stage: load ChatGLM4 weights.
        # HF key pattern: transformer.embedding.word_embeddings.weight
        #                  transformer.encoder.layers.N.*.weight
        #                  transformer.output_layer.weight
        # vLLM ChatGLMModel expects keys without "word_embeddings" prefix
        # on the embedding.
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Strip "transformer." prefix if present (HF convention).
            if name.startswith("transformer."):
                name = name[len("transformer.") :]

            # Map "embedding.word_embeddings" -> "embedding"
            name = name.replace("embedding.word_embeddings", "embedding")

            # Prefix with "model." for our ChatGLMModel wrapper.
            full_name = f"model.{name}"

            if full_name not in params_dict:
                # Try without "model." prefix (output_layer etc.)
                if name in params_dict:
                    full_name = name
                else:
                    continue

            param = params_dict[full_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(full_name)

        logger.info(
            "GLM-4-Voice AR: loaded %d/%d parameters",
            len(loaded_params),
            len(params_dict),
        )
