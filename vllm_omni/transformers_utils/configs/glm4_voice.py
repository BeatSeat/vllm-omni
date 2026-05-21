# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4-Voice config registration with transformers AutoConfig.

Registers GLM4VoiceConfig (model_type="glm4_voice") so that
``AutoConfig.from_pretrained`` returns the correct config class
when ``hf_overrides={"model_type": "glm4_voice"}`` is set.

GLM-4-Voice-9B uses a ChatGLM4 backbone but we register a custom
config to handle the speech-specific parameters (codebook, frame rate,
mel config, interleave pattern).
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class GLM4VoiceConfig(PretrainedConfig):
    """ChatGLM4-based AR model for end-to-end speech dialogue.

    The 9B LLM generates interleaved text + audio tokens (13 text,
    26 audio).  Audio tokens are identified by ``token_id >= audio_offset``
    where ``audio_offset`` is resolved from the tokenizer at init time.

    Special token IDs (``audio_offset``, ``end_token_id``) are loaded
    dynamically from the tokenizer -- the config only stores
    speech-related hyperparameters.
    """

    model_type: str = "glm4_voice"

    def __init__(
        self,
        # ChatGLM4 arch params (mirrored from HF config.json)
        vocab_size: int = 168960,
        hidden_size: int = 4096,
        ffn_hidden_size: int = 13696,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        kv_channels: int = 128,
        max_sequence_length: int = 8192,
        rmsnorm: bool = True,
        # GLM-4-Voice speech params
        audio_codebook_size: int = 16384,
        input_frame_rate: float = 12.5,
        sample_rate: int = 22050,
        mel_dim: int = 80,
        mel_hop_size: int = 256,
        mel_n_fft: int = 1024,
        mel_f_min: float = 0.0,
        mel_f_max: float = 8000.0,
        # Interleave pattern: 13 text tokens followed by 26 audio tokens
        interleave_text_tokens: int = 13,
        interleave_audio_tokens: int = 26,
        # Sampling defaults
        default_temperature: float = 0.2,
        default_top_p: float = 0.8,
        # Token-to-text ratio for max_tokens estimation
        max_token_text_ratio: float = 40.0,
        min_token_text_ratio: float = 4.0,
        # Streaming chunk sizes
        streaming_chunk_sizes: list[int] | None = None,
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_channels = kv_channels
        self.max_sequence_length = max_sequence_length
        self.rmsnorm = rmsnorm
        self.audio_codebook_size = audio_codebook_size
        self.input_frame_rate = input_frame_rate
        self.sample_rate = sample_rate
        self.mel_dim = mel_dim
        self.mel_hop_size = mel_hop_size
        self.mel_n_fft = mel_n_fft
        self.mel_f_min = mel_f_min
        self.mel_f_max = mel_f_max
        self.interleave_text_tokens = interleave_text_tokens
        self.interleave_audio_tokens = interleave_audio_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.max_token_text_ratio = max_token_text_ratio
        self.min_token_text_ratio = min_token_text_ratio
        self.streaming_chunk_sizes = streaming_chunk_sizes or [25, 50, 100, 150, 200]
        super().__init__(**kwargs)


AutoConfig.register("glm4_voice", GLM4VoiceConfig)

__all__ = [
    "GLM4VoiceConfig",
]
