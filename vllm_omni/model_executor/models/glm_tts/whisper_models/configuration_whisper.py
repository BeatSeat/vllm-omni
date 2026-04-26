# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers import WhisperConfig


class WhisperVQConfig(WhisperConfig):
    def __init__(
        self,
        pooling_kernel_size=None,
        pooling_type="max",
        pooling_position=0,
        quantize_vocab_size=None,
        quantize_position=16,
        quantize_commit_coefficient=0.25,
        quantize_loss_scale=1.0,
        quantize_ema_decay=None,
        quantize_restart_interval=None,
        quantize_encoder_only=False,
        quantize_causal_encoder=False,
        quantize_causal_block_size=None,
        skip_language_detection=False,
        encoder_causal_attention=False,
        encoder_causal_convolution=False,
        pitch_control=None,
        pitch_loss_scale=1.0,
        pitch_upsample_scale=4,
        **kwargs,
    ):
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_commit_coefficient = quantize_commit_coefficient
        self.quantize_loss_scale = quantize_loss_scale
        self.quantize_ema_decay = quantize_ema_decay
        self.quantize_restart_interval = quantize_restart_interval
        self.quantize_encoder_only = quantize_encoder_only
        self.quantize_causal_encoder = quantize_causal_encoder
        self.quantize_causal_block_size = quantize_causal_block_size
        self.skip_language_detection = skip_language_detection
        self.encoder_causal_attention = encoder_causal_attention
        self.encoder_causal_convolution = encoder_causal_convolution
        self.pitch_control = pitch_control
        self.pitch_loss_scale = pitch_loss_scale
        self.pitch_upsample_scale = pitch_upsample_scale
        super().__init__(**kwargs)
