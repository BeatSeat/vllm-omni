# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS config registration with transformers AutoConfig.

Registers GLMTTSConfig (model_type="glm_tts") so that
``AutoConfig.from_pretrained("path/to/glm-tts")`` returns the correct config class.

Note: GLM-TTS uses a Llama backbone, but we register a custom config
to handle the special token IDs and flow model parameters.
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.glm_tts.configuration_glm_tts import (
    GLMTTSConfig,
)

AutoConfig.register("glm_tts", GLMTTSConfig)

__all__ = [
    "GLMTTSConfig",
]
