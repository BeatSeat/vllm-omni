# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Re-export: WhisperVQEncoder moved to common/whisper_vq.py.
from vllm_omni.model_executor.models.common.whisper_vq import (  # noqa: F401
    QuantizedBaseModelOutput,
    WhisperVQEncoder,
    remap_legacy_whisper_vq_state_dict,
    vector_quantize,
)
