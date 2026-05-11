# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Re-export: the VQ-specific config class has been retired.  Models should
# now use ``transformers.WhisperConfig`` directly — the checkpoint's
# config.json carries all VQ fields and HF's ``PretrainedConfig`` stores
# them as attributes automatically.
#
# This alias keeps legacy ``from ...configuration_whisper import WhisperVQConfig``
# imports working.
from transformers import WhisperConfig as WhisperVQConfig  # noqa: F401
