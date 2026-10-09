# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


class RequestPreprocessingError(ValueError):
    """A model-declared request contract failure, not a device/worker failure."""
