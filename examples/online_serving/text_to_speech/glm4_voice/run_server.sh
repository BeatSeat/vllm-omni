#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Launch GLM-4-Voice-9B TTS server.
#
# Usage:
#   bash run_server.sh [--port 8000]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_YAML="${SCRIPT_DIR}/../../../../vllm_omni/deploy/glm4_voice.yaml"

python -m vllm_omni.entrypoints.openai.api_server \
    --model THUDM/glm-4-voice-9b \
    --stage-config-path "${DEPLOY_YAML}" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "${1:-8000}" \
    "$@"
