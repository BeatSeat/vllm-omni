#!/bin/bash
# Launch vLLM-Omni server for GLM-TTS models
#
# Usage:
#   ./run_server.sh                           # Default model path
#   ./run_server.sh /path/to/GLM-TTS          # Custom model path
#
# NOTE: The model path should point to the repo ROOT (not llm/ subdirectory).
# model_subdir/tokenizer_subdir in the pipeline config resolve subdirectories.

set -e

MODEL="${1:-/path/to/GLM-TTS}"

echo "Starting GLM-TTS server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/glm_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --omni
