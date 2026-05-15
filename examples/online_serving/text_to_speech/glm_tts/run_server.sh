#!/bin/bash
# Launch vLLM-Omni server for GLM-TTS models
#
# Usage:
#   ./run_server.sh                                      # Default (async_chunk enabled)
#   ./run_server.sh /path/to/GLM-TTS                     # Custom model path
#   ./run_server.sh /path/to/GLM-TTS --no-async-chunk    # Sync two-stage mode
#
# NOTE: The model path should point to the repo ROOT (not llm/ subdirectory).
# model_subdir/tokenizer_subdir in the pipeline config resolve subdirectories.
# Extra arguments after the model path are passed through to the server
# (e.g. --no-async-chunk, --disable-log-stats).

set -e

MODEL="${1:-zai-org/GLM-TTS}"
shift 2>/dev/null || true
EXTRA_ARGS="$@"

echo "Starting GLM-TTS server with model: $MODEL"
[ -n "$EXTRA_ARGS" ] && echo "Extra args: $EXTRA_ARGS"

vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/glm_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --omni \
    $EXTRA_ARGS
