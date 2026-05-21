# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4-Voice pipeline: Stage 0 (ChatGLM4 AR) → Stage 1 (Flow Decoder)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.glm4_voice"

# <|user|> token ID in ChatGLM4Tokenizer acts as end-of-audio marker.
# Resolved dynamically at model init; this is the fallback constant
# from the THUDM/glm-4-voice-9b tokenizer.
_END_TOKEN_ID = 151336

GLM4_VOICE_PIPELINE = PipelineConfig(
    model_type="glm4_voice",
    model_arch="GLM4VoiceForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="glm4_voice_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.ar_to_decoder_async_chunk"),
            sampling_constraints={
                "stop_token_ids": [_END_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="glm4_voice_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.ar_to_decoder",
        ),
    ),
)
