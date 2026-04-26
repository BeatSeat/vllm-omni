# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS pipeline topology (frozen).

Stage 0: glm_tts     — text → speech tokens (LLM autoregressive, Llama backbone).
Stage 1: glm_tts_dit — speech tokens → audio waveform (DiT flow-matching).
  * ``sync_process_input_func`` runs when ``deploy.async_chunk=false``:
    stage 1 builds full-sequence input from complete AR output.
  * ``async_chunk_process_next_stage_input_func`` runs when
    ``deploy.async_chunk=true``: stage 0 streams speech token chunks to
    stage 1 through the shared-memory connector.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.glm_tts"

GLM_TTS_PIPELINE = PipelineConfig(
    model_type="glm_tts",
    model_arch="GLMTTSForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="glm_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            model_subdir="llm",
            tokenizer_subdir="vq32k-phoneme-tokenizer",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.ar_to_dit_async_chunk"),
            sampling_constraints={
                # GLM-TTS uses <|user|> (59253) as end-of-audio token.
                "stop_token_ids": [59253],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="glm_tts_dit",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            model_subdir="flow",
            tokenizer_subdir="llm",
            sync_process_input_func=f"{_PROC}.ar_to_dit",
        ),
    ),
)
