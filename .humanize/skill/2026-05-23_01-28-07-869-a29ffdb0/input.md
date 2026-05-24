# Ask Codex Input

## Question

You are analyzing a draft plan for the vLLM-Omni project (IndexTTS2 TTS model integration). The project extends vLLM for omni-modality model serving.

## Repository Context
- Project: vLLM-Omni (omni-modality model serving framework)
- Branch: feat/index-tts-support
- Key paths:
  - cache/index-tts/indextts/ — Official IndexTTS2 reference implementation
  - vllm_omni/model_executor/models/indextts2/ — vLLM-Omni integration
  - examples/offline_inference/text_to_speech/indextts2/ — Offline test examples
  - examples/online_serving/text_to_speech/indextts2/ — Online serving examples
  - tests/e2e/ — E2E tests (offline + online)
  - vllm_omni/benchmarks/data_modules/seed_tts_eval.py — ASR evaluation framework
  - benchmarks/tts/bench_tts.py — TTS benchmark CLI

## Draft Content
The plan aims to:
1. Align vLLM-Omni IndexTTS2 implementation with official reference (cache/index-tts)
2. Commit and push changes
3. Run Colab tests (authuser=3, L4 GPU) for ASR validation
4. Test both offline (Omni.generate()) and online (/v1/audio/speech API) modes
5. Generate outputs to /content/index-tts-outputs/{offline,online}/
6. Include intermediate outputs and staged progress logging

## Key Findings from Exploration
- 15 component-level comparisons between official and vLLM implementations show high alignment
- Recent git commits (20+) show active convergence work
- Existing Seed-TTS eval framework with Whisper (EN) + Paraformer (ZH) + WER/SIM/UTMOS metrics
- Pipeline: Stage 0 (GPT AR talker) → Stage 1 (S2Mel + BigVGAN), async_chunk=false
- Known gaps: Qwen emotion fallback, CUDA kernel availability, semantic codec strict=False
- Colab MCP tools available for GPU/auth control

Provide your analysis in this exact format:
CORE_RISKS: highest-risk assumptions and potential failure modes
MISSING_REQUIREMENTS: likely omitted requirements or edge cases
TECHNICAL_GAPS: feasibility or architecture gaps
ALTERNATIVE_DIRECTIONS: viable alternatives with tradeoffs
QUESTIONS_FOR_USER: questions that need explicit human decisions
CANDIDATE_CRITERIA: candidate acceptance criteria suggestions

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-05-23_01-28-07
- Tool: codex
