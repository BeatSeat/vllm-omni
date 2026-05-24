# Ask Codex Input

## Question

You are reviewing a CANDIDATE PLAN for the vLLM-Omni IndexTTS2 project. Review it for reasonability and completeness.

## Candidate Plan v1

### Goal
Align vLLM-Omni IndexTTS2 integration with official reference implementation (cache/index-tts/indextts/), validate via ASR on Colab L4, commit/push verified code.

### Acceptance Criteria

AC-1: Code Alignment — vLLM pipeline output matches official infer_v2.py behavior for identical inputs (same text, ref audio, sampling params). Verified by comparing WER of outputs from both paths.
AC-2: Offline Inference — Omni.generate() produces valid 22050 Hz mono WAV for CN/EN prompts on Colab L4. Output saved to /content/index-tts-outputs/offline/.
AC-3: Online Serving — /v1/audio/speech returns valid WAV/PCM for same prompts. Output saved to /content/index-tts-outputs/online/.
AC-4: ASR Validation — WER computed for both modes using Whisper (EN) and Paraformer (ZH). Results reported in manifest JSON per mode.
AC-5: Output Organization — /content/index-tts-outputs/{offline,online}/ with WAV files, metadata JSON, and progress logs.
AC-6: Staged Progress Logging — Each stage prints timestamped progress. Log contains stage markers, timing info, intermediate status.
AC-7: Commit and Push — Clean commit after validation passes, pushed to feat/index-tts-support.

### Key Architecture Decisions
1. L4 Memory: Deploy YAML uses 0.4+0.4 gpu_memory_utilization on same GPU. L4 (24GB) gives ~19.2GB total. Model weights ~4-6GB. Should fit but is tight. Plan to reduce to 0.3+0.3 if needed.
2. Reference Audio: Use real audio from cache/index-tts-vllm/assets/ or examples/ instead of synthetic sine waves.
3. Official Comparison: Run cache/index-tts standalone infer.py first as baseline, then compare vLLM output.
4. ASR Framework: Use Whisper-large-v3 (EN) + funasr Paraformer-zh (ZH) from existing seed_tts_eval.py.
5. Hardware markers: Existing tests require L40. Need custom Colab test script that handles L4 constraints.

### Prior Codex v1 Issues (from first analysis)
- Need real reference audio, not sine waves
- Need negative API tests
- Need output manifest format
- Sample rate is confirmed 22050 Hz consistently
- Emotion model is optional (graceful fallback)
- Benchmark CLI needs IndexTTS2 config entry
- Push timing: should validate before push

### Milestones
M1: Code alignment diff and fixes
M2: Colab L4 test harness with offline/online modes + progress logging
M3: ASR validation and output organization
M4: Commit and push

### Task Breakdown
task1: Systematic diff of official vs vLLM implementation (analyze)
task2: Fix identified alignment gaps (coding)
task3: Create Colab test script for offline mode with progress logging (coding)
task4: Create Colab test script for online mode with progress logging (coding)
task5: Integrate ASR validation (Whisper/Paraformer) into test scripts (coding)
task6: Set up output directory structure and manifest generation (coding)
task7: Run official baseline on Colab for comparison (coding)
task8: Run vLLM offline + online tests on Colab L4 (coding)
task9: Compare results and validate WER thresholds (analyze)
task10: Commit aligned code and push (coding)

Please review and respond in this exact format:
AGREE: points accepted as reasonable
DISAGREE: points considered unreasonable and why
REQUIRED_CHANGES: must-fix items before convergence
OPTIONAL_IMPROVEMENTS: non-blocking improvements
UNRESOLVED: opposite opinions needing user decisions

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-05-23_01-36-18
- Tool: codex
