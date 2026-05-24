³É¹¦: ÒÑÖÕÖ¹ PID 37948 (ÊôÓÚ PID 43408 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 43408 (ÊôÓÚ PID 49028 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 49028 (ÊôÓÚ PID 65896 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 65896 (ÊôÓÚ PID 56364 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 65176 (ÊôÓÚ PID 51600 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 51600 (ÊôÓÚ PID 7644 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 7644 (ÊôÓÚ PID 52304 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 52304 (ÊôÓÚ PID 56364 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 44972 (ÊôÓÚ PID 32244 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 32244 (ÊôÓÚ PID 13652 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 13652 (ÊôÓÚ PID 22724 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 22724 (ÊôÓÚ PID 56364 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 54064 (ÊôÓÚ PID 50704 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 50704 (ÊôÓÚ PID 5556 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 5556 (ÊôÓÚ PID 44620 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 44620 (ÊôÓÚ PID 57072 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 57072 (ÊôÓÚ PID 58936 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 58936 (ÊôÓÚ PID 56364 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 16588 (ÊôÓÚ PID 5116 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 5116 (ÊôÓÚ PID 61404 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 61404 (ÊôÓÚ PID 65840 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 65840 (ÊôÓÚ PID 45136 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 45136 (ÊôÓÚ PID 42876 ×Ó½ø³Ì)µÄ½ø³Ì¡£
³É¹¦: ÒÑÖÕÖ¹ PID 42876 (ÊôÓÚ PID 56364 ×Ó½ø³Ì)µÄ½ø³Ì¡£
AGREE: points accepted as reasonable

- Overall goal is sound: aligning against `cache/index-tts/indextts/infer_v2.py`, validating offline and online paths, then committing only after validation is the right direction.
- Real reference audio is required. The current e2e tests still fall back to sine waves, which is useful only for smoke tests, not quality/parity.
- Colab L4-specific harness is reasonable because current IndexTTS2 e2e tests are marked for L40, while `vllm_omni/deploy/indextts2.yaml` says verified on 1x L40 48GB.
- ASR-based validation using existing Seed-TTS eval machinery is reasonable as a quality gate, especially Whisper for English and Paraformer for Chinese.
- Benchmark registry gap is real: `benchmarks/tts/model_configs.yaml` currently has Qwen3-TTS and VoxCPM2 entries but no `IndexTeam/IndexTTS-2`.
- Push-after-validation is correct.

DISAGREE: points considered unreasonable and why

- AC-1 is too weak and slightly misleading. Matching WER between official and vLLM outputs does not prove â€œsame behaviorâ€; TTS outputs can differ acoustically while preserving transcript WER. Need waveform/audio metadata, duration, ASR transcript, and sampling/config parity checks too.
- â€œIdentical inputsâ€ is underspecified. Official `infer_v2.py` has parameters like `emo_audio_prompt`, `emo_vector`, `use_emo_text`, `emo_alpha`, `interval_silence`, `max_mel_tokens`, `top_p`, `top_k`, `temperature`, and `repetition_penalty`; the plan must explicitly define which subset is in scope.
- L4 memory assumption is optimistic. `0.4 + 0.4` allocation on a single 24 GB GPU may still fail due to framework overhead, CUDA context, ASR models, and colocated server/client processes. Treat `0.3 + 0.3` or sequential model loading as a planned path, not an afterthought.
- AC-4 says WER for both modes, but does not define thresholds. Without language-specific thresholds and acceptable delta vs official baseline, this cannot converge.
- Online serving coverage is incomplete if it only checks valid WAV/PCM. The prior issue â€œNeed negative API testsâ€ is listed but not actually included in milestones/tasks.
- Commit/push as an acceptance criterion is procedural, not validation. Fine as a final step, but it should not be mixed with model correctness criteria.

REQUIRED_CHANGES: must-fix items before convergence

- Define exact parity matrix against official `infer_v2.py`: at minimum CN and EN text, same real `spk_audio_prompt/ref_audio`, same sampling params, same `max_mel_tokens`, same seed where supported, same emotion mode policy.
- Add explicit ASR pass/fail thresholds:
  - absolute WER/CER per language,
  - allowed vLLM-vs-official degradation,
  - minimum audio duration / non-silence / sample rate / mono checks.
- Include real reference audio in the Colab harness from `cache/index-tts-vllm/assets/` or `cache/index-tts-vllm/examples/`; do not allow sine fallback for validation mode.
- Add negative online API tests: missing `ref_audio`, malformed `ref_audio`, unsupported `response_format`, invalid `emo_vector` length/type, invalid `emo_alpha`, empty input.
- Add benchmark config entry for `IndexTeam/IndexTTS-2` in `benchmarks/tts/model_configs.yaml`.
- Include docs/examples check if this is meant to be PR-ready: offline/online README rows or IndexTTS2 usage section appear missing from current example docs.
- Separate ASR model memory from TTS memory on Colab. Prefer run TTS generation first, unload/kill server, then run ASR, or document why simultaneous ASR + TTS fits on L4.
- Manifest schema must be specified before implementation: include prompt id, language, mode, backend, ref audio path/hash, sampling params, output path, sr, channels, duration, ASR transcript, target text, WER/CER, timings, git SHA, GPU, and errors.

OPTIONAL_IMPROVEMENTS: non-blocking improvements

- Add a small concurrent online smoke test, even if IndexTTS2 is non-streaming, to catch request-state leakage.
- Include streamed `pcm` validation separately from non-streaming `wav`, since `async_chunk=false` makes streaming a fallback behavior rather than true incremental synthesis.
- Track RTF, first-token/first-audio latency, and peak GPU memory in the manifest.
- Compare duration ratio and transcript similarity to official baseline, not only WER.
- Validate emotion modes separately: no emotion override, `emo_audio`, `emo_vector`, and `emo_text` if QwenEmotion is available.
- Keep Colab scripts reusable as repo scripts rather than notebook-only cells.

UNRESOLVED: opposite opinions needing user decisions

- Whether official parity means â€œASR transcript parity onlyâ€ or stricter perceptual/speaker similarity parity. WER alone is easy but incomplete.
- Whether validation must include emotion control now, or only plain voice cloning for this convergence pass.
- Whether L4 support is required for normal repo tests or only for the custom Colab validation harness.
- Whether to push after passing Colab only, or require local unit/e2e/ruff-style checks too before commit.
