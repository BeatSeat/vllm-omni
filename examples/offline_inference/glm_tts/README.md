# GLM-TTS Offline Inference Examples

End-to-end text-to-speech synthesis using GLM-TTS (AR + DiT two-stage pipeline).

## Quick Start

```bash
# Basic text-to-speech
python examples/offline_inference/glm_tts/end2end.py \
    --model /path/to/GLM-TTS \
    --text "你好，这是一个语音合成测试。" \
    --output-dir ./output
```

```bash
# Voice cloning
python examples/offline_inference/glm_tts/end2end.py \
    --model /path/to/GLM-TTS \
    --text "你好，这是语音克隆测试。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "这是参考音频的文本内容。" \
    --output-dir ./output
```

## Architecture

GLM-TTS is a two-stage TTS system:

```
Text → [Stage 0: AR Model] → Speech Tokens → [Stage 1: DiT + Vocoder] → Audio
         (Llama-based)          (32k vocab)      (Flow Matching + HiFT)  (24kHz WAV)
```

- **Stage 0 (AR)**: Llama-based autoregressive model generates speech tokens from text at 25 Hz
- **Stage 1 (DiT)**: Flow matching transformer converts speech tokens to mel-spectrograms, then HiFT vocoder synthesizes 24kHz audio (Vocos2D 32kHz fallback)

## Model Path

The `--model` path should point to the **repository root** (not `llm/` subdirectory):

```
GLM-TTS/
├── llm/                         # AR model weights
├── flow/                        # DiT model weights
├── hift/                        # HiFT vocoder (primary, 24kHz)
├── vocos2d/                     # Vocos2D vocoder (fallback, JIT, 32kHz→24kHz)
├── vq32k-phoneme-tokenizer/     # Tokenizer
├── speech_tokenizer/            # Speech tokenizer (for voice cloning)
└── frontend/campplus.onnx       # Speaker embedding (for voice cloning, from GitHub repo)
```

## Examples

| Script | Description |
|--------|-------------|
| `end2end.py` | Text-to-speech synthesis |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | Path to GLM-TTS model root |
| `--text` | "你好，这是一个语音合成测试。" | Text to synthesize |
| `--output-dir` | `./output` | Output directory for WAV files |
| `--ref-audio` | `None` | Reference WAV path or URL for voice cloning |
| `--ref-text` | `None` | Transcript of `--ref-audio` |
| `--deploy-config` | auto-detected | Path to deploy config YAML |
| `--stage-init-timeout` | 600 | Stage init timeout (seconds) |

## Output

Audio files are saved as `output_<request_id>.wav` in the output directory at 24kHz sample rate.

## Notes

- First run may take longer due to model loading and JIT compilation
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` (auto-set by scripts)
- Default sampling: temperature=1.0, top_k=25, top_p=0.8 (RAS method)
