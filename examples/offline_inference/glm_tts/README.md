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

## Architecture

GLM-TTS is a two-stage TTS system:

```
Text → [Stage 0: AR Model] → Speech Tokens → [Stage 1: DiT + Vocoder] → Audio
         (Llama-based)          (32k vocab)      (Flow Matching + Vocos2D)  (24kHz WAV)
```

- **Stage 0 (AR)**: Llama-based autoregressive model generates speech tokens from text at 25 Hz
- **Stage 1 (DiT)**: Flow matching transformer converts speech tokens to mel-spectrograms, then Vocos2D vocoder synthesizes 24kHz audio

## Model Path

The `--model` path should point to the **repository root** (not `llm/` subdirectory):

```
GLM-TTS/
├── llm/                         # AR model weights
├── flow/                        # DiT model weights
├── vocos2d/                     # Vocos2D vocoder (JIT, 32kHz→24kHz)
├── hift/                        # HiFi-GAN vocoder (fallback)
├── vq32k-phoneme-tokenizer/     # Tokenizer
├── ckpt/speech_tokenizer/       # Speech tokenizer (for voice cloning)
└── frontend/campplus.onnx       # Speaker embedding (for voice cloning)
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
| `--deploy-config` | auto-detected | Path to deploy config YAML |
| `--stage-init-timeout` | 600 | Stage init timeout (seconds) |

## Output

Audio files are saved as `output_<request_id>.wav` in the output directory at 24kHz sample rate.

## Notes

- First run may take longer due to model loading and JIT compilation
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` (auto-set by scripts)
- Default sampling: temperature=0.9, top_k=25, repetition_penalty=1.05
