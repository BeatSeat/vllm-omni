# GLM-TTS for Chinese/English TTS on 1x GPU

## Summary

- Vendor: THUDM
- Model: `THUDM/GLM-TTS`
- Task: Text-to-speech synthesis with optional voice cloning
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## When to use this recipe

Use this recipe to serve GLM-TTS as a two-stage TTS system (AR + DiT
flow-matching) for Chinese and English speech synthesis. Supports voice cloning
via reference audio.

## References

- Upstream or canonical docs:
  [THUDM/GLM-TTS on HuggingFace](https://huggingface.co/THUDM/GLM-TTS)
- Related example under `examples/`:
  [`examples/online_serving/glm_tts/README.md`](../../examples/online_serving/glm_tts/README.md)
- Offline inference example:
  [`examples/offline_inference/glm_tts/README.md`](../../examples/offline_inference/glm_tts/README.md)

## Hardware Support

### GPU

### 1x A40 48GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with A40 48GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Start the server from the repository root:

```bash
vllm-omni serve THUDM/GLM-TTS --omni --port 8091
```

Async chunking is enabled by default in the bundled deployment config. For
the sync (non-streaming) path:

```bash
vllm-omni serve THUDM/GLM-TTS --omni --port 8091 --no-async-chunk
```

Use a custom deploy config for advanced cases:

```bash
vllm-omni serve THUDM/GLM-TTS --omni --port 8091 \
  --deploy-config /path/to/your_glm_tts_overrides.yaml
```

#### Verification

Run the bundled OpenAI-compatible client:

```bash
python examples/online_serving/glm_tts/openai_speech_client.py \
  --text "你好，这是一个语音合成测试。"
```

For a quick API smoke test:

```bash
curl http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "THUDM/GLM-TTS",
    "input": "你好，这是一个语音合成测试。",
    "response_format": "wav"
  }' --output test.wav
```

Voice cloning with reference audio:

```bash
python examples/online_serving/glm_tts/openai_speech_client.py \
  --text "你好，这是语音克隆测试。" \
  --ref-audio /path/to/ref.wav \
  --ref-text "这是参考音频的文本内容。"
```

#### Notes

- Memory usage: ~18-20GB total (AR ~10GB, DiT ~8GB); both stages share GPU 0 by default.
- Audio output: 24kHz mono WAV via Vocos2D vocoder.
- Key flags: `--omni` is required; `enforce_eager: true` is the default (CUDA graphs not yet verified for this model).
- Voice cloning: requires `ref_audio` + `ref_text` together. Reference audio should be 3-10 seconds. Feature extraction (WhisperVQ tokenizer, CampPlus ONNX, mel) runs on the model side.
- Known limitations: First request may be slow due to lazy model loading (WhisperVQ, CampPlus ONNX). Warm-cache RTF is approximately 0.6-0.7x on A40.
