# GLM-TTS Online Serving Examples

Scripts for running GLM-TTS via the OpenAI-compatible `/v1/audio/speech` endpoint.

## Start the Server

### Non-streaming (default)

```bash
# Edit MODEL path in run_server.sh, then:
bash run_server.sh /path/to/GLM-TTS
```

## Non-Streaming Client

```bash
# Basic text-to-speech
python openai_speech_client.py --text "你好，这是一个语音合成测试。"

# English text
python openai_speech_client.py --text "Hello, this is a text-to-speech test."

# Custom max tokens and output format
python openai_speech_client.py --text "你好" --max-new-tokens 2048 --response-format mp3 -o output.mp3

# Voice cloning with reference audio
python openai_speech_client.py \
    --text "你好，这是语音克隆测试。" \
    --ref-audio /path/to/ref.wav \
    --ref-text "这是参考音频的文本内容。"
```

## API Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | string | Text to synthesize (required) |
| `model` | string | Model identifier |
| `max_new_tokens` | int | Maximum speech tokens to generate |
| `response_format` | string | Audio format: wav, mp3, flac, pcm, aac, opus |
| `stream` | bool | Enable streaming PCM output |
| `ref_audio` | string | Reference audio URL/base64 for voice cloning |
| `ref_text` | string | Transcript of reference audio (required with ref_audio) |

## Notes

- GLM-TTS outputs audio at **24kHz** sample rate.
- Voice cloning requires `ref_audio` + `ref_text` together. The reference audio should be 3-10 seconds.
- Voice cloning feature extraction (speech tokenizer, campplus, mel) happens on the model side — no `cosyvoice` dependency on the serving layer.
- Pass the GLM-TTS model root (for example `/path/to/GLM-TTS`) to the server. The pipeline config resolves `llm/`, `flow/`, and tokenizer subdirectories internally.
