# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4-Voice online TTS client using OpenAI-compatible API.

Usage:
    # Start server first (see run_server.sh), then:
    python openai_speech_client.py --text "Hello world"
"""

from __future__ import annotations

import argparse

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-4-Voice speech client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--text",
        default="今天天气真不错，适合出去散散步。",
    )
    parser.add_argument("--output", default="glm4_voice_output.wav")
    parser.add_argument("--response-format", default="wav")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.host}:{args.port}/v1",
    )

    if args.stream:
        with client.audio.speech.with_streaming_response.create(
            model="THUDM/glm-4-voice-9b",
            voice="default",
            input=args.text,
            response_format="pcm",
        ) as response:
            with open(args.output, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
            print(f"Streamed audio saved to {args.output}")
    else:
        response = client.audio.speech.create(
            model="THUDM/glm-4-voice-9b",
            voice="default",
            input=args.text,
            response_format=args.response_format,
        )
        response.write_to_file(args.output)
        print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
