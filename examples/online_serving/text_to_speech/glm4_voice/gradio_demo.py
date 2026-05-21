# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4-Voice Gradio demo with streaming support.

Connects to a running vLLM-Omni server and provides a web UI for TTS.

Usage:
    # Start server first (see run_server.sh), then:
    python gradio_demo.py --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import tempfile

import gradio as gr
from openai import OpenAI


def create_demo(server_url: str) -> gr.Blocks:
    client = OpenAI(api_key="EMPTY", base_url=f"{server_url}/v1")

    def synthesize(text: str, response_format: str = "wav") -> str | None:
        if not text.strip():
            return None
        response = client.audio.speech.create(
            model="THUDM/glm-4-voice-9b",
            voice="default",
            input=text,
            response_format=response_format,
        )
        with tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False) as f:
            f.write(response.content)
            return f.name

    with gr.Blocks(title="GLM-4-Voice TTS Demo") as demo:
        gr.Markdown("# GLM-4-Voice Text-to-Speech")
        with gr.Row():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to synthesize...",
                lines=3,
                value="今天天气真不错，适合出去散散步。",
            )
        with gr.Row():
            format_choice = gr.Dropdown(
                choices=["wav", "mp3", "flac"],
                value="wav",
                label="Output Format",
            )
            submit_btn = gr.Button("Synthesize", variant="primary")
        audio_output = gr.Audio(label="Generated Audio", type="filepath")

        submit_btn.click(
            fn=synthesize,
            inputs=[text_input, format_choice],
            outputs=audio_output,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = create_demo(args.server_url)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
