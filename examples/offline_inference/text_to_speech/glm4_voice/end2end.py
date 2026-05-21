# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4-Voice-9B end-to-end offline TTS example.

Usage:
    python end2end.py [--model THUDM/glm-4-voice-9b] [--output output.wav]
"""

from __future__ import annotations

import argparse
import os

import soundfile as sf

# Resolve the deploy config relative to the vllm_omni package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_STAGE_CONFIG = os.path.join(_THIS_DIR, "..", "..", "..", "..", "vllm_omni", "deploy", "glm4_voice.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-4-Voice offline TTS")
    parser.add_argument(
        "--model",
        default="THUDM/glm-4-voice-9b",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--stage-config",
        default=_DEFAULT_STAGE_CONFIG,
        help="Path to stage config YAML",
    )
    parser.add_argument("--output", default="glm4_voice_output.wav")
    parser.add_argument(
        "--text",
        default="今天天气真不错，适合出去散散步。",
        help="Text to synthesize",
    )
    args = parser.parse_args()

    from vllm_omni import Omni
    from vllm_omni.model_executor.models.glm4_voice.glm4_voice import (
        build_glm4_voice_prompt,
    )

    omni = Omni(
        model=args.model,
        stage_configs_path=os.path.abspath(args.stage_config),
        trust_remote_code=True,
    )

    prompt_text = build_glm4_voice_prompt(args.text)
    inputs = {"prompt": prompt_text}

    outputs = omni.generate([inputs])

    if not outputs:
        print("No outputs returned.")
        return

    audio = outputs[0].multimodal_output.get("audio")
    sample_rate = outputs[0].multimodal_output.get("sample_rate", 22050)

    if audio is None:
        print("No audio in output.")
        return

    import torch

    if isinstance(audio, torch.Tensor):
        audio = audio.float().cpu().numpy()
    if isinstance(audio, list):
        import numpy as np

        audio = np.concatenate([a.float().cpu().numpy() if isinstance(a, torch.Tensor) else a for a in audio])

    sf.write(args.output, audio, samplerate=sample_rate)
    print(f"Audio saved to {args.output} ({len(audio)} samples, {sample_rate} Hz)")


if __name__ == "__main__":
    main()
