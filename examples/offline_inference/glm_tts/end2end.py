# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS End-to-End Inference Example.

This example demonstrates how to use GLM-TTS for text-to-speech synthesis.

GLM-TTS is a two-stage TTS system:
  - Stage 0 (AR): Llama-based model generates speech tokens from text
  - Stage 1 (DiT): Flow matching model converts speech tokens to audio

Usage:
    python examples/offline_inference/glm_tts/end2end.py \
        --model /path/to/GLM-TTS \
        --text "Hello, this is a test of the GLM-TTS system." \
        --output-dir ./output

"""

import logging
import os
import time
from pathlib import Path

import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni

logger = logging.getLogger(__name__)

DEFAULT_DEPLOY_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "vllm_omni",
    "deploy",
    "glm_tts.yaml",
)

SAMPLE_RATE = 24000  # Vocos vocoder uses 24kHz


def build_prompt(
    text: str,
    model_path: str,
) -> dict:
    """Build a prompt for GLM-TTS.

    Args:
        text: Text to synthesize

    Returns:
        Dictionary with prompt_token_ids and additional_information
    """
    additional_info: dict = {
        "text": text,
    }

    try:
        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.glm_tts.glm_tts import (
            GLMTTSForConditionalGeneration,
        )
        from vllm_omni.model_executor.models.glm_tts.text_frontend import (
            GLMTTSTextFrontend,
        )

        path = Path(model_path)
        tokenizer_path: str | Path = model_path
        if path.exists():
            candidates = [
                path / "vq32k-phoneme-tokenizer",
                path.parent / "vq32k-phoneme-tokenizer" if path.name == "llm" else None,
            ]
            for candidate in candidates:
                if candidate is not None and candidate.is_dir():
                    tokenizer_path = candidate
                    break
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        prompt_len = GLMTTSForConditionalGeneration.estimate_prompt_len_from_text(
            text=text,
            tokenizer=tokenizer,
            text_frontend=GLMTTSTextFrontend(),
        )
    except Exception as exc:
        logger.warning("Failed to estimate GLM-TTS prompt length, using fallback 2048: %s", exc)
        prompt_len = 2048

    # GLM-TTS AR model builds the actual token sequence in preprocess()
    # Placeholder values are ignored, but length must match preprocess embeddings.
    return {
        "prompt_token_ids": [1] * prompt_len,
        "additional_information": additional_info,
    }


def _save_wav(output_dir: str, request_id: str, mm: dict) -> None:
    """Concatenate audio chunks and write to a wav file."""
    audio_data = mm.get("audio")
    if audio_data is None:
        logger.warning("Request %s: no audio output", request_id)
        return

    sr_raw = mm.get("sr", SAMPLE_RATE)
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

    if isinstance(audio_data, list):
        import numpy as np

        # Check if list elements are tensors or numpy arrays
        if hasattr(audio_data[0], "cpu"):
            audio_np = torch.cat(audio_data, dim=-1).float().cpu().numpy().flatten()
        else:
            audio_np = np.concatenate([np.asarray(a).flatten() for a in audio_data])
    elif hasattr(audio_data, "cpu"):
        audio_np = audio_data.float().cpu().numpy().flatten()
    else:
        import numpy as np

        audio_np = np.asarray(audio_data).flatten().astype(np.float32)
    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio_np, samplerate=sr, format="WAV")
    logger.info("Request %s: saved audio to %s (sr=%d)", request_id, out_wav, sr)


def main(args):
    """Run offline inference with Omni (synchronous)."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    deploy_config = args.deploy_config or DEFAULT_DEPLOY_CONFIG
    model_path = args.model

    inputs = [
        build_prompt(
            text=args.text,
            model_path=model_path,
        )
    ]

    omni = Omni(
        model=model_path,
        stage_configs_path=deploy_config,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_start = time.perf_counter()
    for stage_outputs in omni.generate(inputs):
        # For diffusion outputs, multimodal_output is on the OmniRequestOutput
        # itself (not on request_output.outputs[] which is empty for diffusion)
        mm = stage_outputs.multimodal_output
        if not mm:
            continue
        _save_wav(
            output_dir,
            stage_outputs.request_id,
            mm,
        )
    t_end = time.perf_counter()
    logger.info("Total inference time: %.1f ms", (t_end - t_start) * 1000)


def parse_args():
    parser = FlexibleArgumentParser(description="GLM-TTS Text-to-Speech Example")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model root path (e.g., /path/to/GLM-TTS)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你好，这是一个语音合成测试。",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for audio files",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Path to deploy config YAML (uses default if not specified)",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable stats logging",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Stage initialization timeout in seconds",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    main(args)
