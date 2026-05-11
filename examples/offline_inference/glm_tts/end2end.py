# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS End-to-End Inference Example.

This example demonstrates GLM-TTS zero-shot voice cloning.

GLM-TTS is a two-stage TTS system:
  - Stage 0 (AR): Llama-based model generates speech tokens from text
  - Stage 1 (DiT): Flow matching model converts speech tokens to audio

Usage:
    python examples/offline_inference/glm_tts/end2end.py \
        --model /path/to/GLM-TTS \
        --text "Hello, this is a test of the GLM-TTS system." \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio." \
        --output-dir ./output

"""

import base64
import io
import logging
import os
import time
from collections import defaultdict
from typing import Any
from urllib.request import urlopen

import soundfile as sf
import torch
import yaml

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni
from vllm_omni.model_executor.models.glm_tts.configuration_glm_tts import GLMTTSConfig
from vllm_omni.model_executor.models.glm_tts.glm_tts import (
    _normalize_glm_tts_processor_text,
    load_glm_tts_tokenizer,
    resolve_glm_tts_model_dir,
    resolve_glm_tts_tokenizer_path,
)
from vllm_omni.model_executor.models.glm_tts.text_frontend import GLMTTSTextFrontend

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
SAMPLES_PER_SPEECH_TOKEN = 960  # 24kHz / 25 fps


def build_prompt(
    text: str,
    ref_audio_wav: torch.Tensor | None = None,
    ref_audio_sr: int | None = None,
    ref_text: str | None = None,
) -> dict:
    """Build a prompt for GLM-TTS.

    Args:
        text: Text to synthesize

    Returns:
        Dictionary with raw text, reference audio, and GLM-TTS multimodal kwargs.
    """
    if ref_audio_wav is None or not ref_text:
        raise ValueError("GLM-TTS requires ref_audio and ref_text for zero-shot voice cloning.")
    return {
        "prompt": text,
        "multi_modal_data": {
            "audio": (ref_audio_wav.float().cpu().numpy(), int(ref_audio_sr or SAMPLE_RATE)),
        },
        "modalities": ["audio"],
        "mm_processor_kwargs": {
            "prompt_text": ref_text,
            "sample_rate": int(ref_audio_sr or SAMPLE_RATE),
        },
    }


def _load_ref_audio(ref_audio: str | None) -> tuple[torch.Tensor | None, int | None]:
    if not ref_audio:
        return None, None

    if ref_audio.startswith(("http://", "https://")):
        with urlopen(ref_audio, timeout=60) as response:
            audio_obj = io.BytesIO(response.read())
    elif ref_audio.startswith("data:"):
        _, _, encoded = ref_audio.partition(",")
        audio_obj = io.BytesIO(base64.b64decode(encoded))
    else:
        audio_obj = ref_audio

    wav_np, sr = sf.read(audio_obj, dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    return torch.from_numpy(wav_np), int(sr)


def _audio_to_tensor(mm: dict) -> tuple[torch.Tensor | None, int]:
    """Concatenate audio chunks and write to a wav file."""
    audio_data = mm.get("audio")
    if audio_data is None:
        return None, SAMPLE_RATE

    sr_raw = mm.get("sr", SAMPLE_RATE)
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

    if isinstance(audio_data, list):
        import numpy as np

        if not audio_data:
            return torch.zeros(0, dtype=torch.float32), sr
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
    return torch.as_tensor(audio_np, dtype=torch.float32), sr


def _save_wav(output_dir: str, request_id: str, audio: torch.Tensor, sr: int) -> None:
    """Write a wav file."""
    audio_np = audio.float().cpu().numpy().flatten()
    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio_np, samplerate=sr, format="WAV")
    logger.info("Request %s: saved audio to %s (sr=%d)", request_id, out_wav, sr)


def _first_scalar(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return _first_scalar(value[0])
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0].item())
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except Exception:
            return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _flatten_speech_tokens(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.to(torch.long).reshape(-1).cpu().tolist()
    if isinstance(value, list):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_speech_tokens(item))
        return flattened
    if isinstance(value, tuple):
        flattened = []
        for item in value:
            flattened.extend(_flatten_speech_tokens(item))
        return flattened
    try:
        return [int(value)]
    except Exception:
        return []


def _safe_load_deploy_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Deploy config at {path} is not a YAML mapping.")
    return data


def _resolve_glm_tts_token_counts(model_path: str, text: str, ref_text: str) -> dict[str, int | None]:
    try:
        tokenizer_path = resolve_glm_tts_tokenizer_path(model_path)
        tokenizer = load_glm_tts_tokenizer(
            tokenizer_path,
            model_name_or_path=model_path,
            trust_remote_code=True,
        )
        frontend = GLMTTSTextFrontend()
        normalized_prompt_text = _normalize_glm_tts_processor_text(
            frontend,
            ref_text,
            add_trailing_space=True,
        )
        normalized_text = _normalize_glm_tts_processor_text(frontend, text)
        return {
            "prompt_text_tokens": len(tokenizer.encode(normalized_prompt_text)),
            "target_text_tokens": len(tokenizer.encode(normalized_text)),
        }
    except Exception:
        logger.warning("Failed to resolve GLM-TTS tokenizer counts", exc_info=True)
        return {
            "prompt_text_tokens": None,
            "target_text_tokens": None,
        }


def _resolve_glm_tts_generation_bounds(model_path: str, target_text_tokens: int | None) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {
        "min_ratio": None,
        "max_ratio": None,
        "min_tokens": None,
        "max_tokens": None,
    }
    if target_text_tokens is None:
        return result
    try:
        model_root = resolve_glm_tts_model_dir(model_path)
        llm_dir = os.path.join(model_root, "llm") if os.path.isdir(os.path.join(model_root, "llm")) else model_root
        config = GLMTTSConfig.from_pretrained(llm_dir)
        min_ratio = float(getattr(config, "min_token_text_ratio", 2.0))
        max_ratio = float(getattr(config, "max_token_text_ratio", 20.0))
    except Exception:
        logger.warning("Failed to resolve GLM-TTS config ratios; falling back to defaults", exc_info=True)
        min_ratio = 2.0
        max_ratio = 20.0
    result["min_ratio"] = min_ratio
    result["max_ratio"] = max_ratio
    result["min_tokens"] = int(target_text_tokens * min_ratio)
    result["max_tokens"] = int(target_text_tokens * max_ratio)
    return result


def _log_input_diagnostics(
    *,
    args: Any,
    deploy_config: dict[str, Any],
    ref_audio_wav: torch.Tensor,
    ref_audio_sr: int,
    token_counts: dict[str, int | None],
    generation_bounds: dict[str, int | float | None],
) -> None:
    duration_sec = float(ref_audio_wav.numel()) / float(ref_audio_sr)
    stage_bits = []
    for idx, stage_cfg in enumerate(deploy_config.get("stages", []) or []):
        if not isinstance(stage_cfg, dict):
            continue
        stage_bits.append(
            {
                "index": idx,
                "name": stage_cfg.get("name"),
                "enforce_eager": stage_cfg.get("enforce_eager"),
                "async_scheduling": stage_cfg.get("async_scheduling"),
            }
        )
    logger.info(
        "GLM-TTS offline input: text=%r ref_text=%r ref_audio_sr=%d ref_audio_duration=%.3fs",
        args.text,
        args.ref_text,
        ref_audio_sr,
        duration_sec,
    )
    logger.info(
        "GLM-TTS offline config: deploy_config=%s async_chunk=%s stages=%s",
        args.deploy_config or DEFAULT_DEPLOY_CONFIG,
        deploy_config.get("async_chunk"),
        stage_bits,
    )
    logger.info(
        "GLM-TTS offline dynamic tokens: prompt_text_tokens=%s text_tokens=%s min_tokens=%s max_tokens=%s",
        token_counts.get("prompt_text_tokens"),
        token_counts.get("target_text_tokens"),
        generation_bounds.get("min_tokens"),
        generation_bounds.get("max_tokens"),
    )


def _log_ar_stage_summary(
    request_id: str,
    stage_id: int | None,
    mm: dict[str, Any],
    request_state: dict[str, Any],
) -> None:
    raw_tokens = _flatten_speech_tokens(mm.get("speech_tokens"))
    valid_tokens = [token for token in raw_tokens if token >= 0]
    invalid_tokens = len(raw_tokens) - len(valid_tokens)
    prev_valid_tokens = list(request_state.get("ar_valid_tokens", []))
    if prev_valid_tokens and len(valid_tokens) >= len(prev_valid_tokens) and valid_tokens[: len(prev_valid_tokens)] == prev_valid_tokens:
        merged_valid_tokens = valid_tokens
    else:
        merged_valid_tokens = prev_valid_tokens + valid_tokens
    request_state["ar_total_tokens"] = max(int(request_state.get("ar_total_tokens") or 0), len(raw_tokens))
    request_state["ar_valid_tokens"] = merged_valid_tokens
    request_state["ar_invalid_tokens"] = max(int(request_state.get("ar_invalid_tokens") or 0), invalid_tokens)

    text_token_len = _first_scalar(mm.get("glm_tts_text_token_len"))
    if text_token_len is not None:
        request_state["ar_text_token_len"] = text_token_len

    prompt_speech_token_len = _first_scalar(mm.get("prompt_speech_token_len"))
    if prompt_speech_token_len is not None:
        request_state["prompt_speech_token_len"] = prompt_speech_token_len

    logger.info(
        "GLM-TTS offline AR summary: request=%s stage=%s text_token_len=%s prompt_speech_tokens=%s raw_tokens=%d valid_tokens=%d invalid_tokens=%d head=%s tail=%s",
        request_id,
        stage_id,
        request_state.get("ar_text_token_len"),
        request_state.get("prompt_speech_token_len"),
        len(raw_tokens),
        len(merged_valid_tokens),
        invalid_tokens,
        merged_valid_tokens[:16],
        merged_valid_tokens[-16:] if merged_valid_tokens else [],
    )

    min_tokens = request_state.get("min_tokens")
    if not merged_valid_tokens:
        logger.warning("GLM-TTS offline warning: request=%s AR produced no valid speech tokens.", request_id)
    elif min_tokens is not None and len(merged_valid_tokens) < max(8, int(min_tokens // 2)):
        logger.warning(
            "GLM-TTS offline warning: request=%s AR speech token count looks short (%d < min_tokens=%s).",
            request_id,
            len(merged_valid_tokens),
            min_tokens,
        )


def _log_dit_stage_summary(
    request_id: str,
    stage_id: int | None,
    audio: torch.Tensor,
    sr: int,
    request_state: dict[str, Any],
) -> None:
    sample_count = int(audio.numel())
    duration_sec = float(sample_count) / float(sr)
    request_state["dit_audio_samples"] = int(request_state.get("dit_audio_samples") or 0) + sample_count
    request_state["dit_audio_sr"] = sr
    logger.info(
        "GLM-TTS offline DiT summary: request=%s stage=%s audio_samples=%d sr=%d duration=%.3fs",
        request_id,
        stage_id,
        sample_count,
        sr,
        duration_sec,
    )


def _log_final_request_summary(request_id: str, request_state: dict[str, Any]) -> None:
    valid_tokens = request_state.get("ar_valid_tokens", [])
    final_samples = int(request_state.get("dit_audio_samples") or 0)
    final_sr = int(request_state.get("dit_audio_sr") or SAMPLE_RATE)
    min_tokens = request_state.get("min_tokens")
    max_tokens = request_state.get("max_tokens")
    logger.info(
        "GLM-TTS offline final summary: request=%s prompt_text_tokens=%s target_text_tokens=%s prompt_speech_tokens=%s min_tokens=%s max_tokens=%s ar_valid_tokens=%d final_audio_samples=%d final_audio_duration=%.3fs",
        request_id,
        request_state.get("prompt_text_tokens"),
        request_state.get("target_text_tokens"),
        request_state.get("prompt_speech_token_len"),
        min_tokens,
        max_tokens,
        len(valid_tokens),
        final_samples,
        float(final_samples) / float(final_sr) if final_sr else 0.0,
    )
    if len(valid_tokens) == 0:
        logger.warning("GLM-TTS offline warning: request=%s has empty AR token output.", request_id)
        return

    if min_tokens is not None and len(valid_tokens) < min_tokens:
        logger.warning(
            "GLM-TTS offline warning: request=%s AR generated fewer tokens than configured minimum (%d < %d).",
            request_id,
            len(valid_tokens),
            min_tokens,
        )

    min_expected_samples = None
    if min_tokens is not None:
        min_expected_samples = int(min_tokens) * SAMPLES_PER_SPEECH_TOKEN
        if final_samples and final_samples < int(min_expected_samples * 0.75):
            logger.warning(
                "GLM-TTS offline warning: request=%s final wav looks too short for min_tokens (%d samples < ~%d expected).",
                request_id,
                final_samples,
                min_expected_samples,
            )

    if min_tokens is not None and len(valid_tokens) >= min_tokens and final_samples:
        if final_samples < int(len(valid_tokens) * SAMPLES_PER_SPEECH_TOKEN * 0.5):
            logger.warning(
                "GLM-TTS offline warning: request=%s AR token count looks healthy but final wav is much shorter; problem is more likely in DiT / bridge.",
                request_id,
            )


def main(args):
    """Run offline inference with Omni (synchronous)."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    deploy_config_path = args.deploy_config or DEFAULT_DEPLOY_CONFIG
    deploy_config = _safe_load_deploy_config(deploy_config_path)
    model_path = args.model
    ref_audio_wav, ref_audio_sr = _load_ref_audio(args.ref_audio)
    if ref_audio_wav is None or not args.ref_text:
        raise ValueError("GLM-TTS requires --ref-audio and --ref-text for zero-shot voice cloning.")
    token_counts = _resolve_glm_tts_token_counts(model_path, args.text, args.ref_text)
    generation_bounds = _resolve_glm_tts_generation_bounds(model_path, token_counts.get("target_text_tokens"))
    _log_input_diagnostics(
        args=args,
        deploy_config=deploy_config,
        ref_audio_wav=ref_audio_wav,
        ref_audio_sr=ref_audio_sr,
        token_counts=token_counts,
        generation_bounds=generation_bounds,
    )

    inputs = [
        build_prompt(
            text=args.text,
            ref_audio_wav=ref_audio_wav,
            ref_audio_sr=ref_audio_sr,
            ref_text=args.ref_text,
        )
    ]

    omni = Omni(
        model=model_path,
        stage_configs_path=deploy_config_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_start = time.perf_counter()
    audio_chunks_by_request: dict[str, list[torch.Tensor]] = {}
    prev_audio_count_by_request: dict[str, int] = defaultdict(int)
    sample_rate_by_request: dict[str, int] = {}
    request_diagnostics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            **token_counts,
            **generation_bounds,
            "prompt_speech_token_len": None,
            "ar_total_tokens": 0,
            "ar_valid_tokens": [],
            "ar_invalid_tokens": 0,
            "ar_text_token_len": None,
            "dit_audio_samples": 0,
            "dit_audio_sr": SAMPLE_RATE,
        }
    )
    for stage_outputs in omni.generate(inputs):
        request_state = request_diagnostics[stage_outputs.request_id]
        if stage_outputs.error:
            request_state.setdefault("errors", []).append(str(stage_outputs.error))
            logger.warning(
                "GLM-TTS offline request=%s stage=%s error=%s",
                stage_outputs.request_id,
                stage_outputs.stage_id,
                stage_outputs.error,
            )
        # For diffusion outputs, multimodal_output is on the OmniRequestOutput
        # itself (not on request_output.outputs[] which is empty for diffusion)
        mm = stage_outputs.multimodal_output
        if not mm:
            continue
        if "speech_tokens" in mm:
            _log_ar_stage_summary(stage_outputs.request_id, stage_outputs.stage_id, mm, request_state)
        mm_for_audio = mm
        audio_val = mm.get("audio")
        if isinstance(audio_val, list):
            prev_count = prev_audio_count_by_request[stage_outputs.request_id]
            new_audio = audio_val[prev_count:]
            prev_audio_count_by_request[stage_outputs.request_id] = len(audio_val)
            if not new_audio:
                continue
            mm_for_audio = dict(mm)
            mm_for_audio["audio"] = new_audio
        audio, sr = _audio_to_tensor(mm_for_audio)
        if audio is None:
            logger.info(
                "GLM-TTS offline stage output: request=%s stage=%s final_output_type=%s mm_keys=%s",
                stage_outputs.request_id,
                stage_outputs.stage_id,
                stage_outputs.final_output_type,
                sorted(mm.keys()),
            )
            continue
        if audio.numel() == 0:
            continue
        audio_chunks_by_request.setdefault(stage_outputs.request_id, []).append(audio)
        sample_rate_by_request[stage_outputs.request_id] = sr
        _log_dit_stage_summary(stage_outputs.request_id, stage_outputs.stage_id, audio, sr, request_state)

    for request_id, chunks in audio_chunks_by_request.items():
        _save_wav(output_dir, request_id, torch.cat(chunks, dim=-1), sample_rate_by_request[request_id])
        _log_final_request_summary(request_id, request_diagnostics[request_id])
    for request_id, request_state in request_diagnostics.items():
        if request_id not in audio_chunks_by_request:
            _log_final_request_summary(request_id, request_state)
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
        "--ref-audio",
        type=str,
        default=None,
        help="Reference WAV path or URL for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of --ref-audio for voice cloning",
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
