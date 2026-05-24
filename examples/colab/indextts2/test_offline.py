#!/usr/bin/env python3
"""IndexTTS2 offline inference test for Colab L4 with ASR validation.

Runs Omni.generate() with real reference audio, validates via Whisper ASR,
and writes results to /content/index-tts-outputs/offline/.

Usage (on Colab after installing vllm-omni):
  python test_offline.py \
    --model IndexTeam/IndexTTS-2 \
    --ref-audio /path/to/voice.wav \
    --output-dir /content/index-tts-outputs/offline
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(tag: str, msg: str) -> None:
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def whisper_transcribe(audio_path: str, model_name: str, device: str) -> str:
    import whisper

    kwargs = {} if device == "auto" else {"device": device}
    model = whisper.load_model(model_name, **kwargs)
    result = model.transcribe(
        audio_path,
        language=None,
        temperature=0.0,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    return result.get("text", "").strip()


def preprocess_text(text: str) -> str:
    try:
        import opencc

        text = opencc.OpenCC("t2s").convert(text)
    except Exception:
        pass

    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    for word, digit in word_to_num.items():
        text = re.sub(rf"\b{word}\b", digit, text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.lower().strip()


def cosine_similarity_text(text_a: str, text_b: str, n: int = 3) -> float:
    """Same n-gram text similarity shape as tests.helpers.media."""
    text_a = preprocess_text(text_a)
    text_b = preprocess_text(text_b)
    if not text_a or not text_b:
        return 0.0

    def ngrams(text: str) -> list[str]:
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    ngrams_a = ngrams(text_a)
    ngrams_b = ngrams(text_b)
    if not ngrams_a or not ngrams_b:
        return 0.0

    counter_a = Counter(ngrams_a)
    counter_b = Counter(ngrams_b)
    intersection = set(counter_a.keys()) & set(counter_b.keys())
    dot_product = sum(counter_a[x] * counter_b[x] for x in intersection)

    norm_a = sum(counter_a[x] ** 2 for x in counter_a) ** 0.5
    norm_b = sum(counter_b[x] ** 2 for x in counter_b) ** 0.5
    cosine = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
    length_ratio = min(len(text_a), len(text_b)) / max(len(text_a), len(text_b))
    return cosine * length_ratio


def iter_request_outputs(stage_output):
    request_output = getattr(stage_output, "request_output", None)
    if request_output is None:
        return
    if isinstance(request_output, (list, tuple)):
        yield from request_output
    else:
        yield request_output


def _audio_from_mm(mm):
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if isinstance(audio, list):
        chunks = [chunk.reshape(-1) for chunk in audio if hasattr(chunk, "numel") and chunk.numel() > 0]
        return torch.cat(chunks, dim=0) if chunks else None
    return audio


def _sample_rate_from_mm(mm):
    sr = mm.get("sr")
    if isinstance(sr, list):
        sr = sr[-1] if sr else None
    if hasattr(sr, "item"):
        return int(sr.item())
    return int(sr) if sr is not None else 22050


def extract_audio_from_stage_output(stage_output):
    mm = getattr(stage_output, "multimodal_output", None)
    if mm:
        audio = _audio_from_mm(mm)
        if audio is not None:
            return audio, _sample_rate_from_mm(mm)

    for req_output in iter_request_outputs(stage_output):
        for out in getattr(req_output, "outputs", []):
            mm = getattr(out, "multimodal_output", None)
            if mm:
                audio = _audio_from_mm(mm)
                if audio is not None:
                    return audio, _sample_rate_from_mm(mm)
    return None, 22050


TEST_CASES = [
    {"id": "cn_01", "text": "你好，这是一个合成测试。", "lang": "zh"},
    {"id": "en_01", "text": "Hello, this is a voice synthesis test.", "lang": "en"},
]

EMO_TEST_CASES = [
    {
        "id": "emo_text_01",
        "text": "今天天气真好，心情很开心！",
        "lang": "zh",
        "use_emo_text": True,
        "emo_text": "开心快乐",
    },
    {
        "id": "emo_vec_01",
        "text": "这件事让我很生气。",
        "lang": "zh",
        "emo_vector": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    },
]


def build_prefill_prompt_ids(model_id: str, text: str) -> list[int]:
    from vllm_omni.model_executor.models.indextts2.prompt_utils import (
        build_indextts2_prefill_prompt_ids,
    )

    return build_indextts2_prefill_prompt_ids(model_id, text)


def run_offline_tests(args) -> list[dict]:
    from vllm import SamplingParams

    from vllm_omni import Omni

    log("SETUP", "Initializing Omni model...")
    t0 = time.perf_counter()

    deploy_cfg = args.deploy_config
    if deploy_cfg and not os.path.isfile(deploy_cfg):
        deploy_cfg = None

    omni = Omni(
        model=args.model,
        deploy_config=deploy_cfg,
        stage_init_timeout=args.stage_init_timeout,
    )
    log("SETUP", f"Model loaded in {time.perf_counter() - t0:.1f}s, GPU: {get_gpu_info()}")

    gpt_sp = SamplingParams(
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        max_tokens=1500,
        repetition_penalty=10.0,
        stop_token_ids=[8193],
        seed=42,
        detokenize=False,
    )
    s2mel_sp = SamplingParams(temperature=0.0, max_tokens=65536, detokenize=True)
    sampling_params = [gpt_sp, s2mel_sp]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_hash = md5_file(args.ref_audio) if args.ref_audio else "none"
    git_sha = get_git_sha()
    results = []

    all_cases = TEST_CASES[:]
    if not args.skip_emotion:
        all_cases.extend(EMO_TEST_CASES)

    for case in all_cases:
        cid = case["id"]
        text = case["text"]
        log("OFFLINE", f"Starting: id={cid}, text={text!r}")

        additional: dict = {"text": [text]}
        if args.ref_audio:
            additional["voice"] = [args.ref_audio]
        if "emo_text" in case:
            additional["emo_text"] = [case["emo_text"]]
        if "use_emo_text" in case:
            additional["use_emo_text"] = [case["use_emo_text"]]
        if "emo_vector" in case:
            additional["emo_vector"] = [case["emo_vector"]]

        inputs = {
            "prompt_token_ids": build_prefill_prompt_ids(args.model, text),
            "additional_information": additional,
        }

        t_start = time.perf_counter()
        audio_tensor = None
        sr = 22050

        log("OFFLINE", f"[Stage-0] GPT AR talker starting, text={text!r}")
        for stage_outputs in omni.generate(inputs, sampling_params_list=sampling_params):
            audio_candidate, sr_candidate = extract_audio_from_stage_output(stage_outputs)
            if audio_candidate is not None:
                audio_tensor = audio_candidate
                sr = sr_candidate
        t_gen = time.perf_counter() - t_start

        if audio_tensor is None:
            log("OFFLINE", f"[FAIL] No audio output for {cid}")
            results.append({"id": cid, "status": "FAIL", "reason": "no_audio_output"})
            continue

        wav = audio_tensor.float().cpu().numpy()
        wav_path = str(output_dir / f"output_{cid}.wav")
        sf.write(wav_path, wav, sr, format="WAV", subtype="PCM_16")
        duration = len(wav) / sr

        log("OFFLINE", f"[Stage-1] Complete: wav shape={wav.shape}, sr={sr}, duration={duration:.2f}s ({t_gen:.1f}s)")

        # ASR validation
        log("OFFLINE", f"[ASR] Running Whisper on {cid}...")
        t_asr = time.perf_counter()
        try:
            transcript = whisper_transcribe(wav_path, args.whisper_model, args.whisper_device)
        except Exception as e:
            transcript = ""
            log("OFFLINE", f"[ASR] Whisper failed: {e}")
        t_asr = time.perf_counter() - t_asr

        similarity = cosine_similarity_text(transcript, text)
        passed = similarity > 0.9
        marker = "PASS" if passed else "FAIL"

        log(
            "OFFLINE",
            f'[ASR] Transcript: "{transcript}", similarity: {similarity:.3f} {"✓" if passed else "✗"} ({t_asr:.1f}s)',
        )

        results.append(
            {
                "id": cid,
                "text": text,
                "lang": case["lang"],
                "ref_audio_hash": ref_hash,
                "output_path": wav_path,
                "sample_rate": sr,
                "duration_s": round(duration, 3),
                "generation_time_s": round(t_gen, 3),
                "asr_transcript": transcript,
                "cosine_similarity": round(similarity, 4),
                "asr_passed": passed,
                "whisper_model": args.whisper_model,
                "status": marker,
                "git_sha": git_sha,
                "gpu_type": get_gpu_info(),
                "sampling_params": {
                    "temperature": 0.8,
                    "top_p": 0.8,
                    "top_k": 30,
                    "max_tokens": 1500,
                    "repetition_penalty": 10.0,
                },
            }
        )

    manifest_path = str(output_dir / "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"mode": "offline", "results": results}, f, indent=2, ensure_ascii=False)
    log("OFFLINE", f"Manifest written to {manifest_path}")

    passed_count = sum(1 for r in results if r.get("asr_passed"))
    total = len(results)
    log("OFFLINE", f"Summary: {passed_count}/{total} passed ASR threshold (>0.9)")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS2 offline test with ASR validation")
    parser.add_argument("--model", default="IndexTeam/IndexTTS-2")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-dir", default="/content/index-tts-outputs/offline")
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--skip-emotion", action="store_true")
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model for ASR validation. Buildkite helpers currently use 'small'.",
    )
    parser.add_argument(
        "--whisper-device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device passed to whisper.load_model; use cpu when GPU memory is tight.",
    )
    args = parser.parse_args()

    log("SETUP", "IndexTTS2 Offline Test Starting...")
    results = run_offline_tests(args)

    failed = [r for r in results if not r.get("asr_passed")]
    if failed:
        log("RESULT", f"FAILED: {len(failed)} test(s) below threshold")
        for r in failed:
            log("RESULT", f"  {r['id']}: similarity={r.get('cosine_similarity', 'N/A')}")
        sys.exit(1)
    else:
        log("RESULT", "ALL PASSED")


if __name__ == "__main__":
    main()
