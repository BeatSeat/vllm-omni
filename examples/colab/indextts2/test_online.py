#!/usr/bin/env python3
"""IndexTTS2 online serving test for Colab L4 with ASR validation.

Starts vllm serve, sends requests to /v1/audio/speech, validates via Whisper ASR,
and writes results to /content/index-tts-outputs/online/.

Usage (on Colab):
  python test_online.py \
    --model IndexTeam/IndexTTS-2 \
    --ref-audio /path/to/voice.wav \
    --output-dir /content/index-tts-outputs/online
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


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
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def encode_audio_b64(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1]
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}.get(ext, "audio/wav")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


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


def wait_for_server(url: str, timeout: int = 900) -> bool:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
        elapsed = timeout - (deadline - time.time())
        if int(elapsed) % 30 == 0:
            log("SERVER", f"Waiting for server... ({int(elapsed)}s)")
    return False


def start_server(
    model: str,
    port: int,
    gpu_mem: float,
    deploy_config: str | None,
    stage_init_timeout: int,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_mem),
        "--stage-init-timeout",
        str(stage_init_timeout),
    ]
    if deploy_config:
        cmd.extend(["--deploy-config", deploy_config])
    log("SERVER", f"Starting: {' '.join(cmd)}")
    # Inherit stdout/stderr so startup logs flow into the parent tee.  Leaving a
    # PIPE unread can block noisy vLLM startup before /health becomes ready.
    proc = subprocess.Popen(cmd)
    return proc


TEST_CASES = [
    {"id": "cn_01", "text": "你好，这是一个合成测试。", "lang": "zh"},
    {"id": "en_01", "text": "Welcome to explore voice synthesis technology.", "lang": "en"},
]

EMO_TEST_CASES = [
    {"id": "emo_text_01", "text": "今天天气真好！", "lang": "zh", "extra_params": {"emo_text": "开心快乐"}},
    {
        "id": "emo_vec_01",
        "text": "这件事让我很生气。",
        "lang": "zh",
        "extra_params": {"emo_vector": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]},
    },
]

NEGATIVE_CASES = [
    {
        "id": "neg_no_ref",
        "text": "Test",
        "expect_error": True,
        "payload_override": {"ref_audio": None},
        "desc": "missing ref_audio",
    },
    {
        "id": "neg_bad_b64",
        "text": "Test",
        "expect_error": True,
        "payload_override": {"ref_audio": "data:audio/wav;base64,INVALID!!!"},
        "desc": "malformed base64",
    },
    {
        "id": "neg_bad_emo_vec",
        "text": "Test",
        "expect_error": True,
        "extra_params": {"emo_vector": [0.1, 0.2, 0.3]},
        "desc": "wrong emo_vector length",
    },
    {
        "id": "neg_bad_alpha",
        "text": "Test",
        "expect_error": True,
        "extra_params": {"emo_alpha": 1.5},
        "desc": "emo_alpha > 1.0",
    },
]


def run_online_tests(args) -> list[dict]:
    import httpx

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_b64 = encode_audio_b64(args.ref_audio) if args.ref_audio else None
    ref_hash = md5_file(args.ref_audio) if args.ref_audio else "none"
    git_sha = get_git_sha()
    api_base = f"http://localhost:{args.port}"
    results = []

    server_proc = None
    if not args.server_already_running:
        log("SERVER", "Launching vLLM serve...")
        server_proc = start_server(
            args.model,
            args.port,
            args.gpu_memory_utilization,
            args.deploy_config,
            args.stage_init_timeout,
        )
        log("SERVER", "Waiting for health check...")
        if not wait_for_server(api_base, timeout=args.server_start_timeout):
            log("SERVER", f"FAIL: Server did not start within {args.server_start_timeout}s")
            if server_proc:
                server_proc.kill()
            return [{"id": "server_start", "status": "FAIL", "reason": "timeout"}]
        log("SERVER", f"Server ready, GPU: {get_gpu_info()}")

    all_cases = TEST_CASES[:]
    if not args.skip_emotion:
        all_cases.extend(EMO_TEST_CASES)

    try:
        for case in all_cases:
            cid = case["id"]
            text = case["text"]
            log("ONLINE", f"Starting: id={cid}, text={text!r}")

            payload = {
                "model": args.model,
                "input": text,
                "voice": "default",
                "response_format": "wav",
                "ref_audio": ref_b64,
            }
            if "extra_params" in case:
                payload["extra_params"] = case["extra_params"]

            t_start = time.perf_counter()
            log("ONLINE", "[Request] POST /v1/audio/speech")
            with httpx.Client(timeout=300) as client:
                resp = client.post(
                    f"{api_base}/v1/audio/speech",
                    json=payload,
                    headers={"Authorization": "Bearer sk-empty"},
                )
            t_gen = time.perf_counter() - t_start

            if resp.status_code != 200:
                log("ONLINE", f"[FAIL] HTTP {resp.status_code}: {resp.text[:200]}")
                results.append({"id": cid, "status": "FAIL", "reason": f"http_{resp.status_code}"})
                continue

            wav_path = str(output_dir / f"output_{cid}.wav")
            with open(wav_path, "wb") as f:
                f.write(resp.content)

            import soundfile as sf_lib

            data, sr = sf_lib.read(wav_path)
            sf_lib.write(wav_path, data, sr, format="WAV", subtype="PCM_16")
            duration = len(data) / sr
            log("ONLINE", f"[Response] {len(resp.content):,} bytes, {sr} Hz, {duration:.2f}s ({t_gen:.1f}s)")

            log("ONLINE", f"[ASR] Running Whisper on {cid}...")
            t_asr = time.perf_counter()
            try:
                transcript = whisper_transcribe(wav_path, args.whisper_model, args.whisper_device)
            except Exception as e:
                transcript = ""
                log("ONLINE", f"[ASR] Whisper failed: {e}")
            t_asr = time.perf_counter() - t_asr

            similarity = cosine_similarity_text(transcript, text)
            passed = similarity > 0.9
            marker = "PASS" if passed else "FAIL"
            log(
                "ONLINE",
                f'[ASR] Transcript: "{transcript}", similarity: {similarity:.3f} {"✓" if passed else "✗"} ({t_asr:.1f}s)',
            )

            results.append(
                {
                    "id": cid,
                    "text": text,
                    "lang": case.get("lang", ""),
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
                }
            )

        # Negative API tests
        if not args.skip_negative:
            log("ONLINE", "Running negative API tests...")
            for neg in NEGATIVE_CASES:
                nid = neg["id"]
                payload = {
                    "model": args.model,
                    "input": neg["text"],
                    "voice": "default",
                    "response_format": "wav",
                    "ref_audio": ref_b64,
                }
                if "payload_override" in neg:
                    for k, v in neg["payload_override"].items():
                        if v is None:
                            payload.pop(k, None)
                        else:
                            payload[k] = v
                if "extra_params" in neg:
                    payload["extra_params"] = neg["extra_params"]

                with httpx.Client(timeout=60) as client:
                    resp = client.post(
                        f"{api_base}/v1/audio/speech",
                        json=payload,
                        headers={"Authorization": "Bearer sk-empty"},
                    )

                got_error = resp.status_code >= 400
                passed = got_error == neg["expect_error"]
                marker = "PASS" if passed else "FAIL"
                log("ONLINE", f"[Negative] {nid} ({neg['desc']}): HTTP {resp.status_code} → {marker}")
                results.append(
                    {
                        "id": nid,
                        "desc": neg["desc"],
                        "http_status": resp.status_code,
                        "expected_error": neg["expect_error"],
                        "got_error": got_error,
                        "status": marker,
                    }
                )

    finally:
        if server_proc:
            log("SERVER", "Shutting down server...")
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    manifest_path = str(output_dir / "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"mode": "online", "results": results}, f, indent=2, ensure_ascii=False)
    log("ONLINE", f"Manifest written to {manifest_path}")

    asr_results = [r for r in results if "asr_passed" in r]
    neg_results = [r for r in results if "expected_error" in r]
    asr_passed = sum(1 for r in asr_results if r["asr_passed"])
    neg_passed = sum(1 for r in neg_results if r["status"] == "PASS")
    log("ONLINE", f"ASR: {asr_passed}/{len(asr_results)} passed, Negative: {neg_passed}/{len(neg_results)} passed")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS2 online serving test")
    parser.add_argument("--model", default="IndexTeam/IndexTTS-2")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-dir", default="/content/index-tts-outputs/online")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--server-start-timeout", type=int, default=900)
    parser.add_argument("--server-already-running", action="store_true")
    parser.add_argument("--skip-emotion", action="store_true")
    parser.add_argument("--skip-negative", action="store_true")
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model for ASR validation. Buildkite helpers currently use 'small'.",
    )
    parser.add_argument(
        "--whisper-device",
        default="cpu",
        choices=("auto", "cpu", "cuda"),
        help="Device passed to whisper.load_model; CPU avoids competing with the serving GPU.",
    )
    args = parser.parse_args()

    log("SETUP", "IndexTTS2 Online Serving Test Starting...")
    results = run_online_tests(args)

    failed = [r for r in results if r.get("status") == "FAIL"]
    if failed:
        log("RESULT", f"FAILED: {len(failed)} test(s)")
        for r in failed:
            log("RESULT", f"  {r['id']}: {r.get('reason', r.get('desc', ''))}")
        sys.exit(1)
    else:
        log("RESULT", "ALL PASSED")


if __name__ == "__main__":
    main()
