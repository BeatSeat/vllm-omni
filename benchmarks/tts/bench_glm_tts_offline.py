#!/usr/bin/env python3
"""Offline GLM-TTS benchmark for vLLM Omni.

Supports:
- sync one-shot (Omni.generate)
- streaming (AsyncOmni.generate with async_chunk config)
- voice cloning with ref_audio + ref_text
- batch inputs from seed-tts meta.lst or JSONL

Usage::

    # Single text, sync mode
    python benchmarks/tts/bench_glm_tts_offline.py \
        --model zai-org/GLM-TTS \
        --text "今天天气真不错，适合出去散散步。" \
        --ref-audio /path/to/ref.wav \
        --ref-text "他当时还跟线下其他的站姐吵架，然后，打架进局子了。" \
        --output-dir results/audio/

    # Seed-TTS zh dataset
    python benchmarks/tts/bench_glm_tts_offline.py \
        --model zai-org/GLM-TTS \
        --seed-tts-path /path/to/seedtts_testset \
        --seed-tts-locale zh \
        --num-prompts 20 \
        --output-dir results/audio/

    # Streaming (async_chunk)
    python benchmarks/tts/bench_glm_tts_offline.py \
        --model zai-org/GLM-TTS \
        --stage-configs-path vllm_omni/deploy/glm_tts.yaml \
        --seed-tts-path /path/to/seedtts_testset \
        --seed-tts-locale zh \
        --output-dir results/audio/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

logger = logging.getLogger(__name__)


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "vllm_omni").is_dir():
            return candidate
    return start.parents[2]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
DEFAULT_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "deploy" / "glm_tts.yaml"


@dataclass(frozen=True, slots=True)
class PromptSpec:
    text: str
    label: str
    ref_audio_path: str
    ref_text: str
    utterance_id: str = ""


def _require_soundfile():
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError("soundfile required: pip install soundfile") from exc
    return sf


def _load_ref_audio(path: str) -> tuple[np.ndarray, int]:
    sf = _require_soundfile()
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _build_prompt(
    text: str,
    ref_audio: tuple[np.ndarray, int],
    ref_text: str,
    model: str,
) -> dict[str, Any]:
    from vllm_omni.model_executor.models.glm_tts.glm_tts import (
        build_glm_tts_prefill_metadata,
    )

    return {
        "prompt": text,
        "multi_modal_data": {"audio": ref_audio},
        "modalities": ["audio"],
        "mm_processor_kwargs": {"prompt_text": ref_text},
        "model_intermediate_buffer": build_glm_tts_prefill_metadata(
            model,
            text,
            ref_text,
            trust_remote_code=True,
        ),
    }


def _extract_audio_tensor(mm: dict[str, Any]) -> torch.Tensor:
    audio = mm.get("audio", mm.get("model_outputs"))
    if audio is None:
        raise ValueError("No audio output found in multimodal output.")
    if isinstance(audio, list):
        parts = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio]
        audio = torch.cat(parts, dim=-1) if parts else torch.zeros(0)
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    return audio.float().cpu().reshape(-1)


def _extract_sample_rate(mm: dict[str, Any]) -> int:
    sr_raw = mm.get("sr", 24000)
    if isinstance(sr_raw, list) and sr_raw:
        sr_raw = sr_raw[-1]
    if hasattr(sr_raw, "item"):
        return int(sr_raw.item())
    return int(sr_raw)


def _write_audio_tensor(output_path: Path, audio_tensor: Any, sample_rate: int) -> None:
    sf = _require_soundfile()
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.float().cpu().clamp(-1.0, 1.0).numpy()
    else:
        audio_np = torch.as_tensor(audio_tensor).float().cpu().clamp(-1.0, 1.0).numpy()
    sf.write(output_path, audio_np, sample_rate, format="WAV", subtype="PCM_16")


def _iter_request_multimodal_outputs(request_output: Any):
    outputs = getattr(request_output, "outputs", None)
    if outputs:
        for output in outputs:
            mm = getattr(output, "multimodal_output", None)
            if isinstance(mm, dict):
                yield mm
    mm = getattr(request_output, "multimodal_output", None)
    if isinstance(mm, dict):
        yield mm


def _emit_offline_metrics(
    *,
    request_id: str,
    label: str,
    elapsed_s: float,
    first_audio_elapsed: float | None,
    audio_duration_s: float,
) -> None:
    rtf = round(elapsed_s / audio_duration_s, 6) if audio_duration_s > 0 else None
    metrics = {
        "request_id": request_id,
        "label": label,
        "ttfp_ms": round(first_audio_elapsed * 1000.0, 3) if first_audio_elapsed is not None else None,
        "audio_duration_s": round(audio_duration_s, 6),
        "generation_time_s": round(elapsed_s, 6),
        "rtf": rtf,
    }
    print(f"[OfflineMetrics] {json.dumps(metrics)}")


# ---- Seed-TTS meta.lst loading ----


def _parse_seed_tts_meta(meta_path: Path) -> list[dict[str, str]]:
    rows = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            logger.warning("Skipping malformed meta.lst line: %r", line[:120])
            continue
        rows.append(
            {
                "utterance_id": parts[0].strip(),
                "ref_text": parts[1].strip(),
                "wav_rel": parts[2].strip(),
                "target_text": parts[3].strip(),
            }
        )
    return rows


def _load_prompt_specs(args) -> list[PromptSpec]:
    specs: list[PromptSpec] = []

    if args.seed_tts_path is not None:
        root = Path(args.seed_tts_path).expanduser().resolve()
        locale = args.seed_tts_locale
        meta = root / locale / "meta.lst"
        if not meta.is_file():
            raise FileNotFoundError(f"meta.lst not found: {meta}")
        rows = _parse_seed_tts_meta(meta)
        if not rows:
            raise ValueError(f"No valid rows in {meta}")
        wav_dir = root / locale
        for i, row in enumerate(rows):
            wav_path = (wav_dir / row["wav_rel"]).resolve()
            if not wav_path.is_file():
                logger.warning("Missing wav for %s: %s", row["utterance_id"], wav_path)
                continue
            specs.append(
                PromptSpec(
                    text=row["target_text"],
                    label=f"seedtts_{row['utterance_id']}",
                    ref_audio_path=str(wav_path),
                    ref_text=row["ref_text"],
                    utterance_id=row["utterance_id"],
                )
            )
            if args.num_prompts and len(specs) >= args.num_prompts:
                break
        return specs

    if args.jsonl_prompts is not None:
        with open(args.jsonl_prompts, encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or "text" not in item:
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} requires 'text'")
                ref_audio = item.get("ref_audio", args.ref_audio)
                ref_text = item.get("ref_text", args.ref_text)
                if not ref_audio or not ref_text:
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} requires both ref_audio and ref_text")
                specs.append(
                    PromptSpec(
                        text=item["text"].strip(),
                        label=f"item{len(specs) + 1:03d}",
                        ref_audio_path=ref_audio,
                        ref_text=ref_text,
                    )
                )
        if not specs:
            raise ValueError(f"No prompts found in {args.jsonl_prompts}")
        return specs

    if not args.ref_audio or not args.ref_text:
        raise ValueError("GLM-TTS requires --ref-audio and --ref-text (or --seed-tts-path)")
    specs.append(
        PromptSpec(
            text=args.text,
            label="item001",
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
        )
    )
    return specs


# ---- Sync mode ----


def _run_sync(args) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    ref_audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    def _get_ref_audio(path: str) -> tuple[np.ndarray, int]:
        if path not in ref_audio_cache:
            ref_audio_cache[path] = _load_ref_audio(path)
        return ref_audio_cache[path]

    def _run_single(
        spec: PromptSpec,
        *,
        request_prefix: str,
        save_outputs: bool,
        run_index: int | None = None,
    ) -> tuple[list[Path], float | None, float, float]:
        ref_audio = _get_ref_audio(spec.ref_audio_path)
        prompt = _build_prompt(spec.text, ref_audio, spec.ref_text, args.model)

        saved_paths: list[Path] = []
        first_audio_elapsed: float | None = None
        total_audio_duration_s = 0.0
        t_start = time.perf_counter()

        for stage_outputs in omni.generate(prompt):
            request_output = stage_outputs.request_output
            if request_output is None:
                continue
            for j, mm in enumerate(_iter_request_multimodal_outputs(request_output)):
                try:
                    audio_tensor = _extract_audio_tensor(mm)
                    sr = _extract_sample_rate(mm)
                    if int(audio_tensor.numel()) > 0:
                        if first_audio_elapsed is None:
                            first_audio_elapsed = time.perf_counter() - t_start
                        total_audio_duration_s += float(audio_tensor.numel()) / float(sr)
                except ValueError:
                    continue
                if save_outputs:
                    stem = f"run{run_index + 1}_{spec.label}" if j == 0 else f"run{run_index + 1}_{spec.label}_{j}"
                    out_path = output_dir / f"output_{stem}.wav"
                    _write_audio_tensor(out_path, audio_tensor, sr)
                    saved_paths.append(out_path)

        elapsed_s = time.perf_counter() - t_start
        return saved_paths, first_audio_elapsed, elapsed_s, total_audio_duration_s

    prompt_specs = args.prompt_specs

    # Warmup
    if args.warmup_runs:
        warmup_spec = prompt_specs[0]
        print(f"Warmup: {args.warmup_runs} run(s) using first prompt; outputs discarded.")
        for wi in range(args.warmup_runs):
            t_w = time.perf_counter()
            _, ttfp, elapsed, dur = _run_single(
                warmup_spec,
                request_prefix=f"warmup{wi + 1}",
                save_outputs=False,
            )
            ttfp_s = f", ttfp={ttfp:.2f}s" if ttfp is not None else ""
            rtf_s = f", rtf={elapsed / dur:.3f}" if dur > 0 else ""
            print(f"Warmup {wi + 1}/{args.warmup_runs}: {time.perf_counter() - t_w:.2f}s{ttfp_s}{rtf_s}")

    # Measured runs
    t_total = time.perf_counter()
    all_paths: list[Path] = []
    all_metrics: list[dict[str, Any]] = []

    for run in range(args.num_runs):
        for pi, spec in enumerate(prompt_specs):
            paths, ttfp, elapsed, dur = _run_single(
                spec,
                request_prefix=f"sync_run{run + 1}_{pi + 1:03d}",
                save_outputs=True,
                run_index=run,
            )
            all_paths.extend(paths)

            rtf = elapsed / dur if dur > 0 else float("inf")
            all_metrics.append(
                {
                    "label": spec.label,
                    "utterance_id": spec.utterance_id,
                    "audio_duration_s": round(dur, 3),
                    "generation_time_s": round(elapsed, 3),
                    "rtf": round(rtf, 4),
                    "ttfp_ms": round(ttfp * 1000, 1) if ttfp is not None else None,
                }
            )

            ttfp_s = f", ttfp={ttfp:.2f}s" if ttfp is not None else ""
            print(
                f"[sync] run {run + 1}/{args.num_runs}, "
                f"prompt {pi + 1}/{len(prompt_specs)} ({spec.label}): "
                f"audio={dur:.2f}s, gen={elapsed:.2f}s, rtf={rtf:.3f}{ttfp_s}"
            )
            _emit_offline_metrics(
                request_id=f"sync_run{run + 1}_{spec.label}",
                label=spec.label,
                elapsed_s=elapsed,
                first_audio_elapsed=ttfp,
                audio_duration_s=dur,
            )

    total_elapsed = time.perf_counter() - t_total
    _print_summary(all_metrics, total_elapsed, args)
    _save_results_json(all_metrics, args)
    return all_paths


# ---- Streaming mode ----


def _extract_stream_finished(stage_output: Any) -> bool:
    request_output = getattr(stage_output, "request_output", None)
    finished = getattr(request_output, "finished", None)
    if finished is not None:
        return bool(finished)
    return bool(getattr(stage_output, "finished", False))


async def _run_streaming(args) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omni = AsyncOmni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    ref_audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    def _get_ref_audio(path: str) -> tuple[np.ndarray, int]:
        if path not in ref_audio_cache:
            ref_audio_cache[path] = _load_ref_audio(path)
        return ref_audio_cache[path]

    async def _collect_one(
        spec: PromptSpec,
        request_id: str,
    ) -> tuple[torch.Tensor, int, float, float | None]:
        ref_audio = _get_ref_audio(spec.ref_audio_path)
        prompt = _build_prompt(spec.text, ref_audio, spec.ref_text, args.model)

        delta_chunks: list[torch.Tensor] = []
        sample_rate = 24000
        prev_total = 0
        t_start = time.perf_counter()
        first_audio_elapsed: float | None = None

        async for stage_output in omni.generate(prompt, request_id=request_id):
            mm = getattr(stage_output, "multimodal_output", None)
            if not isinstance(mm, dict):
                ro = getattr(stage_output, "request_output", None)
                if ro is None:
                    continue
                mm = getattr(ro, "multimodal_output", None)
                if not isinstance(mm, dict) and getattr(ro, "outputs", None):
                    mm = getattr(ro.outputs[0], "multimodal_output", None)
            if not isinstance(mm, dict):
                continue

            sample_rate = _extract_sample_rate(mm)
            try:
                w = _extract_audio_tensor(mm)
                n = int(w.numel())
                if n == 0:
                    continue
                finished = _extract_stream_finished(stage_output)
                if n > prev_total:
                    delta = w.reshape(-1)[prev_total:]
                    prev_total = n
                elif finished and n == prev_total:
                    delta = w.reshape(-1)[:0]
                else:
                    delta = w.reshape(-1)
                    prev_total += int(delta.numel())
                if int(delta.numel()) > 0:
                    delta_chunks.append(delta)
                    if first_audio_elapsed is None:
                        first_audio_elapsed = time.perf_counter() - t_start
            except ValueError:
                pass

        if not delta_chunks:
            raise RuntimeError("No audio chunks received.")

        audio_cat = torch.cat([c.reshape(-1) for c in delta_chunks], dim=0)
        elapsed = time.perf_counter() - t_start
        return audio_cat, sample_rate, elapsed, first_audio_elapsed

    prompt_specs = args.prompt_specs

    # Warmup
    if args.warmup_runs:
        spec = prompt_specs[0]
        print(f"Warmup: {args.warmup_runs} run(s) using first prompt.")
        for wi in range(args.warmup_runs):
            t_w = time.perf_counter()
            rid = f"warmup_stream_{wi + 1}_{uuid.uuid4().hex[:8]}"
            audio, sr, elapsed, ttfp = await _collect_one(spec, rid)
            await omni.engine.abort_async([rid])
            dur = float(audio.numel()) / float(sr) if sr > 0 else 0
            ttfp_s = f", ttfp={ttfp:.2f}s" if ttfp is not None else ""
            rtf_s = f", rtf={elapsed / dur:.3f}" if dur > 0 else ""
            print(f"Warmup {wi + 1}/{args.warmup_runs}: {time.perf_counter() - t_w:.2f}s{ttfp_s}{rtf_s}")

    # Measured runs
    t_total = time.perf_counter()
    all_paths: list[Path] = []
    all_metrics: list[dict[str, Any]] = []

    for run in range(args.num_runs):
        for pi, spec in enumerate(prompt_specs):
            rid = f"stream_{run + 1}_{spec.label}_{uuid.uuid4().hex[:8]}"
            audio, sr, elapsed, ttfp = await _collect_one(spec, rid)
            await omni.engine.abort_async([rid])

            dur = float(audio.numel()) / float(sr) if sr > 0 else 0
            rtf = elapsed / dur if dur > 0 else float("inf")

            out_path = output_dir / f"output_run{run + 1}_{spec.label}.wav"
            _write_audio_tensor(out_path, audio, sr)
            all_paths.append(out_path)

            all_metrics.append(
                {
                    "label": spec.label,
                    "utterance_id": spec.utterance_id,
                    "audio_duration_s": round(dur, 3),
                    "generation_time_s": round(elapsed, 3),
                    "rtf": round(rtf, 4),
                    "ttfp_ms": round(ttfp * 1000, 1) if ttfp is not None else None,
                }
            )

            ttfp_s = f", ttfp={ttfp:.2f}s" if ttfp is not None else ""
            print(
                f"[streaming] run {run + 1}/{args.num_runs}, "
                f"prompt {pi + 1}/{len(prompt_specs)} ({spec.label}): "
                f"audio={dur:.2f}s, gen={elapsed:.2f}s, rtf={rtf:.3f}{ttfp_s}"
            )
            _emit_offline_metrics(
                request_id=rid,
                label=spec.label,
                elapsed_s=elapsed,
                first_audio_elapsed=ttfp,
                audio_duration_s=dur,
            )

    total_elapsed = time.perf_counter() - t_total
    _print_summary(all_metrics, total_elapsed, args)
    _save_results_json(all_metrics, args)
    return all_paths


# ---- Summary / results ----


def _print_summary(metrics: list[dict[str, Any]], total_elapsed: float, args: Any) -> None:
    if not metrics:
        return
    print(f"\n{'=' * 70}")
    mode = "streaming" if _is_streaming_config(args.stage_configs_path) else "sync"
    print(f"GLM-TTS OFFLINE BENCHMARK SUMMARY ({mode} mode)")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"Prompts: {len(args.prompt_specs)}, Runs: {args.num_runs}, Warmups: {args.warmup_runs}")
    print(f"Total wall time: {total_elapsed:.1f}s")
    print()
    print(f"{'Label':<30} {'Audio(s)':>8} {'Gen(s)':>8} {'RTF':>8} {'TTFP(ms)':>10}")
    print("-" * 70)
    for m in metrics:
        ttfp_str = f"{m['ttfp_ms']:.0f}" if m["ttfp_ms"] is not None else "n/a"
        print(
            f"{m['label']:<30} {m['audio_duration_s']:>8.2f} "
            f"{m['generation_time_s']:>8.2f} {m['rtf']:>8.4f} {ttfp_str:>10}"
        )

    rtfs = [m["rtf"] for m in metrics if m["rtf"] < float("inf")]
    ttfps = [m["ttfp_ms"] for m in metrics if m["ttfp_ms"] is not None]
    durs = [m["audio_duration_s"] for m in metrics]
    gens = [m["generation_time_s"] for m in metrics]

    print("-" * 70)
    if rtfs:
        import statistics

        print(f"RTF  — mean: {statistics.fmean(rtfs):.4f}, median: {statistics.median(rtfs):.4f}")
    if ttfps:
        import statistics

        print(f"TTFP — mean: {statistics.fmean(ttfps):.0f}ms, median: {statistics.median(ttfps):.0f}ms")
    if durs:
        print(f"Audio — total: {sum(durs):.1f}s, mean: {sum(durs) / len(durs):.2f}s")
    if gens:
        print(f"Gen   — total: {sum(gens):.1f}s")
    print(f"{'=' * 70}")


def _save_results_json(metrics: list[dict[str, Any]], args: Any) -> None:
    if not args.output_dir:
        return
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = args.model.replace("/", "_")
    mode = "streaming" if _is_streaming_config(args.stage_configs_path) else "sync"
    out_path = Path(args.output_dir) / f"bench_glm_tts_{model_slug}_{mode}_{ts}.json"
    result = {
        "model": args.model,
        "mode": mode,
        "stage_config": args.stage_configs_path,
        "num_prompts": len(args.prompt_specs),
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {out_path}")


def _is_streaming_config(path: str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    text = p.read_text(encoding="utf-8")
    return "async_chunk: true" in text.lower() or "async_chunk:true" in text.lower()


def parse_args():
    parser = FlexibleArgumentParser(description="Offline GLM-TTS benchmark (sync + streaming, voice cloning)")
    parser.add_argument("--model", type=str, default="zai-org/GLM-TTS")
    parser.add_argument(
        "--text",
        type=str,
        default="今天天气真不错，适合出去散散步。",
        help="Text to synthesize (single-prompt mode).",
    )
    parser.add_argument("--ref-audio", type=str, default=None, help="Reference audio WAV path.")
    parser.add_argument("--ref-text", type=str, default=None, help="Transcript of reference audio.")

    parser.add_argument("--seed-tts-path", type=str, default=None, help="Root of seed-tts-eval dataset.")
    parser.add_argument("--seed-tts-locale", type=str, default="zh", choices=["en", "zh"])
    parser.add_argument("--num-prompts", type=int, default=None, help="Max prompts from dataset.")
    parser.add_argument("--jsonl-prompts", type=str, default=None, help="JSONL with text/ref_audio/ref_text.")

    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=str(DEFAULT_STAGE_CONFIG),
        help="Stage config YAML. async_chunk in content → streaming mode.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=str, default="output_audio_glm_tts")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--log-stats", dest="log_stats", action="store_true")
    parser.add_argument("--no-log-stats", dest="log_stats", action="store_false")
    parser.set_defaults(log_stats=True)

    args = parser.parse_args()

    if args.seed_tts_path is None and args.jsonl_prompts is None:
        if not args.ref_audio or not args.ref_text:
            parser.error("--ref-audio and --ref-text required (or use --seed-tts-path / --jsonl-prompts)")

    try:
        args.prompt_specs = _load_prompt_specs(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    return args


def main(args) -> int:
    logging.basicConfig(level=logging.INFO)
    is_streaming = _is_streaming_config(args.stage_configs_path)

    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"Mode: {'streaming' if is_streaming else 'sync'}")
    print(f"Prompts: {len(args.prompt_specs)}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Num runs: {args.num_runs}")

    if is_streaming:
        asyncio.run(_run_streaming(args))
    else:
        _run_sync(args)
    return 0


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    raise SystemExit(main(parse_args()))
