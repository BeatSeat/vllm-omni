"""Per-shape CUDA graph capture for the S2Mel DiT transformer core.

Profiling showed the DiT estimator forward is CPU launch-bound (the GPU
is idle ~85% of CFM wall time). Within one request the CFM runs 12 Euler
steps with *identical* tensor shapes (T_in never changes), so the graph
captured on the first step is replayed by every subsequent step. Across
requests T_in varies, so graphs are cached per shape with a small LRU;
a cache miss costs roughly one extra eager step (warm run + capture).

Capture safety relies on the IndexTTS2 S2Mel decoder only enabling this
runner for full-mask shapes:

- ``x_lens == T`` for every sample (the decoder feeds full-length sequences), so
  the attention padding mask is all-True and dense flash attention is
  mathematically identical. The decoder sets ``_assume_full_mask`` on
  the gpt_fast ``Attention`` modules so the backend skips its
  ``torch.any(~mask)`` D2H sync, which would abort stream capture.

Modeled on ``CUDAGraphGLMTTSDiTWrapper``
(vllm_omni/model_executor/models/glm_tts/glm_tts_dit_wrapper.py).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_c: torch.Tensor
    static_pos: torch.Tensor
    static_mask: torch.Tensor | None
    static_out: torch.Tensor


class CUDAGraphDiTRunner:
    """Replay the gpt_fast Transformer forward via per-shape CUDA graphs."""

    def __init__(self, transformer: torch.nn.Module, *, max_graphs: int = 4, enabled: bool = True) -> None:
        self.transformer = transformer
        self.max_graphs = int(max_graphs)
        self.enabled = bool(enabled)
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self.last_call_info: dict[str, object] = {}
        self._stats = {
            "calls": 0,
            "hits": 0,
            "captures": 0,
            "capture_failures": 0,
            "evictions": 0,
            "eager": 0,
        }

    def stats_snapshot(self) -> dict[str, object]:
        """Return bounded cumulative graph-cache telemetry."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
        }

    def _replay(
        self,
        entry: _GraphEntry,
        *,
        x_in: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Replay with only the inputs that vary between Euler steps.

        ``input_pos`` is the same arange slice for a given cache key and the
        runner is enabled only for all-True masks, so their capture-time
        buffers remain valid.  The returned output is also the graph's static
        buffer: all downstream consumers are enqueued on the same CUDA stream
        before the next replay can overwrite it.
        """
        entry.static_x.copy_(x_in)
        entry.static_c.copy_(c)
        entry.graph.replay()
        return entry.static_out

    def __call__(
        self,
        x_in: torch.Tensor,
        c: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self._stats["calls"] += 1
        if not self.enabled or x_in.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            self._stats["eager"] += 1
            self.last_call_info = {
                "mode": "eager",
                "reason": "ineligible",
                "shape": tuple(x_in.shape),
                "cache_size": len(self._cache),
            }
            return self.transformer(x_in, c, input_pos, mask)

        mask_key = None if mask is None else (tuple(mask.shape), mask.dtype)
        device_key = (x_in.device.type, x_in.device.index)
        key = (
            device_key,
            tuple(x_in.shape),
            x_in.dtype,
            tuple(c.shape),
            c.dtype,
            tuple(input_pos.shape),
            input_pos.dtype,
            mask_key,
        )
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(x_in, c, input_pos, mask)
            if entry is None:
                self._stats["capture_failures"] += 1
                self._stats["eager"] += 1
                self.last_call_info = {
                    "mode": "eager",
                    "reason": "capture_failed",
                    "shape": tuple(x_in.shape),
                    "cache_size": len(self._cache),
                }
                return self.transformer(x_in, c, input_pos, mask)
            self._stats["captures"] += 1
            self._cache[key] = entry
            while len(self._cache) > self.max_graphs:
                _, evicted = self._cache.popitem(last=False)
                del evicted
                self._stats["evictions"] += 1
            # Kernels are only recorded (not executed) during capture, so
            # replay once to produce the first real output.
            entry.graph.replay()
            self.last_call_info = {
                "mode": "graph",
                "reason": "capture",
                "shape": tuple(x_in.shape),
                "cache_size": len(self._cache),
            }
            return entry.static_out

        self._cache.move_to_end(key)
        self._stats["hits"] += 1
        output = self._replay(entry, x_in=x_in, c=c)
        self.last_call_info = {
            "mode": "graph",
            "reason": "hit",
            "shape": tuple(x_in.shape),
            "cache_size": len(self._cache),
        }
        if self._stats["calls"] % 1000 == 0:
            logger.info("S2Mel DiT CUDA graph stats: %s", self.stats_snapshot())
        return output

    def _capture(
        self,
        x_in: torch.Tensor,
        c: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> _GraphEntry | None:
        static_x = x_in.clone()
        static_c = c.clone()
        static_pos = input_pos.clone()
        # The mask content is constant (all-True; see module docstring) and
        # ignored by Attention via `_assume_full_mask`, but keep a static
        # clone so the captured graph never dereferences a freed tensor.
        static_mask = mask.contiguous().clone() if mask is not None else None
        try:
            # Eager warm run: cuDNN/flash-attn autotuning and lazy inits
            # must happen outside stream capture.
            with torch.no_grad():
                _ = self.transformer(static_x, static_c, static_pos, static_mask)
            torch.accelerator.synchronize(x_in.device)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_out = self.transformer(static_x, static_c, static_pos, static_mask)
        except Exception:
            logger.warning(
                "Disabling S2Mel DiT CUDA graphs after capture failure for shape=%s",
                tuple(x_in.shape),
                exc_info=True,
            )
            self.enabled = False
            return None
        logger.debug("Captured S2Mel DiT CUDA graph for shape=%s", tuple(x_in.shape))
        return _GraphEntry(
            graph=graph,
            static_x=static_x,
            static_c=static_c,
            static_pos=static_pos,
            static_mask=static_mask,
            static_out=static_out,
        )


_FATAL_CUDA_ERROR_MARKERS = (
    "device-side assert",
    "illegal memory access",
    "misaligned address",
    "uncorrectable ecc",
    "hardware stack error",
    "illegal instruction",
)


def _is_fatal_cuda_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _FATAL_CUDA_ERROR_MARKERS)


def _is_oom_error(exc: BaseException) -> bool:
    oom_types: tuple[type[BaseException], ...] = (torch.OutOfMemoryError,)
    if hasattr(torch.cuda, "OutOfMemoryError"):
        oom_types = (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)
    return isinstance(exc, oom_types) or "out of memory" in str(exc).lower()


@dataclass
class _StepGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_t: torch.Tensor
    static_prompt_x: torch.Tensor
    static_style: torch.Tensor
    static_cond: torch.Tensor
    static_x_lens: torch.Tensor
    static_pre_mask: tuple[torch.Tensor, torch.Tensor | None] | None
    static_unpad_data: tuple[torch.Tensor, torch.Tensor] | None
    static_out: torch.Tensor
    static_reflect_cache: dict | None = None
    stream: object | None = None
    in_use: bool = False


class _SolveGraphSession:
    """Per-solve_euler execution helper backed by CUDAGraphDiTStepRunner."""

    def __init__(
        self,
        runner: CUDAGraphDiTStepRunner,
        *,
        entry: _StepGraphEntry | None,
        cache_key: tuple | None,
        is_ragged: bool,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        style: torch.Tensor,
        precomputed_cond: torch.Tensor,
        pre_mask: tuple[torch.Tensor, torch.Tensor | None] | None,
        unpad_data: tuple[torch.Tensor, torch.Tensor] | None,
        autocast_dtype: torch.dtype | None,
        wavenet_reflect_cache: dict | None,
    ) -> None:
        self._runner = runner
        self._entry = entry
        self._cache_key = cache_key
        self._is_ragged = is_ragged
        self._prompt_x = prompt_x
        self._x_lens = x_lens
        self._style = style
        self._precomputed_cond = precomputed_cond
        self._pre_mask = pre_mask
        self._unpad_data = unpad_data
        self._autocast_dtype = autocast_dtype
        self._wavenet_reflect_cache = wavenet_reflect_cache
        self._step = 0
        self._failed = False

    def _run_eager(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        kw: dict[str, object] = {
            "pre_mask": self._pre_mask,
            "unpad_data": self._unpad_data,
            "precomputed_cond": self._precomputed_cond,
            "_skip_inner_graph": True,
        }
        if self._wavenet_reflect_cache is not None:
            kw["wavenet_reflect_cache"] = self._wavenet_reflect_cache
        if self._autocast_dtype is None:
            return self._runner.estimator(x, self._prompt_x, self._x_lens, t, self._style, self._precomputed_cond, **kw)
        with torch.autocast(x.device.type, dtype=self._autocast_dtype):
            out = self._runner.estimator(x, self._prompt_x, self._x_lens, t, self._style, self._precomputed_cond, **kw)
        return out.float()

    def run_step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        step = self._step
        self._step += 1
        self._runner._stats["calls"] += 1

        if self._failed:
            self._runner._stats["eager"] += 1
            return self._run_eager(x, t)

        if self._entry is not None:
            self._runner._stats["hits"] += 1
            return self._runner._replay_step(self._entry, x=x, t=t)

        # Step 0 serves as the real eager warm pass (no wasted dummy pass).
        if step == 0:
            self._runner._stats["eager"] += 1
            return self._run_eager(x, t)

        # Step 1 captures the full DiT + WaveNet step and replays it.
        entry = self._runner._capture_step(
            x=x,
            t=t,
            prompt_x=self._prompt_x,
            x_lens=self._x_lens,
            style=self._style,
            precomputed_cond=self._precomputed_cond,
            pre_mask=self._pre_mask,
            unpad_data=self._unpad_data,
            autocast_dtype=self._autocast_dtype,
            wavenet_reflect_cache=self._wavenet_reflect_cache,
            is_ragged=self._is_ragged,
            cache_key=self._cache_key,
        )
        if entry is None:
            self._failed = True
            self._runner._stats["capture_failures"] += 1
            self._runner._stats["eager"] += 1
            return self._run_eager(x, t)

        try:
            entry.graph.replay()
        except Exception as exc:
            if _is_fatal_cuda_error(exc):
                raise
            logger.warning(
                "S2Mel DiT full-step CUDA graph first replay failed (ragged=%s) for shape=%s; falling back to eager",
                self._is_ragged,
                tuple(x.shape) if hasattr(x, "shape") else type(x).__name__,
                exc_info=True,
            )
            self._runner._handle_capture_failure(
                is_ragged=self._is_ragged,
                cache_key=self._cache_key,
                oom=_is_oom_error(exc),
            )
            self._failed = True
            self._runner._stats["capture_failures"] += 1
            self._runner._stats["eager"] += 1
            return self._run_eager(x, t)

        self._runner._stats["captures"] += 1
        entry.in_use = True
        self._entry = entry
        if not self._is_ragged and self._cache_key is not None:
            self._runner._store_entry(self._cache_key, entry)
        return entry.static_out

    def close(self) -> None:
        if self._entry is not None:
            self._entry.in_use = False
            if self._is_ragged:
                self._entry = None


class CUDAGraphDiTStepRunner:
    """Capture and replay the full DiT + WaveNet + FinalLayer Euler step.

    Unlike ``CUDAGraphDiTRunner`` (which wraps only ``estimator.transformer``),
    this runner captures the entire per-step ``DiT.forward`` (including
    ``t_embedder``, ``cond_x_merge_linear``, ``transformer``, ``skip_linear``,
    ``wavenet``, ``final_layer``, and output ``float()`` cast) so there are zero
    eager kernel launches between Euler steps.

    - Full-mask shapes (explicit ``full_mask=True`` and ``unpad_data is None``):
      cached across requests in a bounded LRU; per-request static inputs
      (``prompt_x``, ``style``, ``precomputed_cond``) are copied once before
      the Euler loop under a single-active-solve, same-stream guard.
    - Packed-varlen ragged batches (``unpad_data is not None``): step 0 runs
      eager as a real warm pass, step 1 captures an ephemeral graph using the
      shared graph pool, and steps 1..N-1 replay it with zero CPU launch gaps.
    """

    _MAX_FAILED_KEYS = 64

    def __init__(self, estimator: torch.nn.Module, *, max_graphs: int = 8, enabled: bool = True) -> None:
        self.estimator = estimator
        self.max_graphs = int(max_graphs)
        self.enabled = bool(enabled)
        self.ragged_enabled = bool(enabled)
        self.full_mask_enabled = False
        self._cache: OrderedDict[tuple, _StepGraphEntry] = OrderedDict()
        self._failed_keys: OrderedDict[tuple, None] = OrderedDict()
        self._stats = {
            "calls": 0,
            "hits": 0,
            "captures": 0,
            "capture_failures": 0,
            "evictions": 0,
            "eager": 0,
        }

    def stats_snapshot(self) -> dict[str, object]:
        return {**self._stats, "cache_size": len(self._cache)}

    @staticmethod
    def _can_capture_ragged(
        estimator: torch.nn.Module,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
    ) -> bool:
        if device.type != "cuda" or autocast_dtype is None:
            return False
        transformer = getattr(estimator, "transformer", None)
        layers = getattr(transformer, "layers", None)
        if not layers:
            return False
        attn = getattr(layers[0], "attention", None)
        diff_attn = getattr(attn, "diff_attn", None)
        return hasattr(getattr(diff_attn, "attention", None), "_forward_varlen_packed")

    @staticmethod
    def _load_per_solve_inputs(
        entry: _StepGraphEntry,
        *,
        prompt_x: torch.Tensor,
        style: torch.Tensor,
        precomputed_cond: torch.Tensor,
    ) -> None:
        entry.static_prompt_x.copy_(prompt_x)
        entry.static_style.copy_(style)
        entry.static_cond.copy_(precomputed_cond)

    @staticmethod
    def _replay_step(
        entry: _StepGraphEntry,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        entry.static_x.copy_(x)
        entry.static_t.copy_(t)
        entry.graph.replay()
        return entry.static_out

    def _store_entry(self, key: tuple, entry: _StepGraphEntry) -> None:
        self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_graphs:
            # Prefer evicting an idle entry if an overlapping solve somehow holds one.
            evict_key = next((k for k, v in self._cache.items() if not v.in_use), None)
            if evict_key is None:
                break
            evicted = self._cache.pop(evict_key)
            del evicted
            self._stats["evictions"] += 1

    def _handle_capture_failure(
        self,
        *,
        is_ragged: bool,
        cache_key: tuple | None,
        oom: bool,
    ) -> None:
        if is_ragged:
            if not oom:
                self.ragged_enabled = False
            return
        if oom and cache_key is not None:
            self._failed_keys[cache_key] = None
            self._failed_keys.move_to_end(cache_key)
            while len(self._failed_keys) > self._MAX_FAILED_KEYS:
                self._failed_keys.popitem(last=False)
            return
        self.enabled = False
        # Release cached graphs/static buffers when retiring the runner so they
        # are not stranded in GPU memory.
        for key in [k for k, v in self._cache.items() if not v.in_use]:
            del self._cache[key]

    def begin_solve(
        self,
        *,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        style: torch.Tensor,
        precomputed_cond: torch.Tensor,
        pre_mask: tuple[torch.Tensor, torch.Tensor | None] | None,
        unpad_data: tuple[torch.Tensor, torch.Tensor] | None,
        autocast_dtype: torch.dtype | None,
        num_steps: int,
        cfg_enabled: bool = True,
        full_mask: bool | None = None,
        wavenet_reflect_cache: dict | None = None,
    ) -> _SolveGraphSession | None:
        if (
            not self.enabled
            or getattr(self.estimator, "training", False)
            or prompt_x.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
            or num_steps < 2
        ):
            return None

        is_ragged = unpad_data is not None
        if is_ragged:
            if not self.ragged_enabled or num_steps < 4:
                return None
            if not self._can_capture_ragged(self.estimator, prompt_x.device, autocast_dtype):
                return None
            return _SolveGraphSession(
                self,
                entry=None,
                cache_key=None,
                is_ragged=True,
                prompt_x=prompt_x,
                x_lens=x_lens,
                style=style,
                precomputed_cond=precomputed_cond,
                pre_mask=pre_mask,
                unpad_data=unpad_data,
                autocast_dtype=autocast_dtype,
                wavenet_reflect_cache=wavenet_reflect_cache,
            )

        # Dense full-mask LRU requires an explicit caller guarantee (either via
        # `full_mask_enabled=True` set by the decoder or `full_mask=True`) that
        # every row is full-length so the all-True mask is identical across
        # requests sharing the same shape.
        is_full_mask = self.full_mask_enabled if full_mask is None else bool(full_mask)
        if not is_full_mask:
            return None

        device_key = (prompt_x.device.type, prompt_x.device.index)
        cache_key = (
            device_key,
            tuple(prompt_x.shape),
            prompt_x.dtype,
            tuple(style.shape),
            style.dtype,
            tuple(precomputed_cond.shape),
            precomputed_cond.dtype,
            autocast_dtype,
            bool(cfg_enabled),
        )
        if cache_key in self._failed_keys:
            return None

        entry = self._cache.get(cache_key)
        if entry is not None:
            current_stream = torch.cuda.current_stream(prompt_x.device)
            if entry.in_use or (entry.stream is not None and entry.stream != current_stream):
                return None
            self._cache.move_to_end(cache_key)
            entry.in_use = True
            self._load_per_solve_inputs(
                entry,
                prompt_x=prompt_x,
                style=style,
                precomputed_cond=precomputed_cond,
            )
        return _SolveGraphSession(
            self,
            entry=entry,
            cache_key=cache_key,
            is_ragged=False,
            prompt_x=prompt_x,
            x_lens=x_lens,
            style=style,
            precomputed_cond=precomputed_cond,
            pre_mask=pre_mask,
            unpad_data=None,
            autocast_dtype=autocast_dtype,
            wavenet_reflect_cache=None,
        )

    def _capture_step(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        style: torch.Tensor,
        precomputed_cond: torch.Tensor,
        pre_mask: tuple[torch.Tensor, torch.Tensor | None] | None,
        unpad_data: tuple[torch.Tensor, torch.Tensor] | None,
        autocast_dtype: torch.dtype | None,
        wavenet_reflect_cache: dict | None,
        is_ragged: bool,
        cache_key: tuple | None = None,
    ) -> _StepGraphEntry | None:
        orig_stream = torch.cuda.current_stream(x.device)
        try:
            static_x = x.clone()
            static_t = t.clone()
            static_prompt_x = prompt_x.clone()
            static_style = style.clone()
            static_cond = precomputed_cond.clone()
            static_x_lens = x_lens.clone()
            # Clone only the compact [2B, 1, T] mask and build the [2B, 1, T, T]
            # expanded tensor as a zero-copy view (avoids O(B*T^2) memory).
            static_pre_mask: tuple[torch.Tensor, torch.Tensor | None] | None = None
            if pre_mask is not None:
                static_x_mask = pre_mask[0].contiguous().clone()
                static_x_mask_exp = (
                    static_x_mask[:, None, :].expand(-1, -1, static_x_mask.shape[-1], -1)
                    if pre_mask[1] is not None
                    else None
                )
                static_pre_mask = (static_x_mask, static_x_mask_exp)
            static_unpad = (
                (unpad_data[0].contiguous().clone(), unpad_data[1].contiguous().clone())
                if unpad_data is not None
                else None
            )
            kw: dict[str, object] = {
                "pre_mask": static_pre_mask,
                "unpad_data": static_unpad,
                "precomputed_cond": static_cond,
                "_skip_inner_graph": True,
            }
            if is_ragged and wavenet_reflect_cache is not None:
                kw["wavenet_reflect_cache"] = wavenet_reflect_cache

            # Step 0 already executed an eager warm pass on the current stream;
            # wait for it to finish before starting stream capture.
            torch.accelerator.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                if autocast_dtype is None:
                    static_out = self.estimator(
                        static_x,
                        static_prompt_x,
                        static_x_lens,
                        static_t,
                        static_style,
                        static_cond,
                        **kw,
                    )
                else:
                    with torch.autocast(x.device.type, dtype=autocast_dtype):
                        raw_out = self.estimator(
                            static_x,
                            static_prompt_x,
                            static_x_lens,
                            static_t,
                            static_style,
                            static_cond,
                            **kw,
                        )
                    static_out = raw_out.float()
        except Exception as exc:
            if _is_fatal_cuda_error(exc):
                raise
            oom = _is_oom_error(exc)
            logger.warning(
                "S2Mel DiT full-step CUDA graph capture failed (ragged=%s, oom=%s) for shape=%s; falling back to eager",
                is_ragged,
                oom,
                tuple(x.shape) if hasattr(x, "shape") else type(x).__name__,
                exc_info=True,
            )
            self._handle_capture_failure(
                is_ragged=is_ragged,
                cache_key=cache_key,
                oom=oom,
            )
            return None
        finally:
            # PyTorch may raise from capture_end() before restoring the caller
            # stream; always restore it so eager fallback preserves stream order.
            torch.cuda.set_stream(orig_stream)

        return _StepGraphEntry(
            graph=graph,
            static_x=static_x,
            static_t=static_t,
            static_prompt_x=static_prompt_x,
            static_style=static_style,
            static_cond=static_cond,
            static_x_lens=static_x_lens,
            static_pre_mask=static_pre_mask,
            static_unpad_data=static_unpad,
            static_out=static_out,
            static_reflect_cache=wavenet_reflect_cache if is_ragged else None,
            stream=orig_stream,
        )
