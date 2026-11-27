# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.indextts2.dit_cuda_graph import (
    CUDAGraphDiTRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CopySpy:
    def __init__(self):
        self.sources = []

    def copy_(self, source):
        self.sources.append(source)
        return self


class _GraphSpy:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def test_replay_only_updates_dynamic_inputs_and_reuses_static_output():
    runner = CUDAGraphDiTRunner(transformer=object())
    graph = _GraphSpy()
    static_x = _CopySpy()
    static_c = _CopySpy()
    static_pos = _CopySpy()
    static_mask = _CopySpy()
    static_out = object()
    entry = SimpleNamespace(
        graph=graph,
        static_x=static_x,
        static_c=static_c,
        static_pos=static_pos,
        static_mask=static_mask,
        static_out=static_out,
    )
    x_in = object()
    c = object()

    output = runner._replay(entry, x_in=x_in, c=c)

    assert static_x.sources == [x_in]
    assert static_c.sources == [c]
    assert static_pos.sources == []
    assert static_mask.sources == []
    assert graph.replays == 1
    assert output is static_out


def test_graph_stats_start_empty_and_remain_bounded():
    runner = CUDAGraphDiTRunner(transformer=object(), max_graphs=4)

    stats = runner.stats_snapshot()
    stats["hits"] = 99

    assert runner.stats_snapshot() == {
        "calls": 0,
        "hits": 0,
        "captures": 0,
        "capture_failures": 0,
        "evictions": 0,
        "eager": 0,
        "cache_size": 0,
    }


def test_step_runner_loads_per_solve_inputs_once_and_replays_dynamic_x_t():
    from vllm_omni.model_executor.models.indextts2.dit_cuda_graph import (
        CUDAGraphDiTStepRunner,
        _SolveGraphSession,
    )

    runner = CUDAGraphDiTStepRunner(estimator=object(), max_graphs=4)
    graph = _GraphSpy()
    static_x = _CopySpy()
    static_t = _CopySpy()
    static_prompt_x = _CopySpy()
    static_style = _CopySpy()
    static_cond = _CopySpy()
    static_out = object()
    entry = SimpleNamespace(
        graph=graph,
        static_x=static_x,
        static_t=static_t,
        static_prompt_x=static_prompt_x,
        static_style=static_style,
        static_cond=static_cond,
        static_out=static_out,
    )

    prompt_x, style, cond = object(), object(), object()
    runner._load_per_solve_inputs(entry, prompt_x=prompt_x, style=style, precomputed_cond=cond)
    session = _SolveGraphSession(
        runner,
        entry=entry,
        cache_key=("k",),
        is_ragged=False,
        prompt_x=prompt_x,
        x_lens=object(),
        style=style,
        precomputed_cond=cond,
        pre_mask=None,
        unpad_data=None,
        autocast_dtype=None,
        wavenet_reflect_cache=None,
    )
    x0, t0 = object(), object()
    x1, t1 = object(), object()
    assert session.run_step(x0, t0) is static_out
    assert session.run_step(x1, t1) is static_out
    assert static_prompt_x.sources == [prompt_x]
    assert static_style.sources == [style]
    assert static_cond.sources == [cond]
    assert static_x.sources == [x0, x1]
    assert static_t.sources == [t0, t1]
    assert graph.replays == 2
    assert runner.stats_snapshot()["hits"] == 2


def test_step_runner_step0_warms_eager_then_captures_step1_and_falls_back_on_error():
    from vllm_omni.model_executor.models.indextts2.dit_cuda_graph import (
        CUDAGraphDiTStepRunner,
        _SolveGraphSession,
    )

    eager_calls = []

    def _fake_estimator(x, prompt_x, x_lens, t, style, cond, **kw):
        eager_calls.append((x, t, kw.get("_skip_inner_graph")))
        return f"eager-{x}"

    runner = CUDAGraphDiTStepRunner(estimator=_fake_estimator, max_graphs=4)
    capture_attempts = []

    def _fail_capture(**_kw):
        capture_attempts.append(1)
        return None

    runner._capture_step = _fail_capture
    session = _SolveGraphSession(
        runner,
        entry=None,
        cache_key=("k",),
        is_ragged=True,
        prompt_x=object(),
        x_lens=object(),
        style=object(),
        precomputed_cond=object(),
        pre_mask=None,
        unpad_data=None,
        autocast_dtype=None,
        wavenet_reflect_cache={},
    )
    assert session.run_step("x0", "t0") == "eager-x0"
    assert capture_attempts == []
    assert session.run_step("x1", "t1") == "eager-x1"
    assert len(capture_attempts) == 1
    assert session.run_step("x2", "t2") == "eager-x2"
    # After capture failure, no further capture attempts are made in this solve.
    assert len(capture_attempts) == 1
    assert all(skip is True for _, _, skip in eager_calls)
    session.close()


def test_step_runner_guards_in_use_training_full_mask_replay_failure_and_oom():
    from vllm_omni.model_executor.models.indextts2.dit_cuda_graph import (
        CUDAGraphDiTStepRunner,
        _SolveGraphSession,
    )

    class _DummyEstimator:
        training = False

    est = _DummyEstimator()
    runner = CUDAGraphDiTStepRunner(estimator=est, max_graphs=2)

    # First replay failure must NOT publish broken entry into LRU.
    class _FailReplayGraph:
        def replay(self):
            raise RuntimeError("replay boom")

    broken_entry = SimpleNamespace(graph=_FailReplayGraph(), static_out="bad", in_use=False)
    runner._capture_step = lambda **_kw: broken_entry
    runner.estimator = lambda x, *_args, **_kw: f"eager-fallback-{x}"
    session = _SolveGraphSession(
        runner,
        entry=None,
        cache_key=("shape-A",),
        is_ragged=False,
        prompt_x=object(),
        x_lens=object(),
        style=object(),
        precomputed_cond=object(),
        pre_mask=None,
        unpad_data=None,
        autocast_dtype=None,
        wavenet_reflect_cache=None,
    )
    assert session.run_step("x0", "t0") == "eager-fallback-x0"
    assert session.run_step("x1", "t1") == "eager-fallback-x1"
    assert runner.stats_snapshot()["cache_size"] == 0
    assert runner.enabled is False
    session.close()

    # OOM failure blacklists only the failing shape and keeps existing LRU entries alive.
    runner2 = CUDAGraphDiTStepRunner(estimator=est, max_graphs=4)
    keep_entry = SimpleNamespace(graph=_GraphSpy(), in_use=False)
    runner2._cache[("small",)] = keep_entry
    runner2._handle_capture_failure(is_ragged=False, cache_key=("huge",), oom=True)
    assert runner2.enabled is True
    assert ("small",) in runner2._cache
    assert ("huge",) in runner2._failed_keys

    # Non-OOM failure disables runner and releases idle cached entries.
    runner2._handle_capture_failure(is_ragged=False, cache_key=("small",), oom=False)
    assert runner2.enabled is False
    assert len(runner2._cache) == 0


def test_step_runner_begin_solve_gates_full_mask_training_in_use_and_stream(monkeypatch):
    import torch

    from vllm_omni.model_executor.models.indextts2.dit_cuda_graph import (
        CUDAGraphDiTStepRunner,
        _StepGraphEntry,
    )

    class _FakeCudaTensor:
        def __init__(self, shape, dtype="bf16"):
            self.shape = shape
            self.dtype = dtype
            self.device = SimpleNamespace(type="cuda", index=0)

    class _DummyEstimator:
        training = False

    stream_a = object()
    stream_b = object()
    current_stream = {"s": stream_a}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev=None: current_stream["s"])

    est = _DummyEstimator()
    runner = CUDAGraphDiTStepRunner(estimator=est, max_graphs=2)
    prompt_x = _FakeCudaTensor((2, 80, 16))
    x_lens = _FakeCudaTensor((1,), "int64")
    style = _FakeCudaTensor((2, 192))
    cond = _FakeCudaTensor((2, 16, 512))

    # Default full_mask_enabled=False rejects dense full-mask LRU solves.
    assert (
        runner.begin_solve(
            prompt_x=prompt_x,
            x_lens=x_lens,
            style=style,
            precomputed_cond=cond,
            pre_mask=None,
            unpad_data=None,
            autocast_dtype=None,
            num_steps=4,
        )
        is None
    )

    # Training mode rejects solves even when full_mask_enabled=True.
    runner.full_mask_enabled = True
    est.training = True
    assert (
        runner.begin_solve(
            prompt_x=prompt_x,
            x_lens=x_lens,
            style=style,
            precomputed_cond=cond,
            pre_mask=None,
            unpad_data=None,
            autocast_dtype=None,
            num_steps=4,
        )
        is None
    )
    est.training = False

    # Cached entry rejects overlapping in_use and cross-stream reuse.
    cache_key = (
        ("cuda", 0),
        (2, 80, 16),
        "bf16",
        (2, 192),
        "bf16",
        (2, 16, 512),
        "bf16",
        None,
        True,
    )
    entry = _StepGraphEntry(
        graph=_GraphSpy(),
        static_x=_CopySpy(),
        static_t=_CopySpy(),
        static_prompt_x=_CopySpy(),
        static_style=_CopySpy(),
        static_cond=_CopySpy(),
        static_x_lens=_CopySpy(),
        static_pre_mask=None,
        static_unpad_data=None,
        static_out="out",
        stream=stream_a,
    )
    runner._cache[cache_key] = entry

    # Cross-stream call is rejected.
    current_stream["s"] = stream_b
    assert (
        runner.begin_solve(
            prompt_x=prompt_x,
            x_lens=x_lens,
            style=style,
            precomputed_cond=cond,
            pre_mask=None,
            unpad_data=None,
            autocast_dtype=None,
            num_steps=4,
        )
        is None
    )

    # Same-stream call acquires entry and marks in_use=True; overlapping call is rejected.
    current_stream["s"] = stream_a
    session = runner.begin_solve(
        prompt_x=prompt_x,
        x_lens=x_lens,
        style=style,
        precomputed_cond=cond,
        pre_mask=None,
        unpad_data=None,
        autocast_dtype=None,
        num_steps=4,
    )
    assert session is not None
    assert entry.in_use is True
    assert (
        runner.begin_solve(
            prompt_x=prompt_x,
            x_lens=x_lens,
            style=style,
            precomputed_cond=cond,
            pre_mask=None,
            unpad_data=None,
            autocast_dtype=None,
            num_steps=4,
        )
        is None
    )
    session.close()
    assert entry.in_use is False
