# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``duplex_incremental_fbank`` is bitwise equal to the remote ``StreamingMelProcessorExact._extract_full``.

The reference is the remote code of openbmb/MiniCPM-o-4_5 in the transformers
dynamic-module cache; without it the bitwise tests skip.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.duplex.incremental_fbank as incremental_fbank
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.incremental_fbank import enable_incremental_fbank
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SR, _HOP = 16000, 160


@pytest.fixture(scope="module")
def remote() -> ModuleType:
    from transformers.utils import HF_MODULES_CACHE

    candidates = [
        path
        for path in Path(HF_MODULES_CACHE, "transformers_modules").glob("*/*/processing_minicpmo.py")
        if "class StreamingMelProcessorExact" in path.read_text()
    ]
    if not candidates:
        pytest.skip("MiniCPM-o 4.5 remote processing code is not in the transformers module cache")
    from transformers.dynamic_module_utils import init_hf_modules

    init_hf_modules()
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return importlib.import_module(".".join(path.relative_to(HF_MODULES_CACHE).with_suffix("").parts))


def _mel(remote: ModuleType, *, incremental: bool) -> Any:
    mel = remote.StreamingMelProcessorExact(  # Stage 0's streaming configuration
        feature_extractor=remote.MiniCPMAAudioProcessor(),
        chunk_ms=1000,
        first_chunk_ms=1035,
        cnn_redundancy_ms=20,
        enable_sliding_window=True,
        slide_trigger_seconds=30.0,
        slide_stride_seconds=10.0,
    )
    assert not incremental or enable_incremental_fbank(mel)
    return mel


@pytest.mark.parametrize("kind", ["below_5s", "above_5s", "slide"])
def test_random_buffers_are_bitwise_equal(remote, kind: str) -> None:
    # A cached buffer, then a longer one (slid by a whole number of hops for "slide").
    rng = np.random.default_rng({"below_5s": 1, "above_5s": 2, "slide": 3}[kind])
    inc = _mel(remote, incremental=True)
    for _ in range(40):
        total = int(rng.integers(2_000, 5 * _SR) if kind == "below_5s" else rng.integers(5 * _SR, 640_000))
        audio = (rng.standard_normal(total) * rng.choice([1e-4, 0.01, 0.1, 0.9])).astype(np.float32)
        if rng.random() < 0.3:
            audio[: int(rng.integers(0, total))] = 0.0
        prev_end = int(rng.integers(400, min(total, 480_000) + 1))
        inc.reset()
        inc.buffer, inc.left_samples_dropped = audio[:prev_end], 0
        inc._extract_full()
        dropped = int(rng.integers(1, max(2, (prev_end - 400) // _HOP + 1))) * _HOP if kind == "slide" else 0
        dropped = min(dropped, (prev_end - 400) // _HOP * _HOP)
        start = max(prev_end, 5 * _SR) if kind == "above_5s" else prev_end
        end = int(rng.integers(min(start, total), min(total, dropped + 480_000) + 1))
        inc.buffer, inc.left_samples_dropped = audio[dropped:end], dropped
        got = inc._extract_full()
        assert torch.equal(got, remote.StreamingMelProcessorExact._extract_full(inc)), (prev_end, dropped, end)


def test_streams_are_bitwise_equal(remote) -> None:
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(66 * _SR) * 0.05).astype(np.float32)
    audio[_SR * 3 : _SR * 7] *= 20  # loud stretch for the dynamic range
    audio[_SR * 12 : _SR * 14] = 0.0  # digital silence
    ref, inc = _mel(remote, incremental=False), _mel(remote, incremental=True)
    position, slides = 0, 0
    while position < len(audio):
        size = ref.get_chunk_size()
        dropped = ref.left_samples_dropped
        expected, expected_info = ref.process(audio[position : position + size])
        got, got_info = inc.process(audio[position : position + size])
        assert torch.equal(got, expected) and got_info == expected_info
        position += size
        slides += ref.left_samples_dropped != dropped
    assert slides >= 2


def test_self_check_rejects_a_different_original(remote) -> None:
    class ShiftedMel(remote.StreamingMelProcessorExact):
        def _extract_full(self):
            return super()._extract_full() + 1e-6

    mel = ShiftedMel(feature_extractor=remote.MiniCPMAAudioProcessor(), chunk_ms=1000)
    assert enable_incremental_fbank(mel) is False and type(mel) is ShiftedMel
    assert enable_incremental_fbank(SimpleNamespace(buffer=np.zeros(0, dtype=np.float32))) is False


@pytest.mark.parametrize("value", [True, False, "true"])
def test_runtime_enables_it_only_when_switched_on(monkeypatch, value) -> None:
    enable = MagicMock(return_value=True)
    monkeypatch.setattr(incremental_fbank, "enable_incremental_fbank", enable)

    class Processor:
        _streaming_mel_processor = None

        def set_streaming_mode(self, **kwargs) -> None:
            self._streaming_mel_processor = SimpleNamespace(**kwargs)

    stage_model = SimpleNamespace(config=SimpleNamespace(duplex_incremental_fbank=value), processor=Processor())
    runtime = MiniCPMO45Stage0DuplexRuntime(stage_model, device="cpu")
    processor = runtime._configure_streaming_processor(SimpleNamespace(streaming_processor=None))
    assert enable.call_args_list == ([((processor._streaming_mel_processor,),)] if value is True else [])
