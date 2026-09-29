# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import gc

import pytest

from vllm_omni.utils import duplex_gc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("env", "thresholds"),
    [
        (None, None),
        ("1", (50_000, 20, 20)),
        ("900,5,7", (900, 5, 7)),
        ("junk", None),
        ("1,2", None),
        ("10,-1,10", None),
    ],
)
def test_duplex_gc_policy_env(
    monkeypatch: pytest.MonkeyPatch, env: str | None, thresholds: tuple[int, int, int] | None
) -> None:
    import vllm.utils.gc_utils as gc_utils

    frozen: list[bool] = []
    monkeypatch.setattr(gc_utils, "freeze_gc_heap", lambda: frozen.append(True))
    if env is None:
        monkeypatch.delenv("VLLM_OMNI_DUPLEX_GC", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_DUPLEX_GC", env)
    previous = gc.get_threshold()
    try:
        duplex_gc.apply_duplex_gc_policy()
        assert gc.get_threshold() == (thresholds or previous)
        assert frozen == ([True] if thresholds else [])
    finally:
        gc.set_threshold(*previous)
