# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in GC policy for duplex API servers (``VLLM_OMNI_DUPLEX_GC``).

Once warm-up is done, freeze the heap and raise the collection thresholds, so
later collections neither walk the long-lived startup objects nor run as often.
"""

from __future__ import annotations

import gc
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_GC_POLICY_ENV = "VLLM_OMNI_DUPLEX_GC"
_DUPLEX_GC_THRESHOLDS = (50_000, 20, 20)


def apply_duplex_gc_policy() -> None:
    """Freeze the heap and raise the GC thresholds, when ``VLLM_OMNI_DUPLEX_GC`` asks for it.

    ``1`` (or ``true``/``yes``/``on``) uses ``_DUPLEX_GC_THRESHOLDS``; three
    comma-separated integers set the gen0, gen1 and gen2 thresholds instead.
    """
    raw = os.environ.get(_GC_POLICY_ENV, "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return
    thresholds = _DUPLEX_GC_THRESHOLDS
    if raw not in ("1", "true", "yes", "on"):
        try:
            parsed = tuple(int(part) for part in raw.split(","))
        except ValueError:
            parsed = ()
        if len(parsed) != 3 or min(parsed) < 0:
            logger.warning("Ignoring %s=%r: expected 1, 0 or three thresholds gen0,gen1,gen2", _GC_POLICY_ENV, raw)
            return
        thresholds = parsed
    from vllm.utils.gc_utils import freeze_gc_heap

    previous = gc.get_threshold()
    freeze_gc_heap()
    gc.set_threshold(*thresholds)
    logger.info(
        "Duplex GC policy (%s): %d objects frozen; thresholds %s -> %s",
        _GC_POLICY_ENV,
        gc.get_freeze_count(),
        previous,
        gc.get_threshold(),
    )
