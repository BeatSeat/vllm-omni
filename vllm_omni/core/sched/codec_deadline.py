# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deadline-driven batching for a streaming codec stage (``additional_config.codec_deadline``).

A Code2Wav decode costs nearly the same for one stream as for several (H100
Whole-Euler graph replay, w50: 17.7 ms at B=1, 22.5 at B=4, 39.9 at B=16), but
with synchronous scheduling every chunk that arrives is decoded in the next
step, so 16 duplex sessions run ~1.25 rows per step. This policy holds a ready
continuation chunk while its stream can afford it and releases it together with
others, earliest deadline first.

Each stream (one Stage-2 request) gets a ledger rebuilt from this stage's own
emit times -- the transport delay to the client shifts the emit times and the
client's arrival times alike, so it cancels:

* zero-stall deadline: the end of the audio the client already holds
  (``DUPLEX_CLIENT_PREBUFFER_S`` once, for the session's first segment, then
  play on arrival);
* rate deadline: the arrival that keeps ``stream_rtf`` at ``rtf_cap`` if this
  chunk were the stream's last.

A chunk may wait until ``min(deadlines) - margin_s - T(eager_rows) - T(step)``
and never longer than ``max_hold_s``. ``T`` is an EWMA of this stage's own
step time per Code2Wav graph tier (``VLLM_OMNI_CFM_GRAPH_GRID``, else powers of
two, up to ``max_rows``); ``T(eager_rows)`` reserves the eager step that may
start just before the chunk falls due, since a synchronous step cannot be
preempted. First chunks (``chunk_seq == 0``), last chunks (``last_chunk`` /
``turn_end``; not ``tts_is_last_chunk``, which is true for every duplex unit)
and chunks without a ledger are eager: they release at once, with only the rows
due within one step. A deadline step fills its graph tier with not-yet-due rows
unless a stream is still waiting for its first chunk, which would otherwise
queue behind the wider step. A breaker stops holding for ``breaker_cooloff_s``
once more than ``breaker_late_frac`` of the last ``breaker_window`` held chunks
arrived after their zero-stall deadline.

Pure Python; the scheduler supplies the ready chunks and outputs.
"""

from __future__ import annotations

import bisect
import os
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import DUPLEX_CLIENT_PREBUFFER_S

logger = init_logger(__name__)

#: Codec frames per second of the MiniCPM-o Talker codec.
CODEC_FRAME_RATE_HZ = 25.0


def graph_step_tiers(max_rows: int) -> list[int]:
    """Code2Wav graph batch sizes up to ``max_rows`` (``VLLM_OMNI_CFM_GRAPH_GRID``, else powers of two)."""
    raw = os.environ.get("VLLM_OMNI_CFM_GRAPH_GRID")
    if raw:
        sizes = sorted({int(size) for size in raw.split(",") if size.strip()})
    else:
        sizes = [1 << power for power in range(max_rows.bit_length())]
    return sorted({size for size in sizes if 0 < size < max_rows} | {max_rows})


def seed_step_s(rows: int) -> float:
    """Initial step-time estimate (H100 d16 serving: B=1 ~0.11 s, B=4 ~0.20 s, B=8 ~0.28 s)."""
    return 0.09 + 0.024 * rows


@dataclass(frozen=True)
class CodecDeadlineConfig:
    enabled: bool = False
    margin_s: float = 0.1
    rtf_cap: float = 1.1
    max_hold_s: float = 0.35
    max_rows: int = 8
    eager_rows: int = 4
    step_alpha: float = 0.1
    gap_reanchor_s: float = 1.0
    breaker_window: int = 50
    breaker_late_frac: float = 0.1
    breaker_cooloff_s: float = 10.0
    stats: bool = False

    def __post_init__(self) -> None:
        for name in ("enabled", "stats"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"codec_deadline.{name} must be a boolean")
        for name in ("margin_s", "max_hold_s", "gap_reanchor_s", "breaker_cooloff_s"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"codec_deadline.{name} must be non-negative")
        if self.rtf_cap <= 0 or not 0 < self.step_alpha <= 1 or not 0 <= self.breaker_late_frac <= 1:
            raise ValueError("codec_deadline: rtf_cap > 0, 0 < step_alpha <= 1, 0 <= breaker_late_frac <= 1")
        if self.max_rows < 1 or self.eager_rows < 1 or self.breaker_window < 1:
            raise ValueError("codec_deadline: max_rows, eager_rows and breaker_window must be >= 1")

    @classmethod
    def from_additional_config(cls, additional_config: object) -> CodecDeadlineConfig | None:
        raw = additional_config.get("codec_deadline") if isinstance(additional_config, Mapping) else None
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("additional_config.codec_deadline must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"additional_config.codec_deadline: unknown keys {sorted(unknown)}")
        return cls(**dict(raw))


def _scalar(value: Any) -> Any:
    """A plain Python value from a connector meta field (tensor, array, list or scalar)."""
    if isinstance(value, list | tuple):
        return _scalar(value[0]) if value else None
    numel = getattr(value, "numel", None)
    if callable(numel):
        return value.reshape(-1)[0].item() if numel() else None
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "size", 1) == 1:
        return item()
    return value


@dataclass(frozen=True, slots=True)
class ChunkMeta:
    chunk_seq: int
    cache_epoch: int
    last_chunk: bool
    turn_end: bool
    codec_chunk_frames: int
    duplex_epoch: int | None

    @classmethod
    def from_info(cls, info: object) -> ChunkMeta | None:
        """The producer metadata of a request's current chunk; None when it carries none."""
        meta = info.get("meta") if isinstance(info, Mapping) else None
        if not isinstance(meta, Mapping) or meta.get("chunk_seq") is None:
            return None
        try:
            duplex_epoch = _scalar(meta.get("duplex_epoch"))
            return cls(
                chunk_seq=int(_scalar(meta.get("chunk_seq"))),
                cache_epoch=int(_scalar(meta.get("cache_epoch")) or 0),
                last_chunk=bool(_scalar(meta.get("last_chunk"))),
                turn_end=bool(_scalar(meta.get("turn_end"))),
                codec_chunk_frames=int(_scalar(meta.get("codec_chunk_frames")) or 0),
                duplex_epoch=None if duplex_epoch is None else int(duplex_epoch),
            )
        except (TypeError, ValueError):
            return None


@dataclass(slots=True)
class StreamLedger:
    key: tuple[int | None, int]
    anchor: float
    play_end: float
    audio_s: float
    d1: float
    late_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ReadyChunk:
    request_id: str
    meta: ChunkMeta | None


@dataclass
class DeadlinePlan:
    held: set[str] = field(default_factory=set)
    released: list[str] = field(default_factory=list)
    trigger: str = "none"
    #: Metadata of every ready chunk this step (filled by the scheduler).
    metas: dict[str, ChunkMeta | None] = field(default_factory=dict)


class StepTimeModel:
    """EWMA of this stage's step wall time per graph tier: mean + 1.5 x mean absolute deviation."""

    def __init__(self, tiers: Iterable[int], alpha: float) -> None:
        self.tiers = list(tiers)
        self.mean = [seed_step_s(tier) for tier in self.tiers]
        self.mad = [0.0 for _ in self.tiers]
        self.alpha = alpha

    def tier_index(self, rows: int) -> int:
        return min(bisect.bisect_left(self.tiers, max(1, rows)), len(self.tiers) - 1)

    def tier_capacity(self, rows: int) -> int:
        return self.tiers[self.tier_index(rows)]

    def estimate(self, rows: int) -> float:
        index = self.tier_index(rows)
        return self.mean[index] + 1.5 * self.mad[index]

    def observe(self, rows: int, elapsed_s: float) -> None:
        if rows <= 0 or elapsed_s <= 0:
            return
        index = self.tier_index(rows)
        error = elapsed_s - self.mean[index]
        self.mean[index] += self.alpha * error
        self.mad[index] += self.alpha * (abs(error) - self.mad[index])


class LateBreaker:
    """Stops holding for a cool-off once too many held chunks arrived late."""

    def __init__(self, window: int, late_frac: float, cooloff_s: float) -> None:
        self.outcomes: deque[bool] = deque(maxlen=window)
        self.window = window
        self.late_frac = late_frac
        self.cooloff_s = cooloff_s
        self.open_until = 0.0
        self.opens = 0

    def is_open(self, now: float) -> bool:
        return now < self.open_until

    def record(self, late: bool, now: float) -> None:
        self.outcomes.append(late)
        if sum(self.outcomes) > self.late_frac * self.window:
            self.open_until = now + self.cooloff_s
            self.opens += 1
            self.outcomes.clear()
            logger.warning(
                "Codec deadline batching: held chunks arrived late; holding off for %.1f s (opens=%d)",
                self.cooloff_s,
                self.opens,
            )


class CodecDeadlinePolicy:
    """Per-step hold/release decisions and the per-stream playback ledgers (see module docstring)."""

    def __init__(self, config: CodecDeadlineConfig) -> None:
        self.config = config
        self.ledgers: dict[str, StreamLedger] = {}
        #: Streams whose client already spent its jitter buffer (once per session).
        self.buffered: set[str] = set()
        self.step_time = StepTimeModel(graph_step_tiers(config.max_rows), config.step_alpha)
        self.breaker = LateBreaker(config.breaker_window, config.breaker_late_frac, config.breaker_cooloff_s)
        self.ready_since: dict[str, float] = {}
        #: Requests whose current chunk was held at least once.
        self.held_once: set[str] = set()
        #: Chunk metadata, held flag and trigger at dispatch, consumed by ``on_output``.
        self.dispatched: dict[str, tuple[ChunkMeta | None, bool, str]] = {}
        self._last_trigger = "none"
        self.next_release_at: float | None = None
        self.last_schedule_t: float | None = None
        self._stats = _DeadlineStats() if config.stats else None

    # ---- planning ------------------------------------------------------ #

    def _eager(self, chunk: ReadyChunk, now: float) -> bool:
        meta = chunk.meta
        if self.breaker.is_open(now) or meta is None:
            return True
        if meta.chunk_seq == 0 or meta.last_chunk or meta.turn_end or meta.codec_chunk_frames <= 0:
            return True
        if meta.duplex_epoch is None:
            return True
        ledger = self.ledgers.get(chunk.request_id)
        if ledger is None or ledger.key != (meta.duplex_epoch, meta.cache_epoch):
            return True
        return now - ledger.play_end > self.config.gap_reanchor_s

    def release_by(self, chunk: ReadyChunk, now: float, ready_rows: int) -> float:
        """Latest dispatch time of a held chunk (``r_eff``)."""
        meta = chunk.meta
        ledger = self.ledgers[chunk.request_id]
        assert meta is not None
        chunk_s = meta.codec_chunk_frames / CODEC_FRAME_RATE_HZ
        rate_deadline = ledger.anchor + self.config.rtf_cap * (ledger.audio_s + chunk_s - ledger.d1)
        deadline = min(ledger.play_end, rate_deadline)
        # Its own step, and an eager step that may start just before it falls due.
        steps_s = self.step_time.estimate(ready_rows) + self.step_time.estimate(self.config.eager_rows)
        release = deadline - self.config.margin_s - steps_s
        since = self.ready_since.setdefault(chunk.request_id, now)
        return min(release, since + self.config.max_hold_s)

    def plan(self, ready: list[ReadyChunk], now: float, *, onset_pending: bool = False) -> DeadlinePlan:
        """This step's held and released chunks.

        ``onset_pending``: a stream is waiting for its first chunk; a deadline
        step then releases only what is due instead of filling its graph tier.
        """
        config = self.config
        self.last_schedule_t = now
        self.next_release_at = None
        ready_ids = {chunk.request_id for chunk in ready}
        for request_id in [rid for rid in self.ready_since if rid not in ready_ids]:
            del self.ready_since[request_id]
        if not ready:
            return DeadlinePlan()
        for chunk in ready:
            self.ready_since.setdefault(chunk.request_id, now)
        eager = [chunk.request_id for chunk in ready if self._eager(chunk, now)]
        eager_set = set(eager)
        release_by = {
            chunk.request_id: self.release_by(chunk, now, len(ready))
            for chunk in ready
            if chunk.request_id not in eager_set
        }
        paced = sorted(release_by, key=release_by.__getitem__)
        one_step = self.step_time.estimate(1)
        due = [rid for rid in paced if release_by[rid] <= now]
        soon = [rid for rid in paced if now < release_by[rid] <= now + one_step]
        if eager:
            trigger = "eager"
            # Protect the first chunk: only rows that are due anyway ride along.
            selected = eager + due + soon[: max(0, config.eager_rows - len(eager) - len(due))]
        elif due:
            trigger = "deadline"
            selected = due + soon
            if not onset_pending:
                target = min(config.max_rows, self.step_time.tier_capacity(len(selected)))
                selected += [rid for rid in paced if rid not in selected][: max(0, target - len(selected))]
        elif len(ready) >= config.max_rows:
            trigger = "rows"
            selected = paced[: config.max_rows]
        else:
            self.next_release_at = release_by[paced[0]]
            plan = DeadlinePlan(held=set(ready_ids), trigger="hold")
            self._note_plan(plan)
            return plan
        selected = selected[: max(config.max_rows, len(eager) + len(due))]
        held = ready_ids - set(selected)
        if held:
            self.next_release_at = min(release_by[rid] for rid in held if rid in release_by)
        plan = DeadlinePlan(held=held, released=selected, trigger=trigger)
        self._note_plan(plan)
        return plan

    def _note_plan(self, plan: DeadlinePlan) -> None:
        self.held_once.update(plan.held)
        self._last_trigger = plan.trigger
        if self._stats is not None:
            self._stats.record_plan(plan)

    def on_scheduled(self, request_id: str, meta: ChunkMeta | None, now: float) -> None:
        """The scheduler dispatched this request's chunk."""
        held = request_id in self.held_once
        self.held_once.discard(request_id)
        since = self.ready_since.pop(request_id, None)
        self.dispatched[request_id] = (meta, held, self._last_trigger)
        if held and since is not None and self._stats is not None:
            self._stats.holds_s.append(now - since)

    def next_release_in(self, now: float) -> float | None:
        if self.next_release_at is None:
            return None
        return max(0.0, self.next_release_at - now)

    # ---- outputs ------------------------------------------------------- #

    def on_output(self, request_id: str, audio_s: float | None, now: float) -> None:
        meta, held, trigger = self.dispatched.pop(request_id, (None, False, "none"))
        if meta is None:
            return
        if audio_s:
            key = (meta.duplex_epoch, meta.cache_epoch)
            ledger = self.ledgers.get(request_id)
            if (
                ledger is None
                or meta.chunk_seq == 0
                or ledger.key != key
                or now - ledger.play_end > self.config.gap_reanchor_s
            ):
                # The client buffers once per session: its first segment only.
                first = request_id not in self.buffered and meta.duplex_epoch == 0 and meta.cache_epoch == 0
                buffer_s = DUPLEX_CLIENT_PREBUFFER_S if first else 0.0
                self.buffered.add(request_id)
                self.ledgers[request_id] = StreamLedger(
                    key=key, anchor=now, play_end=now + buffer_s + audio_s, audio_s=audio_s, d1=audio_s
                )
            else:
                late = now > ledger.play_end
                if late:
                    ledger.late_s += now - ledger.play_end
                    ledger.play_end = now
                if held:
                    self.breaker.record(late, now)
                    if self._stats is not None and late:
                        self._stats.late[trigger] = self._stats.late.get(trigger, 0) + 1
                ledger.play_end += audio_s
                ledger.audio_s += audio_s
        if meta.last_chunk or meta.turn_end:
            # The next turn re-anchors (without buffering again).
            self.ledgers.pop(request_id, None)

    def observe_step(self, rows: int, now: float) -> None:
        if self.last_schedule_t is not None and rows > 0:
            self.step_time.observe(rows, now - self.last_schedule_t)
        if self._stats is not None:
            self._stats.maybe_log(self)

    def forget(self, request_id: str) -> None:
        self.ledgers.pop(request_id, None)
        self.buffered.discard(request_id)
        self.ready_since.pop(request_id, None)
        self.held_once.discard(request_id)
        self.dispatched.pop(request_id, None)


class _DeadlineStats:
    def __init__(self, every: int = 200) -> None:
        self.every = every
        self._reset()

    def _reset(self) -> None:
        self.steps = 0
        self.rows: dict[int, int] = {}
        self.triggers: dict[str, int] = {}
        self.holds_s: list[float] = []
        #: Held chunks that still arrived late, by the trigger of the step that released them.
        self.late: dict[str, int] = {}

    def record_plan(self, plan: DeadlinePlan) -> None:
        self.steps += 1
        rows = len(plan.released)
        self.rows[rows] = self.rows.get(rows, 0) + 1
        self.triggers[plan.trigger] = self.triggers.get(plan.trigger, 0) + 1

    def maybe_log(self, policy: CodecDeadlinePolicy) -> None:
        if self.steps < self.every:
            return
        holds = sorted(self.holds_s)
        released = sum(rows * count for rows, count in self.rows.items())
        busy = sum(count for rows, count in self.rows.items() if rows)
        logger.info(
            "Codec deadline batching: %d planned steps rows=%s rows/step %.2f triggers=%s "
            "held chunks %d hold p50 %.0f ms p90 %.0f ms late=%s breaker opens %d step_s=%s",
            self.steps,
            dict(sorted(self.rows.items())),
            released / busy if busy else 0.0,
            dict(sorted(self.triggers.items())),
            len(holds),
            1000 * holds[len(holds) // 2] if holds else 0.0,
            1000 * holds[min(len(holds) - 1, int(len(holds) * 0.9))] if holds else 0.0,
            dict(sorted(self.late.items())),
            policy.breaker.opens,
            [round(policy.step_time.estimate(tier), 3) for tier in policy.step_time.tiers],
        )
        self._reset()


def build_codec_deadline_policy(
    additional_config: object,
    *,
    async_scheduling: bool,
    native_data_plane: bool,
    idle_wait_s: float,
) -> CodecDeadlinePolicy | None:
    """The policy for a stage, or None (with the reason logged) when it is off or cannot run."""
    config = CodecDeadlineConfig.from_additional_config(additional_config)
    if config is None or not config.enabled:
        return None
    reason = None
    if async_scheduling:
        reason = "async_scheduling is on (the batch queue would delay each step's output by one step)"
    elif native_data_plane:
        reason = "the native MRv2 data plane is not supported"
    elif idle_wait_s <= 0:
        reason = "VLLM_OMNI_STAGE_IDLE_WAIT_S is 0 (a hold needs the event-driven idle park)"
    if reason is not None:
        logger.warning("Codec deadline batching disabled: %s", reason)
        return None
    policy = CodecDeadlinePolicy(config)
    logger.info(
        "Codec deadline batching enabled: margin %.2f s, max hold %.2f s, rtf cap %.2f, max rows %d, step tiers %s",
        config.margin_s,
        config.max_hold_s,
        config.rtf_cap,
        config.max_rows,
        policy.step_time.tiers,
    )
    return policy
