# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.sample.metadata import SamplingMetadata


@dataclass(frozen=True, slots=True)
class DuplexSamplingRow:
    """Request-local context for the experimental duplex sampling hook."""

    row_idx: int
    request_id: str
    session_id: str | None
    seq: int | None
    payload: dict[str, object] | None
    max_tokens: int | None
    # Host copies of the row's sampling parameters (None when the runner has none).
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None


def _host_value(values: Any, row_idx: int, cast: type) -> Any:
    """``values[row_idx]`` of a host array as ``cast``, or None when there is none."""
    if values is None:
        return None
    try:
        return cast(values[row_idx])
    except (IndexError, TypeError, ValueError):
        return None


class DuplexSamplingHelper:
    """Tracks duplex sampling rows outside the generic AR model runner."""

    def __init__(self) -> None:
        self.active_request_ids: set[str] = set()
        self.hook_active = False

    @staticmethod
    def _is_duplex_data_plane_info(info: object) -> bool:
        duplex = info.get("duplex") if isinstance(info, dict) else None
        return isinstance(duplex, dict) and duplex.get("data_plane") is True

    @staticmethod
    def _request_intermediate_info(runner: object, req_id: str) -> dict[str, Any] | None:
        model_intermediate_buffer = getattr(runner, "model_intermediate_buffer", {})
        info = model_intermediate_buffer.get(req_id) if isinstance(model_intermediate_buffer, dict) else None
        if isinstance(info, dict):
            return info
        model_state = getattr(runner, "model_state", None)
        if model_state is not None and hasattr(model_state, "intermediate_buffer"):
            ib = model_state.intermediate_buffer
            idx = getattr(ib, "req_id_to_index", {}).get(req_id)
            if idx is not None and idx < len(getattr(ib, "buffers", [])):
                return ib.buffers[idx]
        requests = getattr(runner, "requests", {})
        req_state = requests.get(req_id) if isinstance(requests, dict) else None
        info = getattr(req_state, "additional_information_cpu", None)
        return info if isinstance(info, dict) else None

    def refresh_active_request(self, runner: object, req_id: str) -> None:
        if self._is_duplex_data_plane_info(self._request_intermediate_info(runner, req_id)):
            self.active_request_ids.add(req_id)
        else:
            self.active_request_ids.discard(req_id)

    def update_states(self, runner: Any, scheduler_output: Any) -> None:
        self.active_request_ids.difference_update(str(req_id) for req_id in scheduler_output.finished_req_ids)
        for request in scheduler_output.scheduled_new_reqs:
            self.refresh_active_request(runner, str(request.req_id))

    def rows(self, runner: Any) -> tuple[DuplexSamplingRow, ...]:
        rows: list[DuplexSamplingRow] = []
        input_batch = getattr(runner, "input_batch", None)
        req_ids = [str(req_id) for req_id in getattr(input_batch, "req_ids", [])]
        requests = getattr(runner, "requests", {})
        req_states = getattr(runner, "req_states", None)
        temperature_cpu = getattr(input_batch, "temperature_cpu", None)
        top_k_cpu = getattr(input_batch, "top_k_cpu", None)
        top_p_cpu = getattr(input_batch, "top_p_cpu", None)
        for row_idx, req_id in enumerate(req_ids):
            if req_id not in self.active_request_ids:
                continue
            info = self._request_intermediate_info(runner, req_id)
            duplex = info.get("duplex") if isinstance(info, dict) else None
            if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
                continue
            session_id = duplex.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                session_id = None
            try:
                seq_raw = duplex.get("seq")
                seq = int(seq_raw) if seq_raw is not None else None
            except (TypeError, ValueError):
                seq = None
            payload = duplex.get("payload")
            if not isinstance(payload, dict):
                payload = None
            request = requests.get(req_id) if isinstance(requests, dict) else None
            sampling_params = getattr(request, "sampling_params", None)
            try:
                max_tokens = int(getattr(sampling_params, "max_tokens", 0) or 0)
            except (TypeError, ValueError):
                max_tokens = 0
            if max_tokens <= 0 and req_states is not None and hasattr(req_states, "max_tokens"):
                req_idx = getattr(req_states, "req_id_to_index", {}).get(req_id)
                if req_idx is not None and req_idx < len(req_states.max_tokens):
                    try:
                        max_tokens = int(req_states.max_tokens[req_idx])
                    except (TypeError, ValueError):
                        max_tokens = 0
            rows.append(
                DuplexSamplingRow(
                    row_idx=row_idx,
                    request_id=req_id,
                    session_id=session_id,
                    seq=seq,
                    payload=payload,
                    max_tokens=max_tokens if max_tokens > 0 else None,
                    temperature=_host_value(temperature_cpu, row_idx, float),
                    top_k=_host_value(top_k_cpu, row_idx, int),
                    top_p=_host_value(top_p_cpu, row_idx, float),
                )
            )
        return tuple(rows)

    def clear(self) -> None:
        self.active_request_ids.clear()
        self.hook_active = False


class DuplexSamplingRunnerMixin:
    """Runner-side wiring for the experimental duplex sampling hook.

    GPU and NPU AR runners are siblings, not parent and child, so neither
    inherits the other's ``_sample``. Keep the hook here and have each runner
    opt in, instead of copying the GPU ``_sample`` path and dropping the
    ``prepare_duplex_sampling`` call.
    """

    def _init_duplex_sampling_state(self) -> None:
        """Reset hook resolution. Call from ``__init__``, before ``load_model``."""
        self._duplex_sampling_hook = None
        self._duplex_sampling_hook_resolved = False

    def _resolve_duplex_sampling_hook(self, *, force: bool = False):
        """Bind ``model.prepare_duplex_sampling``. Call with ``force`` after load."""
        if not force and getattr(self, "_duplex_sampling_hook_resolved", False):
            return self._duplex_sampling_hook
        candidate = getattr(getattr(self, "model", None), "prepare_duplex_sampling", None)
        self._duplex_sampling_hook = candidate if callable(candidate) else None
        self._duplex_sampling_hook_resolved = True
        if self._duplex_sampling_hook is not None and not hasattr(self, "_duplex_sampling_helper"):
            self._duplex_sampling_helper = DuplexSamplingHelper()
        return self._duplex_sampling_hook

    def _update_duplex_sampling_states(self, scheduler_output: SchedulerOutput) -> None:
        """Refresh which batch rows are duplex rows. Call from ``_update_states``."""
        if self._resolve_duplex_sampling_hook() is None:
            return
        helper = getattr(self, "_duplex_sampling_helper", None)
        if helper is not None:
            helper.update_states(self, scheduler_output)

    def _apply_duplex_sampling(self, logits: torch.Tensor, prepared_sampling_metadata: SamplingMetadata) -> None:
        """Hand the model its duplex rows. Call from ``_sample``, before the sampler."""
        prepare_duplex_sampling = self._resolve_duplex_sampling_hook()
        if prepare_duplex_sampling is None:
            return
        helper = getattr(self, "_duplex_sampling_helper", None)
        rows = helper.rows(self) if helper is not None and helper.active_request_ids else ()
        # Publish the row set even when empty, so the model skips its per-row
        # prompt-token scan (a host sync per row) in turn-based serving.
        prepare_duplex_sampling(logits, prepared_sampling_metadata, rows)
        if helper is not None:
            helper.hook_active = bool(rows)

    def _clear_duplex_sampling(self) -> None:
        helper = getattr(self, "_duplex_sampling_helper", None)
        if helper is not None:
            helper.clear()

    def _maybe_apply_duplex_window_reanchor(self, scheduler_output: SchedulerOutput | None = None) -> None:
        """Apply in-place KV reanchor and rotation on worker before model forward."""
        reanchor_hook = getattr(getattr(self, "model", None), "apply_duplex_kv_reanchor", None)
        if callable(reanchor_hook):
            try:
                reanchor_hook(self, scheduler_output=scheduler_output)
            except TypeError:
                reanchor_hook(self)
            return

        helper = getattr(self, "_duplex_window_helper", None)
        if helper is not None and hasattr(helper, "maybe_apply_reanchor"):
            try:
                helper.maybe_apply_reanchor(self, scheduler_output=scheduler_output)
            except TypeError:
                helper.maybe_apply_reanchor(self)
            return

        input_batch = getattr(self, "input_batch", None)
        if input_batch is None:
            return
        num_reqs = getattr(input_batch, "num_reqs", len(getattr(input_batch, "req_ids", [])))
        req_ids = getattr(input_batch, "req_ids", [])[:num_reqs]
        has_reanchor = False
        for req_id in req_ids:
            info = DuplexSamplingHelper._request_intermediate_info(self, req_id)
            if isinstance(info, dict) and isinstance(info.get("duplex"), dict):
                if "stage0_reanchor" in info["duplex"]:
                    has_reanchor = True
                    break
        if not has_reanchor:
            return

        model_module = getattr(getattr(self, "model", None), "__module__", "")
        if "minicpmo_4_5" in model_module:
            from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
                MiniCPMO45DuplexWorkerHelper,
            )

            MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(self, scheduler_output=scheduler_output)


__all__ = ["DuplexSamplingHelper", "DuplexSamplingRow", "DuplexSamplingRunnerMixin"]
