# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for GLM-4-Voice: AR → Decoder bridge.

Two functions:
- ``ar_to_decoder``: Non-streaming (sync) — collects all speech tokens
  from completed AR output and forwards to the decoder stage.
- ``ar_to_decoder_async_chunk``: Streaming — filters audio tokens from
  the interleaved AR stream and emits delta chunks via
  ``OmniPayloadStruct``.

Audio tokens in the AR output are identified by ``token_id >= audio_offset``.
The offset is resolved from runtime/config metadata when available, with
151552 kept as the THUDM/glm-4-voice-9b fallback.  Audio tokens are converted
to 0-based speech tokens (0..16383) before forwarding to the decoder.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

_AUDIO_OFFSET = 151552
_AUDIO_VOCAB_SIZE = 16384
_OFFICIAL_STREAMING_CHUNK_SIZES = [25, 50, 100, 150, 200]


def _ensure_list(x: Any) -> list:
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    if x is None:
        return []
    try:
        return list(x)
    except TypeError:
        return [x]


def _connector_extra(transfer_manager: Any) -> dict[str, Any]:
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    if isinstance(raw_cfg, dict):
        extra = raw_cfg.get("extra", raw_cfg)
        return extra if isinstance(extra, dict) else {}
    return {}


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _as_int(value[0], default)
    return int(value)


def _resolve_audio_offset(cfg: dict[str, Any], request: Any | None = None) -> int:
    """Resolve GLM-4-Voice audio token offset, matching the official tokenizer.

    The official demo obtains this via
    ``tokenizer.convert_tokens_to_ids("<|audio_0|>")``.  The stage processor
    may run without tokenizer access, so prefer injected config/request
    metadata and retain the known THUDM fallback.
    """
    for key in ("audio_offset", "glm4_voice_audio_offset"):
        if key in cfg:
            return _as_int(cfg.get(key), _AUDIO_OFFSET)

    additional = getattr(request, "additional_information", None)
    if isinstance(additional, dict):
        for key in ("audio_offset", "glm4_voice_audio_offset"):
            if key in additional:
                return _as_int(additional.get(key), _AUDIO_OFFSET)

    return _AUDIO_OFFSET


def _parse_chunk_sizes(cfg: dict[str, Any]) -> list[int]:
    """Return official progressive chunk sizes unless a fixed size is requested."""
    raw_sizes = (
        cfg.get("glm4_voice_chunk_sizes")
        or cfg.get("codec_chunk_frames_list")
        or cfg.get("streaming_chunk_sizes")
    )
    if raw_sizes is not None:
        if isinstance(raw_sizes, str):
            sizes = [int(x.strip()) for x in raw_sizes.split(",") if x.strip()]
        else:
            sizes = [int(x) for x in _ensure_list(raw_sizes)]
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError(f"Invalid GLM-4-Voice chunk sizes: {raw_sizes!r}")
        return sizes

    # Backward compatibility for existing configs/tests that set only a fixed
    # codec_chunk_frames.  New GLM-4-Voice configs should use the official list.
    if "codec_chunk_frames" in cfg:
        chunk_size = int(cfg.get("codec_chunk_frames", 25))
        if chunk_size <= 0:
            raise ValueError(f"Invalid codec_chunk_frames={chunk_size}")
        return [chunk_size]

    return list(_OFFICIAL_STREAMING_CHUNK_SIZES)


def _current_chunk_size(chunk_sizes: list[int], chunk_index: int) -> int:
    return chunk_sizes[min(max(chunk_index, 0), len(chunk_sizes) - 1)]


def ar_to_decoder(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Non-streaming AR → Decoder transfer.

    Collects all audio tokens from the AR stage output, converts them
    to 0-based speech tokens, and packages as ``OmniTokensPrompt``
    for the decoder stage.
    """
    engine_inputs: list[OmniTokensPrompt] = []

    for source_output in source_outputs:
        output = source_output.outputs[0]
        output_ids = _ensure_list(getattr(output, "cumulative_token_ids", []))

        speech_tokens: list[int] = []
        for tok in output_ids:
            tok_int = int(tok)
            if tok_int >= _AUDIO_OFFSET:
                speech_tok = tok_int - _AUDIO_OFFSET
                if 0 <= speech_tok < _AUDIO_VOCAB_SIZE:
                    speech_tokens.append(speech_tok)

        if not speech_tokens:
            continue

        req_id = str(getattr(source_output, "request_id", "0"))

        additional_info: dict[str, Any] = {
            "meta": {
                "stream_finished": True,
                "finished": True,
                "req_id": [req_id],
            },
        }

        engine_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=speech_tokens,
                additional_information=additional_info,
            )
        )

    if not engine_inputs:
        logger.warning("GLM-4-Voice: no speech tokens from AR stage")

    return engine_inputs


def ar_to_decoder_async_chunk(
    transfer_manager: Any,
    pooling_output: Any,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Streaming AR → Decoder transfer with delta chunks.

    Accumulates audio tokens from the interleaved AR stream and emits
    delta (new-only) chunks when ``codec_chunk_frames`` tokens are ready.
    Each chunk becomes ``prompt_token_ids`` in the decoder via the
    ``codes.audio`` → ``code_predictor_codes`` framework path.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    cfg = _connector_extra(transfer_manager)
    audio_offset = _resolve_audio_offset(cfg, request)
    chunk_sizes = _parse_chunk_sizes(cfg)

    request_state = transfer_manager.request_payload.get(request_id)
    if not isinstance(request_state, dict) or "_glm4_voice_state" not in request_state:
        request_state = {
            "_glm4_voice_state": {
                "seen_len": 0,
                "emitted_len": 0,
                "chunk_idx": 0,
                "terminal_sent": False,
            },
        }
        transfer_manager.request_payload[request_id] = request_state

    state = request_state["_glm4_voice_state"]
    if state.get("terminal_sent", False):
        return None

    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []))
    seen_len = state["seen_len"]
    new_tokens = output_token_ids[seen_len:]
    state["seen_len"] = len(output_token_ids)

    if not hasattr(transfer_manager, "code_prompt_token_ids"):
        transfer_manager.code_prompt_token_ids = defaultdict(list)
    token_list = transfer_manager.code_prompt_token_ids[request_id]
    for tok in new_tokens:
        tok_int = int(tok)
        if tok_int >= audio_offset:
            speech_tok = tok_int - audio_offset
            if 0 <= speech_tok < _AUDIO_VOCAB_SIZE:
                token_list.append([speech_tok])

    total_audio = len(token_list)
    emitted_len = state["emitted_len"]
    pending = total_audio - emitted_len
    chunk_size = _current_chunk_size(chunk_sizes, int(state.get("chunk_idx", 0)))

    should_emit = pending >= chunk_size or (finished and pending > 0)

    if not should_emit:
        if finished and not state["terminal_sent"]:
            state["terminal_sent"] = True
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(
                    finished=torch.tensor(True, dtype=torch.bool),
                    stream_finished=torch.tensor(True, dtype=torch.bool),
                    req_id=[request_id],
                ),
            )
        return None

    delta_tokens = [int(frame[0]) for frame in token_list[emitted_len:total_audio]]
    state["emitted_len"] = total_audio
    state["chunk_idx"] = int(state.get("chunk_idx", 0)) + 1

    payload = OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(delta_tokens, dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=torch.tensor(finished, dtype=torch.bool),
            req_id=[request_id],
        ),
    )

    if finished:
        state["terminal_sent"] = True

    return payload
