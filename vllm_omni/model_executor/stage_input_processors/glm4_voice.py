# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for GLM-4-Voice: AR → Decoder bridge.

Two functions:
- ``ar_to_decoder``: Non-streaming (sync) transfer of all speech tokens.
- ``ar_to_decoder_async_chunk``: Streaming transfer with progressive
  chunk sizes [25, 50, 100, 150, 200] matching the reference
  ``web_demo.py`` implementation.

Audio tokens in the AR output are interleaved with text tokens.  Only
tokens >= ``audio_offset`` are forwarded to the decoder; text tokens
are filtered out.  The ``audio_offset`` is extracted from the AR stage's
``additional_information``.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

# Progressive chunk sizes (from reference web_demo.py).
_CHUNK_SIZES = [25, 50, 100, 150, 200]


def _extract_speech_tokens(
    source_outputs: list[Any],
) -> list[int]:
    """Collect speech_tokens from all AR output steps."""
    tokens: list[int] = []
    for output in source_outputs:
        mm = getattr(output, "multimodal_output", None)
        if mm is None:
            continue
        st = mm.get("speech_tokens")
        if st is not None:
            if isinstance(st, list):
                tokens.extend(st)
            else:
                tokens.append(int(st))
    return tokens


def _extract_last_speech_token(source_outputs: list[Any]) -> int:
    """Extract the last speech token from AR output."""
    for output in reversed(source_outputs):
        mm = getattr(output, "multimodal_output", None)
        if mm is None:
            continue
        last = mm.get("last_speech_token", -1)
        if last >= 0:
            return int(last)
    return -1


def ar_to_decoder(
    source_outputs: list[Any],
    prompt: Any,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Non-streaming AR → Decoder transfer.

    Collects all speech tokens from the AR stage output and packages
    them as an ``OmniTokensPrompt`` for the decoder stage.
    """
    speech_tokens = _extract_speech_tokens(source_outputs)
    if not speech_tokens:
        logger.warning("GLM-4-Voice: no speech tokens from AR stage")
        return []

    # Filter out any invalid tokens.
    speech_tokens = [t for t in speech_tokens if 0 <= t < 16384]

    additional_info: dict[str, Any] = {
        "speech_tokens": speech_tokens,
        "stream_finished": True,
        "prompt_token": None,
        "prompt_feat": None,
        "embedding": None,
    }

    # Propagate voice clone data if available.
    if source_outputs:
        last_mm = getattr(source_outputs[-1], "multimodal_output", None)
        if last_mm is not None:
            for key in ("prompt_token", "prompt_feat", "embedding"):
                if key in last_mm:
                    additional_info[key] = last_mm[key]

    # Build a minimal OmniTokensPrompt for the decoder.
    return [
        OmniTokensPrompt(
            prompt_token_ids=[0],  # dummy token; decoder ignores input_ids
            additional_information=additional_info,
        )
    ]


# Per-request async state.
_async_state: dict[str, dict[str, Any]] = {}


def ar_to_decoder_async_chunk(
    transfer_manager: Any,
    pooling_output: Any,
    request: Any,
    is_finished: bool,
) -> Any | None:
    """Streaming AR → Decoder transfer with progressive chunk sizes.

    Accumulates audio tokens from the interleaved AR stream and emits
    cumulative prefix chunks at progressive sizes [25, 50, 100, 150, 200].
    """
    req_id = str(getattr(request, "request_id", "0"))

    # Initialize per-request state.
    if req_id not in _async_state:
        _async_state[req_id] = {
            "audio_tokens": [],
            "chunk_idx": 0,
            "emitted_len": 0,
            "terminal_sent": False,
        }

    state = _async_state[req_id]

    # Extract new speech token from this AR step.
    if pooling_output is not None:
        mm = getattr(pooling_output, "multimodal_output", None)
        if mm is not None:
            last_token = mm.get("last_speech_token", -1)
            if last_token >= 0:
                state["audio_tokens"].append(last_token)

    audio_tokens = state["audio_tokens"]
    chunk_idx = state["chunk_idx"]
    emitted_len = state["emitted_len"]

    # Determine current chunk threshold.
    if chunk_idx < len(_CHUNK_SIZES):
        chunk_threshold = _CHUNK_SIZES[chunk_idx]
    else:
        chunk_threshold = _CHUNK_SIZES[-1]

    pending = len(audio_tokens) - emitted_len
    should_emit = pending >= chunk_threshold or (is_finished and pending > 0)

    if not should_emit:
        if is_finished and not state["terminal_sent"]:
            # No new tokens but generation is done — send terminal signal.
            state["terminal_sent"] = True
            _async_state.pop(req_id, None)
            return OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information={
                    "speech_tokens": [],
                    "stream_finished": True,
                },
            )
        return None

    # Emit current chunk (cumulative prefix of all audio tokens so far).
    current_tokens = list(audio_tokens)
    state["emitted_len"] = len(audio_tokens)
    if chunk_idx < len(_CHUNK_SIZES) - 1:
        state["chunk_idx"] = chunk_idx + 1

    additional_info: dict[str, Any] = {
        "speech_tokens": current_tokens,
        "stream_finished": is_finished,
    }

    if is_finished:
        state["terminal_sent"] = True
        _async_state.pop(req_id, None)

    return OmniTokensPrompt(
        prompt_token_ids=[0],
        additional_information=additional_info,
    )
