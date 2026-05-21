# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for GLM-4-Voice: AR → Decoder bridge.

Two functions:
- ``ar_to_decoder``: Non-streaming (sync) — collects all speech tokens
  from completed AR output and forwards to the decoder stage.
- ``ar_to_decoder_async_chunk``: Streaming — filters audio tokens from
  the interleaved AR stream and emits delta chunks via
  ``OmniPayloadStruct``.

Audio tokens in the AR output are identified by ``token_id >= audio_offset``
(151552 for THUDM/glm-4-voice-9b).  They are converted to 0-based speech
tokens (0..16383) before forwarding to the decoder.
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

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))

    request_state = transfer_manager.request_payload.get(request_id)
    if not isinstance(request_state, dict) or "_glm4_voice_state" not in request_state:
        request_state = {
            "_glm4_voice_state": {
                "seen_len": 0,
                "emitted_len": 0,
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
        if tok_int >= _AUDIO_OFFSET:
            speech_tok = tok_int - _AUDIO_OFFSET
            if 0 <= speech_tok < _AUDIO_VOCAB_SIZE:
                token_list.append([speech_tok])

    total_audio = len(token_list)
    emitted_len = state["emitted_len"]
    pending = total_audio - emitted_len

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
