# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for GLM-TTS: AR → DiT Pipeline.

Supports both sync (non-streaming) and async_chunk (streaming) modes.
Adapted for LLM_GENERATION execution type on stage 1 (DiT).
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)


def _copy_voice_clone_payload(
    src: dict[str, Any],
    dst: dict[str, Any],
    *,
    to_cpu: bool = False,
    skip_existing: bool = False,
) -> None:
    if not skip_existing or "prompt_speech_token" not in dst:
        prompt = src.get("prompt_speech_token")
        if prompt is None:
            prompt = src.get("prompt_token")
        if prompt is not None:
            if to_cpu:
                prompt = _to_cpu_tensor(prompt)
            if prompt is not None:
                dst["prompt_speech_token"] = prompt

    for key in ("prompt_feat", "embedding"):
        if skip_existing and key in dst:
            continue
        val = src.get(key)
        if val is not None:
            if to_cpu:
                val = _to_cpu_tensor(val)
            if val is not None:
                dst[key] = val


# ---------------------------------------------------------------------------
# Sync processor: collect all speech tokens from AR, pass to DiT
# ---------------------------------------------------------------------------


def ar_to_dit(
    source_outputs: list[Any],
    _prompt: Any = None,
    _requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[Any]:
    """Non-streaming processor: collect all speech tokens from AR, pass to DiT.

    Also propagates voice cloning data (prompt_token, prompt_feat, embedding)
    from the AR model's multimodal_output to the DiT stage.

    Args:
        source_outputs: Outputs from the upstream AR stage.
        streaming_context: Unused for sync mode; kept for interface parity with
            other stage processors.

    Returns:
        List of OmniTokensPrompt for DiT stage.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    del streaming_context

    ar_outputs = source_outputs
    dit_inputs: list[OmniTokensPrompt] = []

    for output in ar_outputs:
        out = output.outputs[0]
        mm = out.multimodal_output

        speech_tokens = mm.get("speech_tokens")

        if speech_tokens is None:
            logger.warning("No speech_tokens in AR output, returning empty DiT input")
            dit_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[0],  # LLM_GENERATION needs at least 1 token
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information={
                        "speech_tokens": [],
                        "error": "No speech_tokens in AR output",
                    },
                )
            )
            continue

        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens = speech_tokens.to(torch.long).reshape(-1)
            # Filter -1 values (invalid/placeholder markers from prefill)
            # NOTE: 0 is VALID (first audio token <|audio_0|>), only -1 is invalid
            valid_tokens = speech_tokens[speech_tokens >= 0]
            token_list = valid_tokens.cpu().tolist()
        else:
            # Filter -1 from list as well
            token_list = [t for t in speech_tokens if t >= 0]

        if not token_list:
            logger.warning("No valid speech tokens after filtering -1 markers")
            additional_info: dict[str, Any] = {
                "speech_tokens": [],
                "error": "No valid speech tokens after filtering -1 markers",
            }
            _copy_voice_clone_payload(mm, additional_info)
            dit_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[0],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information=additional_info,
                )
            )
            continue

        min_t, max_t = min(token_list), max(token_list)
        if min_t < 0 or max_t >= 32768:
            logger.warning(
                "Invalid speech token range after filtering: range=[%d, %d]",
                min_t,
                max_t,
            )
            additional_info = {
                "speech_tokens": [],
                "error": f"Invalid speech token range: [{min_t}, {max_t}]",
            }
            _copy_voice_clone_payload(mm, additional_info)
            dit_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[0],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information=additional_info,
                )
            )
            continue

        logger.debug(
            "ar_to_dit: %d valid speech tokens, range=[%d, %d]",
            len(token_list),
            min_t,
            max_t,
        )

        # Build additional_information for DiT forward()
        additional_info = {
            "speech_tokens": token_list,
        }
        # Propagate voice cloning data from AR model's multimodal_output
        _copy_voice_clone_payload(mm, additional_info)

        # Speech tokens as prompt_token_ids for LLM_GENERATION scheduler
        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_list,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_info,
            )
        )

    return dit_inputs


# ---------------------------------------------------------------------------
# Helper: extract last speech token from AR model pooling output
# ---------------------------------------------------------------------------


def _extract_last_speech_token(pooling_output: dict[str, Any]) -> int | None:
    """Extract the last valid speech token from AR model output.

    GLM-TTS AR produces one speech token per decode step.
    Returns the token ID (relative to ATS, i.e. 0-based), or None.
    """
    speech_tokens = pooling_output.get("speech_tokens")
    if not isinstance(speech_tokens, torch.Tensor) or speech_tokens.numel() == 0:
        return None
    token_val = int(speech_tokens.reshape(-1).to(torch.long)[-1].item())
    # -1 = invalid/EOA marker
    if token_val < 0:
        return None
    return token_val


def _to_cpu_tensor(value: Any) -> torch.Tensor | None:
    """Convert value to CPU tensor if possible."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        if not value:
            return None
        if isinstance(value[0], torch.Tensor):
            return value[0].detach().cpu()
    return None


def _decode_additional_information(additional_information: Any) -> dict[str, Any]:
    """Decode additional_information to plain tensors/lists.

    Align with CosyVoice3's async-chunk path: tensor payloads must be
    reconstructed before we try to forward voice-clone conditioning.
    """
    return deserialize_additional_information(additional_information)


# ---------------------------------------------------------------------------
# Async streaming processor: emit speech token chunks as AR produces them
# ---------------------------------------------------------------------------


def ar_to_dit_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async streaming processor: emit speech token chunks as AR produces them.

    Accumulates per-step speech tokens and emits fixed-size cumulative prefixes
    for GLM-TTS flow-cache streaming.

    Follows the CosyVoice3 talker2code2wav_async_chunk transfer pattern:
    - Per-request state tracking (seen tokens, prompt sent flag)
    - First chunk carries voice clone conditioning payload
    - Each chunk includes token_offset, stream_finished, req_id
    - code_predictor_codes for chunk_transfer_adapter consumption

    Unlike codec left-context decoders, official GLM-TTS sends the cumulative
    AR token prefix to the flow stage on every chunk and reuses diffusion
    latents internally.  Here token_offset means the stable token prefix that
    has already been emitted, so DiT can crop regenerated audio after sampling.

    GLM-TTS produces single-token-per-step (no multi-codebook), so each entry
    in code_prompt_token_ids is a plain int, not a list of codebook values.
    """
    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    if request_id is None:
        raise ValueError("GLM-TTS async chunk request is missing request id")
    finished = bool(is_finished or request.is_finished())

    # Read connector chunk config (supports progressive list or single int)
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_frames_cfg = cfg.get("codec_chunk_frames", 25)
    if isinstance(chunk_frames_cfg, list):
        progressive_chunk_sizes = [int(c) for c in chunk_frames_cfg]
    else:
        progressive_chunk_sizes = [int(chunk_frames_cfg)]
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    crossfade_sec = float(cfg.get("crossfade_sec", 0.1))

    if not progressive_chunk_sizes or any(c <= 0 for c in progressive_chunk_sizes) or left_context_size_config < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_frames_cfg}, "
            f"codec_left_context_frames={left_context_size_config}"
        )

    # Initialize per-request state (like CosyVoice3)
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    code_prompt_token_ids = getattr(transfer_manager, "code_prompt_token_ids", None)
    if code_prompt_token_ids is None:
        code_prompt_token_ids = {}
        transfer_manager.code_prompt_token_ids = code_prompt_token_ids
    code_prompt_token_ids.setdefault(request_id, [])
    request_state = request_payload.get(request_id)
    if not isinstance(request_state, dict) or "_glm_tts_async_state" not in request_state:
        # Extract voice clone conditioning from request additional_information
        info = _decode_additional_information(getattr(request, "additional_information", None))
        prompt_payload: dict[str, Any] = {}
        _copy_voice_clone_payload(info, prompt_payload, to_cpu=True)

        # Also try to extract from pooling_output (first call)
        if isinstance(pooling_output, dict):
            _copy_voice_clone_payload(pooling_output, prompt_payload, to_cpu=True, skip_existing=True)

        request_state = {
            "_glm_tts_async_state": {
                "seen_len": 0,
                "sent_prompt": False,
                "emitted_chunks": 0,
                "emitted_token_len": 0,
                "terminal_sent": False,
                "prompt_payload": prompt_payload,
                "chunk_sizes_history": [],
                "block_pattern": progressive_chunk_sizes,
            }
        }
        request_payload[request_id] = request_state

    state = request_state["_glm_tts_async_state"]
    if bool(state.get("terminal_sent", False)):
        return None

    # Accumulate new speech token from this step
    # code_prompt_token_ids is always available via the mixin property.
    if isinstance(pooling_output, dict):
        token = _extract_last_speech_token(pooling_output)
        if token is not None:
            code_prompt_token_ids[request_id].append(token)
    elif not finished:
        return None

    token_frames = code_prompt_token_ids[request_id]
    length = len(token_frames)

    if length <= 0:
        if finished:
            payload: dict[str, Any] = {
                "codes": {"audio": []},
                "meta": {
                    "finished": torch.tensor(True, dtype=torch.bool),
                    "left_context_size": 0,
                },
                "token_offset": 0,
                "left_context_size": 0,
                "req_id": [request_id],
                "stream_finished": torch.tensor(True, dtype=torch.bool),
            }
            if not state.get("sent_prompt", False):
                payload.update(state.get("prompt_payload", {}))
                state["sent_prompt"] = True
            state["terminal_sent"] = True
            return payload
        return None

    emitted_token_len = int(state.get("emitted_token_len", 0))

    # If AR has finished exactly on a chunk boundary, emit the cumulative
    # prefix one final time.  The official GLM-TTS streaming path keeps a
    # lookahead/fade tail from non-final chunks and flushes it only when the
    # final full-prefix pass is marked finished.
    if finished and length <= emitted_token_len:
        payload = {
            "codes": {"audio": list(token_frames)},
            "meta": {
                "finished": torch.tensor(True, dtype=torch.bool),
                "left_context_size": emitted_token_len,
            },
            "token_offset": emitted_token_len,
            "left_context_size": emitted_token_len,
            "req_id": [request_id],
            "stream_finished": torch.tensor(True, dtype=torch.bool),
            "chunk_sizes_history": list(state.get("chunk_sizes_history", [])),
            "block_pattern": list(state.get("block_pattern", progressive_chunk_sizes)),
            "crossfade_sec": crossfade_sec,
        }
        if not state.get("sent_prompt", False):
            payload.update(state.get("prompt_payload", {}))
            state["sent_prompt"] = True
        state["terminal_sent"] = True
        return payload

    # Progressive chunk size: 25 → 50 → 200 (official GLM-TTS pattern)
    chunk_count = int(state.get("emitted_chunks", 0))
    if chunk_count < len(progressive_chunk_sizes):
        current_chunk_size = progressive_chunk_sizes[chunk_count]
    else:
        current_chunk_size = progressive_chunk_sizes[-1]

    # Determine chunk boundaries. Official GLM-TTS streams cumulative prefixes
    # (`all_patch_token`) into the flow stage; prefix reuse happens inside DiT.
    available = max(0, length - emitted_token_len)

    if not finished:
        if available < current_chunk_size:
            return None

    # Send the cumulative prefix through the current chunk. token_offset marks
    # the already emitted stable prefix and is used only for output cropping.
    if emitted_token_len == 0:
        end_index = min(length, current_chunk_size)
        token_offset = 0
    else:
        end_index = length if finished else min(length, emitted_token_len + current_chunk_size)
        token_offset = emitted_token_len

    # Track actual chunk token sizes for lookahead/cache slicing.  Keep the
    # DiT attention block pattern fixed to the configured official pattern
    # (25 -> 50 -> 200 by default), rather than using a short final chunk.
    actual_new_tokens = end_index - emitted_token_len
    chunk_sizes_history: list[int] = list(state.get("chunk_sizes_history", []))
    chunk_sizes_history.append(actual_new_tokens)
    block_pattern = list(state.get("block_pattern", progressive_chunk_sizes))

    # GLM-TTS: single token per frame, no codebook interleaving
    code_predictor_codes = list(token_frames[:end_index])

    payload = {
        "codes": {"audio": code_predictor_codes},
        "meta": {
            "finished": torch.tensor(finished, dtype=torch.bool),
            "left_context_size": token_offset,
        },
        "token_offset": token_offset,
        "left_context_size": token_offset,
        "req_id": [request_id],
        "stream_finished": torch.tensor(finished, dtype=torch.bool),
        "chunk_sizes_history": chunk_sizes_history,
        "block_pattern": block_pattern,
        "crossfade_sec": crossfade_sec,
    }

    # First chunk: attach voice clone conditioning payload
    if not state.get("sent_prompt", False):
        payload.update(state.get("prompt_payload", {}))
        state["sent_prompt"] = True

    # Update state
    if not finished:
        state["emitted_token_len"] = max(emitted_token_len, end_index)
    else:
        state["terminal_sent"] = True

    state["emitted_chunks"] = int(state.get("emitted_chunks", 0)) + 1
    state["chunk_sizes_history"] = chunk_sizes_history

    return payload
