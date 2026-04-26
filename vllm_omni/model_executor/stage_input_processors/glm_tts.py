# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for GLM-TTS: AR → DiT Pipeline.

Supports both sync (non-streaming) and async_chunk (streaming) modes.
Adapted for LLM_GENERATION execution type on stage 1 (DiT).
"""

from collections import defaultdict
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Sync processor: collect all speech tokens from AR, pass to DiT
# ---------------------------------------------------------------------------


def ar_to_dit(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-streaming processor: collect all speech tokens from AR, pass to DiT.

    Also propagates voice cloning data (prompt_token, prompt_feat, embedding)
    from the AR model's multimodal_output to the DiT stage.

    Args:
        stage_list: List of stage outputs.
        engine_input_source: Source stage indices.
        prompt: Original prompt (unused, voice clone data comes from AR output).
        requires_multimodal_data: Whether multimodal data is required.

    Returns:
        List of OmniTokensPrompt for DiT stage.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

    ar_outputs = _validate_stage_inputs(stage_list, engine_input_source)
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
            for key in ("prompt_token", "prompt_feat", "embedding"):
                val = mm.get(key)
                if val is not None:
                    additional_info[key] = val
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
            dit_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[0],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information={
                        "speech_tokens": [],
                        "error": f"Invalid speech token range: [{min_t}, {max_t}]",
                    },
                )
            )
            continue

        logger.info(
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
        for key in ("prompt_token", "prompt_feat", "embedding"):
            val = mm.get(key)
            if val is not None:
                additional_info[key] = val

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
    """Extract dict from additional_information object."""
    if isinstance(additional_information, dict):
        return additional_information
    if additional_information is not None and hasattr(additional_information, "entries"):
        result: dict[str, Any] = {}
        for key, entry in additional_information.entries.items():
            if hasattr(entry, "tensor_data") and entry.tensor_data is not None:
                result[key] = entry.tensor_data
            elif hasattr(entry, "list_data") and entry.list_data is not None:
                result[key] = entry.list_data
        return result
    return {}


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
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    # Read connector chunk config
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    if chunk_size <= 0 or left_context_size_config < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}"
        )

    # Initialize per-request state (like CosyVoice3)
    if not hasattr(transfer_manager, "request_payload"):
        transfer_manager.request_payload = {}

    request_state = transfer_manager.request_payload.get(request_id)
    if not isinstance(request_state, dict) or "_glm_tts_async_state" not in request_state:
        # Extract voice clone conditioning from request additional_information
        info = _decode_additional_information(getattr(request, "additional_information", None))
        prompt_payload: dict[str, Any] = {}
        for key in ("prompt_token", "prompt_feat", "embedding"):
            value = _to_cpu_tensor(info.get(key))
            if value is not None:
                prompt_payload[key] = value

        # Also try to extract from pooling_output (first call)
        if isinstance(pooling_output, dict):
            for key in ("prompt_token", "prompt_feat", "embedding"):
                if key in prompt_payload:
                    continue
                value = _to_cpu_tensor(pooling_output.get(key))
                if value is not None:
                    prompt_payload[key] = value

        request_state = {
            "_glm_tts_async_state": {
                "seen_len": 0,
                "sent_prompt": False,
                "emitted_chunks": 0,
                "emitted_token_len": 0,
                "terminal_sent": False,
                "prompt_payload": prompt_payload,
            }
        }
        transfer_manager.request_payload[request_id] = request_state

    state = request_state["_glm_tts_async_state"]
    if bool(state.get("terminal_sent", False)):
        return None

    # Accumulate new speech token from this step
    if not hasattr(transfer_manager, "code_prompt_token_ids"):
        transfer_manager.code_prompt_token_ids = defaultdict(list)

    if isinstance(pooling_output, dict):
        token = _extract_last_speech_token(pooling_output)
        if token is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(token)
    elif not finished:
        return None

    token_frames = transfer_manager.code_prompt_token_ids[request_id]
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

    # Check if finished but all tokens already emitted
    if finished and length <= emitted_token_len:
        payload = {
            "codes": {"audio": []},
            "meta": {
                "finished": torch.tensor(True, dtype=torch.bool),
                "left_context_size": emitted_token_len,
            },
            "token_offset": emitted_token_len,
            "left_context_size": emitted_token_len,
            "req_id": [request_id],
            "stream_finished": torch.tensor(True, dtype=torch.bool),
        }
        if not state.get("sent_prompt", False):
            payload.update(state.get("prompt_payload", {}))
            state["sent_prompt"] = True
        state["terminal_sent"] = True
        return payload

    # Determine chunk boundaries. Official GLM-TTS streams cumulative prefixes
    # (`all_patch_token`) into the flow stage; prefix reuse happens inside DiT.
    available = max(0, length - emitted_token_len)

    if not finished:
        if available < chunk_size:
            return None

    # Send the cumulative prefix through the current chunk. token_offset marks
    # the already emitted stable prefix and is used only for output cropping.
    if emitted_token_len == 0:
        end_index = min(length, chunk_size)
        token_offset = 0
    else:
        end_index = length if finished else min(length, emitted_token_len + chunk_size)
        token_offset = emitted_token_len

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
    return payload
