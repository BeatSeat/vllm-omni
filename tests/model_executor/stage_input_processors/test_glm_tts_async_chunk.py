# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.glm_tts.glm_tts_dit_wrapper import (
    split_request_ids,
    valid_speech_tokens,
)
from vllm_omni.model_executor.stage_input_processors.glm_tts import (
    ar_to_dit,
    ar_to_dit_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Request:
    external_req_id = "req-1"
    additional_information = None

    def is_finished(self) -> bool:
        return False


def _transfer_manager() -> SimpleNamespace:
    connector = SimpleNamespace(
        config={
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 25,
            }
        }
    )
    return SimpleNamespace(connector=connector)


def _progressive_transfer_manager() -> SimpleNamespace:
    connector = SimpleNamespace(
        config={
            "extra": {
                "codec_chunk_frames": [25, 50, 200],
                "codec_left_context_frames": 25,
            }
        }
    )
    return SimpleNamespace(connector=connector)


def _source_output(mm: dict) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                multimodal_output=mm,
            )
        ]
    )


def test_ar_to_dit_filters_invalid_tokens_and_preserves_conditioning() -> None:
    prompt_token = torch.tensor([[1, 2, 3]])
    prompt_feat = torch.randn(1, 4, 80)
    embedding = torch.randn(1, 192)
    source_outputs = [
        _source_output(
            {
                "speech_tokens": torch.tensor([[-1, 0, 7, -1, 32767]]),
                "prompt_token": prompt_token,
                "prompt_feat": prompt_feat,
                "embedding": embedding,
            }
        )
    ]

    # Build mock stage_list matching the standard process_engine_inputs interface
    stage_client = SimpleNamespace(engine_outputs=source_outputs)
    stage_list = [stage_client]  # stage_id=0 is the AR stage
    engine_input_source = [0]
    outputs = ar_to_dit(stage_list, engine_input_source)

    assert len(outputs) == 1
    assert outputs[0]["prompt_token_ids"] == [0, 7, 32767]
    additional_info = outputs[0]["additional_information"]
    assert additional_info["speech_tokens"] == [0, 7, 32767]
    assert additional_info["prompt_speech_token"] is prompt_token
    assert additional_info["prompt_feat"] is prompt_feat
    assert additional_info["embedding"] is embedding


def test_dit_explicit_empty_speech_tokens_override_scheduler_placeholder() -> None:
    placeholder = torch.tensor([0], dtype=torch.long)

    token = valid_speech_tokens([], device=placeholder.device, fallback=placeholder)

    assert token.numel() == 0


def test_ar_to_dit_async_chunk_restores_serialized_conditioning_payload() -> None:
    transfer = _transfer_manager()
    request = _Request()
    prompt_token = torch.tensor([[1, 2, 3]])
    prompt_feat = torch.randn(1, 4, 80)
    embedding = torch.randn(1, 192)
    request.additional_information = serialize_additional_information(
        {
            "prompt_speech_token": prompt_token,
            "prompt_feat": prompt_feat,
            "embedding": embedding,
        }
    )

    payload = ar_to_dit_async_chunk(
        transfer,
        {"speech_tokens": torch.tensor([[0]])},
        request,
        is_finished=False,
    )

    assert payload is None
    state = transfer.request_payload["req-1"]["_glm_tts_async_state"]
    prompt_payload = state["prompt_payload"]
    assert torch.equal(prompt_payload["prompt_speech_token"], prompt_token)
    assert torch.equal(prompt_payload["prompt_feat"], prompt_feat)
    assert torch.equal(prompt_payload["embedding"], embedding)


def test_split_request_ids_uses_seq_token_counts() -> None:
    ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)

    split = split_request_ids(ids, seq_token_counts=[2, 3, 1])

    assert [part.tolist() for part in split] == [[1, 2], [3, 4, 5], [6]]


def test_ar_to_dit_async_chunk_emits_cumulative_prefixes() -> None:
    transfer = _transfer_manager()
    request = _Request()
    payloads = []

    for token in range(75):
        payload = ar_to_dit_async_chunk(
            transfer,
            {"speech_tokens": torch.tensor([[token]])},
            request,
            is_finished=False,
        )
        if payload is not None:
            payloads.append(payload)

    assert len(payloads) == 3
    assert payloads[0]["codes"]["audio"] == list(range(25))
    assert payloads[0]["token_offset"] == 0
    assert payloads[0]["block_pattern"] == [25]
    assert payloads[1]["codes"]["audio"] == list(range(50))
    assert payloads[1]["token_offset"] == 25
    assert payloads[1]["block_pattern"] == [25]
    assert payloads[2]["codes"]["audio"] == list(range(75))
    assert payloads[2]["token_offset"] == 50
    assert payloads[2]["block_pattern"] == [25]


def test_ar_to_dit_async_chunk_keeps_progressive_block_pattern_fixed() -> None:
    transfer = _progressive_transfer_manager()
    request = _Request()
    payloads = []

    for token in range(30):
        payload = ar_to_dit_async_chunk(
            transfer,
            {"speech_tokens": torch.tensor([[token]])},
            request,
            is_finished=False,
        )
        if payload is not None:
            payloads.append(payload)

    final_payload = ar_to_dit_async_chunk(
        transfer,
        None,
        request,
        is_finished=True,
    )

    assert len(payloads) == 1
    assert payloads[0]["chunk_sizes_history"] == [25]
    assert payloads[0]["block_pattern"] == [25, 50, 200]
    assert final_payload is not None
    assert final_payload["chunk_sizes_history"] == [25, 5]
    assert final_payload["block_pattern"] == [25, 50, 200]


def test_ar_to_dit_async_chunk_terminal_flushes_boundary_prefix() -> None:
    transfer = _transfer_manager()
    request = _Request()
    payloads = []

    for token in range(50):
        payload = ar_to_dit_async_chunk(
            transfer,
            {"speech_tokens": torch.tensor([[token]])},
            request,
            is_finished=False,
        )
        if payload is not None:
            payloads.append(payload)

    final_payload = ar_to_dit_async_chunk(
        transfer,
        None,
        request,
        is_finished=True,
    )

    assert len(payloads) == 2
    assert final_payload is not None
    assert final_payload["codes"]["audio"] == list(range(50))
    assert final_payload["token_offset"] == 50
    assert final_payload["left_context_size"] == 50
    assert final_payload["req_id"] == ["req-1"]
    assert bool(final_payload["stream_finished"].item())


def test_ar_to_dit_async_chunk_finished_flushes_partial_unsent_tail() -> None:
    transfer = _transfer_manager()
    request = _Request()
    payloads = []

    for token in range(30):
        payload = ar_to_dit_async_chunk(
            transfer,
            {"speech_tokens": torch.tensor([[token]])},
            request,
            is_finished=False,
        )
        if payload is not None:
            payloads.append(payload)

    final_payload = ar_to_dit_async_chunk(
        transfer,
        None,
        request,
        is_finished=True,
    )

    assert len(payloads) == 1
    assert payloads[0]["codes"]["audio"] == list(range(25))
    assert payloads[0]["token_offset"] == 0
    assert final_payload is not None
    assert final_payload["codes"]["audio"] == list(range(30))
    assert final_payload["token_offset"] == 25
    assert final_payload["left_context_size"] == 25
    assert final_payload["req_id"] == ["req-1"]
    assert bool(final_payload["meta"]["finished"].item())
    assert bool(final_payload["stream_finished"].item())
    assert (
        ar_to_dit_async_chunk(
            transfer,
            None,
            request,
            is_finished=True,
        )
        is None
    )
