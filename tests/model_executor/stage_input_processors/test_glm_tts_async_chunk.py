# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.glm_tts import ar_to_dit_async_chunk

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
    assert payloads[1]["codes"]["audio"] == list(range(50))
    assert payloads[1]["token_offset"] == 25
    assert payloads[2]["codes"]["audio"] == list(range(75))
    assert payloads[2]["token_offset"] == 50


def test_ar_to_dit_async_chunk_terminal_empty_payload_keeps_cleanup_metadata() -> None:
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
    assert final_payload["codes"]["audio"] == []
    assert final_payload["token_offset"] == 50
    assert final_payload["left_context_size"] == 50
    assert final_payload["req_id"] == ["req-1"]
    assert bool(final_payload["stream_finished"].item())
