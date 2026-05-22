# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.glm4_voice import ar_to_decoder_async_chunk


_AUDIO_OFFSET = 151552


def _tm(
    *,
    chunk_frames: int | None = 50,
    chunk_sizes: list[int] | None = None,
    audio_offset: int | None = None,
):
    extra = {}
    if chunk_sizes is not None:
        extra["codec_chunk_frames_list"] = chunk_sizes
    elif chunk_frames is not None:
        extra["codec_chunk_frames"] = chunk_frames
    if audio_offset is not None:
        extra["audio_offset"] = audio_offset
    return SimpleNamespace(
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )


def _req(tokens: list[int], *, finished: bool = False):
    return SimpleNamespace(
        external_req_id="req-0",
        output_token_ids=tokens,
        is_finished=lambda: finished,
    )


def test_glm4_voice_async_chunk_holds_until_configured_chunk_size():
    transfer_manager = _tm(chunk_frames=50)

    pending = ar_to_decoder_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req([_AUDIO_OFFSET + i for i in range(49)]),
    )
    assert pending is None

    payload = ar_to_decoder_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req([_AUDIO_OFFSET + i for i in range(50)]),
    )

    assert payload is not None
    assert payload.codes.audio.tolist() == list(range(50))
    assert payload.meta.finished.item() is False


def test_glm4_voice_async_chunk_flushes_short_tail_on_finish():
    transfer_manager = _tm(chunk_frames=50)

    payload = ar_to_decoder_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req([_AUDIO_OFFSET + 7, _AUDIO_OFFSET + 8], finished=True),
        is_finished=True,
    )

    assert payload is not None
    assert payload.codes.audio.tolist() == [7, 8]
    assert payload.meta.finished.item() is True


def test_glm4_voice_async_chunk_uses_official_progressive_chunk_sizes():
    transfer_manager = _tm(chunk_sizes=[25, 50, 100, 150, 200])

    first = ar_to_decoder_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req([_AUDIO_OFFSET + i for i in range(25)]),
    )
    assert first is not None
    assert first.codes.audio.tolist() == list(range(25))

    pending = ar_to_decoder_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req([_AUDIO_OFFSET + i for i in range(74)]),
    )
    assert pending is None

    second = ar_to_decoder_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req([_AUDIO_OFFSET + i for i in range(75)]),
    )
    assert second is not None
    assert second.codes.audio.tolist() == list(range(25, 75))


def test_glm4_voice_async_chunk_uses_configured_audio_offset():
    audio_offset = 200000
    transfer_manager = _tm(chunk_frames=2, audio_offset=audio_offset)

    payload = ar_to_decoder_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_req([_AUDIO_OFFSET + 5, audio_offset + 7, audio_offset + 8]),
    )

    assert payload is not None
    assert payload.codes.audio.tolist() == [7, 8]
