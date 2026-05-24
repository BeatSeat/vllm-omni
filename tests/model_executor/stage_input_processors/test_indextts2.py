# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.indextts2 import (
    _strip_stop_token,
    talker2s2mel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage_output(mm: dict, *, finished: bool = True):
    return SimpleNamespace(
        finished=finished,
        outputs=[SimpleNamespace(multimodal_output=mm)],
    )


def test_strip_stop_token_pads_to_longest_valid_length():
    codes = torch.tensor(
        [
            [10, 11, 8193, 99],
            [20, 21, 22, 23],
        ],
        dtype=torch.long,
    )
    latent = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    clean_codes, clean_latent, code_lens = _strip_stop_token(codes, latent)

    assert code_lens.tolist() == [2, 4]
    assert clean_codes.tolist() == [
        [10, 11, 8193, 8193],
        [20, 21, 22, 23],
    ]
    assert clean_latent.shape == (2, 4, 3)
    assert torch.equal(clean_latent[0, :2], latent[0, :2])
    assert torch.equal(clean_latent[1], latent[1])


def test_talker2s2mel_skips_unfinished_and_missing_payloads():
    valid_mm = {
        "codes": {"mel": torch.tensor([1, 2])},
        "hidden_states": {"latent": torch.ones(2, 4)},
    }
    missing_codes = {"hidden_states": {"latent": torch.ones(2, 4)}}

    prompts = talker2s2mel(
        [
            _stage_output(valid_mm, finished=False),
            _stage_output(missing_codes, finished=True),
            _stage_output(valid_mm, finished=True),
        ]
    )

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [0]


def test_talker2s2mel_trims_stop_token_and_transfers_metadata_on_cpu():
    mel_codes = torch.tensor([7, 8, 8193, 9], dtype=torch.long)
    latent = torch.arange(4 * 5, dtype=torch.float16).reshape(4, 5)
    s_ref = torch.ones(1, 3, 1024, dtype=torch.float16)
    ref_mel = torch.ones(1, 80, 4, dtype=torch.float16)
    style = torch.ones(192, dtype=torch.float16)
    mm = {
        "codes": {"mel": mel_codes},
        "hidden_states": {"latent": latent},
        "meta": {
            "S_ref": s_ref,
            "ref_mel": ref_mel,
            "style": style,
        },
    }

    prompts = talker2s2mel([_stage_output(mm)])

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info["mel_codes"].device.type == "cpu"
    assert info["latent"].device.type == "cpu"
    assert info["S_ref"].device.type == "cpu"
    assert info["ref_mel"].device.type == "cpu"
    assert info["style"].device.type == "cpu"
    assert info["mel_codes"].dtype == torch.long
    assert info["latent"].dtype == torch.float32
    assert info["code_lens"].tolist() == [2]
    assert info["mel_codes"].tolist() == [[7, 8]]
    assert info["latent"].shape == (1, 2, 5)
    assert info["S_ref"].dtype == torch.float32
    assert info["ref_mel"].dtype == torch.float32
    assert info["style"].dtype == torch.float32
