# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 Thinker preserves multimodal encoding and the llm2tts row ledger."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni as omni
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "profile,capacities,kv_gib,runners",
    [
        ("minicpmo_4_5_turn_mrv2.yaml", [16, 8, 8], 2, [True, True, True]),
        ("minicpmo_4_5_turn_mrv2_h200.yaml", [16, 16, 8], 4, [True, True, True]),
        ("minicpmo_4_5_duplex_mrv2.yaml", [16, 16, 16], 4, [True, True, True]),
    ],
)
def test_mrv2_profile_retains_full_thinker_handoff(profile, capacities, kv_gib, runners, monkeypatch):
    from pathlib import Path

    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy
    from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
    from vllm_omni.platforms import current_omni_platform

    # merge_pipeline_deploy resolves the platform again; test the CUDA profile
    # consistently even when this CPU test runs on a ROCm host.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    deploy = Path(__file__).resolve().parents[4] / "vllm_omni/deploy" / profile
    config = _apply_platform_overrides(load_deploy_config(deploy), platform="cuda")
    stages = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == runners
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [False, True, True]
    assert [s.yaml_engine_args["max_num_seqs"] for s in stages] == capacities
    assert stages[1].yaml_engine_args["kv_cache_memory_bytes"] == kv_gib * 1024**3


def _model(mocker, *, v2=True, session="turn", async_chunk=False):
    thinker = torch.nn.Module()
    thinker.make_empty_intermediate_tensors = lambda: None
    mocker.patch.object(omni, "init_vllm_registered_model", return_value=thinker)
    mocker.patch("vllm_omni.model_executor.models.minicpmo_4_5.duplex.compat.patch_minicpmo_remote_config")
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            multimodal_config=None,
            model_stage="llm",
            use_v2_model_runner=v2,
            session_mode=session,
            async_chunk=async_chunk,
        )
    )
    return omni.MiniCPMO45OmniForConditionalGeneration(vllm_config=config)


def test_mrv2_thinker_keeps_native_multimodal_embeddings_and_raw_ids(mocker):
    model = _model(mocker)
    batch = SimpleNamespace(input_ids=torch.tensor([11, 22, 33]), num_tokens_after_padding=3)
    embeddings = torch.randn(3, 4)
    runner = SimpleNamespace(
        model=model,
        supports_mm_inputs=True,
        is_first_pp_rank=True,
        model_state=SimpleNamespace(dummy_inputs_embeds=lambda _: embeddings),
    )
    ids, embeds, _ = OmniGPUModelRunner._prepare_mm_inputs(runner, None, batch, dummy_run=True)
    assert ids is batch.input_ids
    assert embeds is embeddings
    hidden = torch.randn(3, 4)
    model.thinker.forward = mocker.Mock(return_value=hidden)
    result = model(ids, torch.tensor([5, 6, 7]), inputs_embeds=embeds)
    assert result is hidden
    torch.testing.assert_close(model.thinker.forward.call_args.kwargs["inputs_embeds"], embeddings)


def test_row_ledger_uses_live_batch_after_replay(mocker):
    model = _model(mocker)
    # Chunked prefill (two tokens), decode (one token), and graph padding.
    batch = SimpleNamespace(
        req_ids=["a", "b"],
        input_ids=torch.tensor([11, 12, 41, 0]),
        positions=torch.tensor([6, 7, 20, 0]),
    )
    hidden = torch.randn(4, 8)
    for token in [41, 53]:
        batch.input_ids[2] = token
        batch.positions[2] += 1
        out = model.make_omni_output_mrv2(
            hidden,
            input_batch=batch,
            req_states=None,
            model_intermediate_buffer=[{}, {}],
        )
        assert out.text_hidden_states is hidden
        assert out.multimodal_outputs["latent"] is hidden
        torch.testing.assert_close(out.multimodal_outputs["latent_input_ids"], batch.input_ids[:, None])
        torch.testing.assert_close(out.multimodal_outputs["latent_positions"], batch.positions[:, None])


def test_mrv2_thinker_duplex_output_and_prompt_rows(mocker):
    model = _model(mocker, session="duplex")
    assert model.has_preprocess is True
    batch = SimpleNamespace(
        req_ids=["a", "b"],
        input_ids=torch.tensor([101, 102]),
        positions=torch.tensor([0, 1]),
    )
    hidden = torch.randn(2, 8)
    buffers = [
        {
            "duplex": {
                "duplex_prompt_token_ids": [1, 2, 3],
                "special_token_ids": {"tts_bos_token_id": 151703, "turn_start_token_id": 151644},
            }
        },
        {},
    ]
    out = model.make_omni_output_mrv2(
        hidden,
        input_batch=batch,
        req_states=None,
        model_intermediate_buffer=buffers,
    )
    assert out.text_hidden_states is hidden
    assert out.multimodal_outputs["latent"] is hidden
    assert out.multimodal_outputs["duplex_prompt_token_ids"] == [[1, 2, 3], None]
    assert "tts_bos_token_id" in out.multimodal_outputs["meta"]
    assert out.multimodal_outputs["meta"]["tts_bos_token_id"][0].item() == 151703
    assert out.multimodal_outputs["meta"]["tts_bos_token_id"][1] is None
    # Host-only metadata: a device tensor would cost a blocking H2D per row and key.
    assert out.multimodal_outputs["meta"]["tts_bos_token_id"][0].device.type == "cpu"


def test_mrv2_turn_thinker_retains_native_multimodal_path(mocker):
    assert _model(mocker, session="turn").has_preprocess is False


def test_mrv2_thinker_custom_sampler_and_lifecycle(mocker):
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2_sampling import (
        MiniCPMO45DuplexSampler,
    )
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    model = _model(mocker, session="duplex")
    base_sampler = mocker.MagicMock()
    sampler, _ = model.mrv2_custom_sampler(base_sampler)
    assert isinstance(sampler, MiniCPMO45DuplexSampler)
    assert model._mrv2_duplex_sampler is sampler

    state_mock = mocker.MagicMock(spec=OmniModelState)
    state_mock.model = model
    resolved = OmniModelState.custom_sampler(state_mock, base_sampler)
    assert resolved is not None
    assert isinstance(resolved[0], MiniCPMO45DuplexSampler)

    model.preprocess(
        torch.tensor([1]),
        torch.randn(1, 4),
        req_id="req-1",
        duplex={"data_plane": True, "session_id": "s1", "epoch": 0, "seq": 1},
    )
    assert "req-1" in model._mrv2_sampling_infos

    model.on_requests_finished({"req-1"})
    assert "req-1" not in model._mrv2_sampling_infos
