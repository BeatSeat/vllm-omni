# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The duplex MRv2 deploy profile resolves to the config keys the model relies on.

Full-duplex Model Runner V2 serving is selected by
``minicpmo_4_5_duplex_mrv2.yaml``. Its H200 throughput numbers (Stage 1
``max_num_seqs: 16`` and the CUDA ``kv_cache_memory_bytes: 4 GiB`` overlay) are
inherited from ``minicpmo_4_5.yaml`` through ``base_config``, so that one profile
is the canonical duplex-MRv2 entry point. A separate ``*_duplex_mrv2_h200.yaml``
overlay would resolve to the exact same effective config: these tests pin the
effective keys and reject such a redundant duplicate (or a malformed profile) so
it fails fast at review time instead of drifting silently.
"""

from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "minicpmo_4_5_duplex_mrv2.yaml"


def _resolve_cuda_stages(monkeypatch):
    # ``merge_pipeline_deploy`` re-resolves the platform from the host; pin
    # "cuda" so the H200/CUDA overlay capacity is validated even when this CPU
    # test runs on a ROCm/NPU CI runner.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform="cuda")
    return config, merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)


def test_duplex_mrv2_profile_requests_duplex_and_v2(monkeypatch) -> None:
    config, stages = _resolve_cuda_stages(monkeypatch)
    assert config.session_mode == "duplex"
    # Thinker (0), Talker (1) and Code2Wav (2) all execute on Model Runner V2;
    # an unset runner (malformed profile) would surface here as ``use_v2_model_runner`` False.
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, True, True]


def test_duplex_mrv2_profile_carries_h200_capacity(monkeypatch) -> None:
    config, stages = _resolve_cuda_stages(monkeypatch)
    # Stage 0 duplex preprocessing stays on the synchronous path while the
    # downstream stages keep streaming chunk transfer.
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [False, True, True]
    # H200 throughput is inherited from ``minicpmo_4_5.yaml``: 16 concurrent
    # sessions on every stage and a 4 GiB Talker (Stage 1) KV budget.
    assert [s.yaml_engine_args["max_num_seqs"] for s in stages] == [16, 16, 16]
    assert stages[1].yaml_engine_args["kv_cache_memory_bytes"] == 4 * 1024**3
    # The Stage 0 audio-encoder CUDA-graph buckets are sized from the duplex
    # session count, so the model depends on this override surviving resolution.
    assert (
        stages[0].yaml_engine_args["hf_overrides"]["duplex_audio_encoder_cuda_graph_batch_sizes_from_sessions"] is True
    )


def test_no_redundant_duplex_mrv2_h200_overlay() -> None:
    # The base profile already resolves the H200 capacity, so a separate
    # ``*_duplex_mrv2_h200.yaml`` is a redundant duplicate; consolidate into the
    # base profile instead of shipping a second no-op overlay.
    deploy_dir = Path(get_deploy_config_path(_DEPLOY)).resolve().parent
    found = sorted(p.name for p in deploy_dir.glob("minicpmo_4_5_duplex_mrv2*.yaml"))
    assert found == [_DEPLOY], (
        "minicpmo_4_5_duplex_mrv2.yaml already inherits the H200 Stage-1 capacity "
        "(16 seqs / 4 GiB KV) from minicpmo_4_5.yaml; a separate *_duplex_mrv2_h200.yaml "
        "overlay is a redundant duplicate that resolves to the same effective config."
    )
