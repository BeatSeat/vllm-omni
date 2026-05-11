import importlib
import os
import threading
import time
import types

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_init_utils import LogicalStageInitPlan, ReplicaInitPlan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_llm_metadata(
    stage_id: int,
    *,
    replica_id: int = 0,
    final_output: bool = False,
    final_output_type: str | None = None,
    is_comprehension: bool = False,
    model_stage: str | None = None,
    hf_overrides: dict[str, object] | None = None,
):
    return types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="llm",
        runtime_cfg={},
        prompt_expand_func=None,
        final_output=final_output,
        final_output_type=final_output_type,
        default_sampling_params=types.SimpleNamespace(name=f"sp-{stage_id}-{replica_id}"),
        custom_process_input_func=None,
        engine_input_source=[] if stage_id == 0 else [stage_id - 1],
        engine_output_type="token_ids",
        model_stage=model_stage,
        hf_overrides=hf_overrides,
        replica_id=replica_id,
        is_comprehension=is_comprehension,
    )


def _make_diffusion_metadata(stage_id: int, *, replica_id: int = 0, final_output_type: str = "image"):
    return types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="diffusion",
        runtime_cfg={"devices": str(replica_id)},
        prompt_expand_func=None,
        final_output=True,
        final_output_type=final_output_type,
        default_sampling_params=types.SimpleNamespace(name=f"dsp-{stage_id}-{replica_id}"),
        custom_process_input_func=None,
        engine_input_source=[],
        model_stage=None,
        hf_overrides=None,
        cfg_kv_collect_func=None,
        replica_id=replica_id,
    )


def _make_llm_plan(
    stage_idx: int,
    *,
    configured_stage_id: int,
    vllm_config: object,
    num_replicas: int = 1,
    final_output: bool = False,
    final_output_type: str | None = None,
    is_comprehension: bool = False,
    model_stage: str | None = None,
    hf_overrides: dict[str, object] | None = None,
):
    replicas: list[ReplicaInitPlan] = []
    for replica_id in range(num_replicas):
        stage_cfg = types.SimpleNamespace(
            stage_id=configured_stage_id,
            stage_type="llm",
            runtime=types.SimpleNamespace(devices=str(replica_id)),
            engine_args={},
        )
        replicas.append(
            ReplicaInitPlan(
                replica_id=replica_id,
                num_replicas=num_replicas,
                launch_mode="local",
                stage_cfg=stage_cfg,
                metadata=_make_llm_metadata(
                    configured_stage_id,
                    replica_id=replica_id,
                    final_output=final_output,
                    final_output_type=final_output_type,
                    is_comprehension=is_comprehension and replica_id == 0,
                    model_stage=model_stage,
                    hf_overrides=hf_overrides,
                ),
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
                stage_vllm_config=vllm_config,
                executor_class=object,
            )
        )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        configured_stage_id=configured_stage_id,
        replicas=replicas,
    )


def _make_diffusion_plan(
    stage_idx: int,
    *,
    configured_stage_id: int,
    num_replicas: int = 1,
):
    replicas: list[ReplicaInitPlan] = []
    for replica_id in range(num_replicas):
        stage_cfg = types.SimpleNamespace(
            stage_id=configured_stage_id,
            stage_type="diffusion",
            runtime=types.SimpleNamespace(devices=str(replica_id)),
            engine_args={},
        )
        replicas.append(
            ReplicaInitPlan(
                replica_id=replica_id,
                num_replicas=num_replicas,
                launch_mode="local",
                stage_cfg=stage_cfg,
                metadata=_make_diffusion_metadata(configured_stage_id, replica_id=replica_id),
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
            )
        )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        configured_stage_id=configured_stage_id,
        replicas=replicas,
    )



def test_stage_engine_core_client_module_reload_keeps_forward_refs_deferred():
    """Regression test for forward references in make_async_mp_client."""
    import vllm_omni.engine.stage_engine_core_client as client_mod

    importlib.reload(client_mod)

    assert client_mod.StageEngineCoreClientBase.make_async_mp_client.__annotations__["return"] == (
        "StageEngineCoreClient | DPLBStageEngineCoreClient"
    )


def test_initialize_stages_restores_device_visibility_after_diffusion_init(monkeypatch):
    """Regression test for stage device env leakage across stage init.

    Diffusion init mutates process-level CUDA visibility. Ensure AsyncOmniEngine
    restores the previous value after diffusion stage setup.
    """
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.model = "dummy-model"
    engine.config_path = "dummy-config"
    engine.num_stages = 1
    engine.async_chunk = False
    engine.diffusion_batch_size = 1
    engine.single_stage_mode = False
    engine._single_stage_id_filter = None
    engine._omni_master_server = None
    engine.stage_configs = [types.SimpleNamespace(stage_id=0, stage_type="diffusion")]

    env_var = current_omni_platform.device_control_env_var
    old_env = os.environ.get(env_var)
    os.environ[env_var] = "0,1"

    diffusion_client = types.SimpleNamespace(is_comprehension=False)

    metadata = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        runtime_cfg={"devices": "1"},
        prompt_expand_func=None,
    )

    monkeypatch.setattr(engine_mod, "prepare_engine_environment", lambda: None)
    monkeypatch.setattr(engine_mod, "load_omni_transfer_config_for_model", lambda *_: None)
    monkeypatch.setattr(engine_mod, "extract_stage_metadata", lambda _cfg: metadata)
    monkeypatch.setattr(engine_mod, "get_stage_connector_spec", lambda **_: {})
    monkeypatch.setattr(engine_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))

    def _fake_setup_stage_devices(_stage_id, _runtime_cfg):
        # Simulate diffusion setup mutating process-global visibility.
        current_omni_platform.set_device_control_env_var("1")

    monkeypatch.setattr(engine_mod, "setup_stage_devices", _fake_setup_stage_devices)
    monkeypatch.setattr(engine_mod, "inject_kv_stage_info", lambda *_: None)
    monkeypatch.setattr(engine_mod, "initialize_diffusion_stage", lambda *_, **__: diffusion_client)
    monkeypatch.setattr(
        engine_mod,
        "finalize_initialized_stages",
        lambda stage_clients, _input_processor: (
            stage_clients,
            [types.SimpleNamespace()],
            [{"final_output_type": "image"}],
        ),
    )

    try:
        engine._initialize_stages(stage_init_timeout=1)
        assert os.environ.get(env_var) == "0,1"
    finally:
        if old_env is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old_env


def test_initialize_stages_passes_stage_init_timeout_to_diffusion_handshake(monkeypatch):
    """Regression test for stage_init_timeout passing to complete_diffusion_handshake
    in the diffusion stage path.
    """
    import vllm_omni.diffusion.data as diffusion_data_mod
    import vllm_omni.diffusion.stage_diffusion_client as client_mod
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.log_stats = False
    engine.model = "dummy-model"
    engine.config_path = "dummy-config"
    engine.num_stages = 2
    engine.async_chunk = False
    engine.diffusion_batch_size = 1
    engine.single_stage_mode = False
    engine._omni_master_server = None
    engine.stage_configs = [types.SimpleNamespace(stage_id=0, stage_type="diffusion", engine_args={})]

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    cfg1 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [
        _make_llm_plan(
            0,
            configured_stage_id=0,
            vllm_config=cfg0,
            num_replicas=2,
            is_comprehension=True,
            model_stage="glm_tts",
            hf_overrides={"min_token_text_ratio": 2.0},
        ),
        _make_llm_plan(1, configured_stage_id=1, vllm_config=cfg1, final_output=True),
    ]

    stage0_client_r0 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=True,
        final_output=False,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp0"),
    )
    stage0_client_r1 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=False,
        final_output=False,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp0r1"),
    )
    stage1_client_r0 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=False,
        final_output=True,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp1"),
    )
    initialized_clients = {
        0: [stage0_client_r0, stage0_client_r1],
        1: [stage1_client_r0],
    }

    stage0_output_processor = object()
    stage1_output_processor = object()
    top_level_input_processor = object()

    monkeypatch.setattr(engine_mod, "prepare_engine_environment", lambda: None)
    monkeypatch.setattr(engine_mod, "load_omni_transfer_config_for_model", lambda *_: None)
    monkeypatch.setattr(engine_mod, "compute_replica_layout", lambda _cfgs: ([2, 1], {}))
    monkeypatch.setattr(engine, "_build_logical_stage_init_plans", lambda *_: (stage_plans, None))
    monkeypatch.setattr(engine, "_initialize_stage_replicas", lambda *_: initialized_clients)
    monkeypatch.setattr(
        engine_mod,
        "build_llm_stage_output_processor",
        lambda plan, _cfg: stage0_output_processor if plan.stage_idx == 0 else stage1_output_processor,
    )
    monkeypatch.setattr(engine_mod, "build_stage0_input_processor", lambda _cfg: top_level_input_processor)

    engine._initialize_stages(stage_init_timeout=1)

    assert len(engine.stage_pools) == 2
    assert engine.input_processor is top_level_input_processor
    assert engine.stage_clients == [stage0_client_r0, stage1_client_r0]
    assert engine.stage_vllm_configs == [cfg0, cfg1]
    assert engine.output_processors == [stage0_output_processor, stage1_output_processor]
    assert engine.default_sampling_params_list == [
        stage0_client_r0.default_sampling_params,
        stage1_client_r0.default_sampling_params,
    ]
    assert engine.stage_metadata == [
        {
            "final_output": False,
            "final_output_type": None,
            "stage_type": "llm",
            "model_stage": "glm_tts",
            "hf_overrides": {"min_token_text_ratio": 2.0},
        },
        {
            "final_output": True,
            "final_output_type": None,
            "stage_type": "llm",
            "model_stage": None,
            "hf_overrides": None,
        },
    ]


def test_build_logical_stage_init_plans_applies_replica_device_splits(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    engine.model = "dummy-model"
    engine.async_chunk = False
    engine.single_stage_mode = False
    engine._single_stage_id_filter = None
    engine.stage_configs = [
        types.SimpleNamespace(stage_id=0, stage_type="llm", engine_args={}, runtime=types.SimpleNamespace(devices="0")),
        types.SimpleNamespace(
            stage_id=1, stage_type="llm", engine_args={}, runtime=types.SimpleNamespace(devices="1,2,3")
        ),
    ]

    metadata_by_stage = {
        0: _make_llm_metadata(0),
        1: _make_llm_metadata(1),
    }

    monkeypatch.setattr(
        engine_mod,
        "extract_stage_metadata",
        lambda cfg: types.SimpleNamespace(**metadata_by_stage[cfg.stage_id].__dict__),
    )
    monkeypatch.setattr(engine_mod, "get_stage_connector_spec", lambda **_: {})
    monkeypatch.setattr(engine_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))
    monkeypatch.setattr(engine_mod, "build_engine_args_dict", lambda *_, **__: {})
    monkeypatch.setattr(
        engine_mod,
        "build_vllm_config",
        lambda stage_cfg, *_args, **_kwargs: (types.SimpleNamespace(tag=f"cfg-{stage_cfg.stage_id}"), object),
    )

    stage_plans, prompt_expand_func = engine._build_logical_stage_init_plans(
        omni_transfer_config=None,
        replicas_per_stage=[1, 3],
        replica_devices_map={1: ["1", "2", "3"]},
    )

    assert prompt_expand_func is None
    assert [plan.configured_stage_id for plan in stage_plans] == [0, 1]
    assert [replica.stage_cfg.runtime.devices for replica in stage_plans[1].replicas] == ["1", "2", "3"]
    assert [replica.replica_id for replica in stage_plans[1].replicas] == [0, 1, 2]
    assert all(replica.num_replicas == 3 for replica in stage_plans[1].replicas)


def test_initialize_stage_replicas_collects_results_by_stage_and_replica_id(monkeypatch):
    engine = object.__new__(AsyncOmniEngine)

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    cfg1 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [
        _make_llm_plan(0, configured_stage_id=0, vllm_config=cfg0, num_replicas=2),
        _make_llm_plan(1, configured_stage_id=1, vllm_config=cfg1, num_replicas=2),
    ]

    clients = {
        (0, 0): types.SimpleNamespace(name="stage0-replica0"),
        (0, 1): types.SimpleNamespace(name="stage0-replica1"),
        (1, 0): types.SimpleNamespace(name="stage1-replica0"),
        (1, 1): types.SimpleNamespace(name="stage1-replica1"),
    }

    def _initialize_replica(plan, _stage_init_timeout, _stage_launch_lock):
        time.sleep(0.02 * (3 - plan.metadata.stage_id - plan.replica_id))
        return clients[(plan.metadata.stage_id, plan.replica_id)]

    monkeypatch.setattr(engine, "_initialize_replica", _initialize_replica)

    initialized_clients = engine._initialize_stage_replicas(stage_plans, stage_init_timeout=123)

    assert initialized_clients == {
        0: [clients[(0, 0)], clients[(0, 1)]],
        1: [clients[(1, 0)], clients[(1, 1)]],
    }


def test_initialize_stages_cleans_up_successful_replicas_after_partial_multi_replica_failure(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    engine.model = "dummy-model"
    engine.config_path = "dummy-config"
    engine.num_stages = 1
    engine.async_chunk = False
    engine.diffusion_batch_size = 1
    engine.single_stage_mode = False
    engine._single_stage_id_filter = None
    engine._omni_master_server = None
    engine.stage_configs = [types.SimpleNamespace()]

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [_make_llm_plan(0, configured_stage_id=0, vllm_config=cfg0, num_replicas=2)]
    initialized_client = types.SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(engine_mod, "prepare_engine_environment", lambda: None)
    monkeypatch.setattr(engine_mod, "load_omni_transfer_config_for_model", lambda *_: None)
    monkeypatch.setattr(engine_mod, "compute_replica_layout", lambda _cfgs: ([2], {}))
    monkeypatch.setattr(engine, "_build_logical_stage_init_plans", lambda *_: (stage_plans, None))

    def _initialize_replica(plan, _stage_init_timeout, _stage_launch_lock):
        if plan.replica_id == 0:
            return initialized_client
        time.sleep(0.05)
        raise RuntimeError("replica launch failed")

    monkeypatch.setattr(engine, "_initialize_replica", _initialize_replica)

    captured_cleanup: list[list[object]] = []

    def _capture_shutdown(clients):
        captured_cleanup.append(list(clients))

    monkeypatch.setattr(engine, "_shutdown_initialized_clients", _capture_shutdown)

    with pytest.raises(RuntimeError, match="replica launch failed"):
        engine._initialize_stages(stage_init_timeout=1)

    assert captured_cleanup == [[initialized_client]]


def test_initialize_stages_cleans_up_late_successful_replicas_after_early_multi_replica_failure(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    engine.model = "dummy-model"
    engine.config_path = "dummy-config"
    engine.num_stages = 1
    engine.async_chunk = False
    engine.diffusion_batch_size = 1
    engine.single_stage_mode = False
    engine._single_stage_id_filter = None
    engine._omni_master_server = None
    engine.stage_configs = [types.SimpleNamespace()]

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [_make_llm_plan(0, configured_stage_id=0, vllm_config=cfg0, num_replicas=2)]
    initialized_client = types.SimpleNamespace(shutdown=lambda: None)
    replica1_started = threading.Event()

    monkeypatch.setattr(engine_mod, "prepare_engine_environment", lambda: None)
    monkeypatch.setattr(engine_mod, "load_omni_transfer_config_for_model", lambda *_: None)
    monkeypatch.setattr(engine_mod, "compute_replica_layout", lambda _cfgs: ([2], {}))
    monkeypatch.setattr(engine, "_build_logical_stage_init_plans", lambda *_: (stage_plans, None))

    def _initialize_replica(plan, _stage_init_timeout, _stage_launch_lock):
        if plan.replica_id == 0:
            assert replica1_started.wait(timeout=1)
            raise RuntimeError("replica launch failed")
        replica1_started.set()
        time.sleep(0.05)
        return initialized_client

    monkeypatch.setattr(engine, "_initialize_replica", _initialize_replica)

    captured_cleanup: list[list[object]] = []

    def _capture_shutdown(clients):
        captured_cleanup.append(list(clients))

    monkeypatch.setattr(engine, "_shutdown_initialized_clients", _capture_shutdown)

    with pytest.raises(RuntimeError, match="replica launch failed"):
        engine._initialize_stages(stage_init_timeout=1)

    assert captured_cleanup == [[initialized_client]]


def test_launch_llm_stage_passes_stage_init_timeout_to_complete_stage_handshake(monkeypatch):
    """Regression test for stage_init_timeout reaching complete_stage_handshake
    in the LLM stage path.
    """
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.log_stats = False
    engine.model = "dummy-model"
    engine.single_stage_mode = False
    engine._omni_master_server = None
    engine.stage_configs = []

    metadata = types.SimpleNamespace(stage_id=0, runtime_cfg={"devices": "0"})
    fake_vllm_config = types.SimpleNamespace()
    fake_addresses = types.SimpleNamespace()
    fake_proc = types.SimpleNamespace()

    captured_timeout = None

    device_env_var = current_omni_platform.device_control_env_var
    prev_device_env = os.environ.get(device_env_var)
    os.environ[device_env_var] = "0"

    monkeypatch.setattr(engine_mod, "setup_stage_devices", lambda *_: None)
    monkeypatch.setattr(engine_mod, "build_engine_args_dict", lambda *_, **__: {})
    monkeypatch.setattr(engine_mod, "build_vllm_config", lambda *_, **__: (fake_vllm_config, object))
    monkeypatch.setattr(engine_mod, "acquire_device_locks", lambda *_: [])
    monkeypatch.setattr(
        engine_mod,
        "spawn_stage_core",
        lambda **_: (fake_addresses, fake_proc, "ipc://handshake"),
    )

    def _capture_stage_timeout(_proc, _handshake_addr, _addresses, _vllm_cfg, handshake_timeout):
        nonlocal captured_timeout
        captured_timeout = handshake_timeout

    monkeypatch.setattr(engine_mod, "complete_stage_handshake", _capture_stage_timeout)

    try:
        engine._launch_llm_stage(
            stage_cfg=types.SimpleNamespace(engine_args={}),
            metadata=metadata,
            stage_connector_spec={},
            stage_init_timeout=302,
            llm_stage_launch_lock=threading.Lock(),
        )
    finally:
        if prev_device_env is None:
            os.environ.pop(device_env_var, None)
        else:
            os.environ[device_env_var] = prev_device_env

    assert captured_timeout == 302


def test_launch_llm_stage_releases_launch_lock_before_complete_stage_handshake(monkeypatch):
    """Regression test for parallel LLM stage startup during handshake wait."""
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.log_stats = False
    engine.model = "dummy-model"
    engine.single_stage_mode = False
    engine._omni_master_server = None
    engine.stage_configs = []

    fake_vllm_config = types.SimpleNamespace()
    fake_addresses = types.SimpleNamespace()
    shared_launch_lock = threading.Lock()
    counter_lock = threading.Lock()
    first_handshake_started = threading.Event()
    second_stage_spawned = threading.Event()
    allow_first_handshake_to_finish = threading.Event()
    launch_errors: list[BaseException] = []
    spawn_count = 0

    device_env_var = current_omni_platform.device_control_env_var
    prev_device_env = os.environ.get(device_env_var)
    os.environ[device_env_var] = "0"

    monkeypatch.setattr(engine_mod, "setup_stage_devices", lambda *_: None)
    monkeypatch.setattr(engine_mod, "build_engine_args_dict", lambda *_, **__: {})
    monkeypatch.setattr(engine_mod, "build_vllm_config", lambda *_, **__: (fake_vllm_config, object))
    monkeypatch.setattr(engine_mod, "acquire_device_locks", lambda *_: [])

    def _spawn_stage_core(**_):
        nonlocal spawn_count
        with counter_lock:
            spawn_count += 1
            call_idx = spawn_count
        if call_idx == 2:
            second_stage_spawned.set()
        return fake_addresses, types.SimpleNamespace(), f"ipc://handshake-{call_idx}"

    def _complete_stage_handshake(_proc, handshake_address, _addresses, _vllm_cfg, _timeout):
        if handshake_address == "ipc://handshake-1":
            first_handshake_started.set()
            assert second_stage_spawned.wait(timeout=1), (
                "second stage did not reach spawn_stage_core while first stage waited in handshake"
            )
            assert allow_first_handshake_to_finish.wait(timeout=1), (
                "second stage did not enter handshake while first stage was still waiting"
            )
        else:
            allow_first_handshake_to_finish.set()

    monkeypatch.setattr(engine_mod, "spawn_stage_core", _spawn_stage_core)
    monkeypatch.setattr(engine_mod, "complete_stage_handshake", _complete_stage_handshake)

    def _launch_stage(stage_id: int) -> None:
        metadata = types.SimpleNamespace(stage_id=stage_id, runtime_cfg={"devices": str(stage_id)})
        try:
            engine._launch_llm_stage(
                stage_cfg=types.SimpleNamespace(engine_args={}),
                metadata=metadata,
                stage_connector_spec={},
                stage_init_timeout=302,
                llm_stage_launch_lock=shared_launch_lock,
            )
        except BaseException as exc:  # pragma: no cover - surfaced through assertion below
            launch_errors.append(exc)

    try:
        first_thread = threading.Thread(target=_launch_stage, args=(0,))
        first_thread.start()
        assert first_handshake_started.wait(timeout=1), "first stage never entered handshake"

        second_thread = threading.Thread(target=_launch_stage, args=(1,))
        second_thread.start()

        first_thread.join(timeout=3)
        second_thread.join(timeout=3)
    finally:
        if prev_device_env is None:
            os.environ.pop(device_env_var, None)
        else:
            os.environ[device_env_var] = prev_device_env

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_stage_spawned.is_set()
    assert not launch_errors


def test_attach_llm_stage_uses_omni_input_preprocessor(monkeypatch):
    """Regression test for GLM-Image t2i preprocessing path.

    Stage-0 InputProcessor must use OmniInputPreprocessor so text prompts with
    mm_processor_kwargs go through multimodal preprocessing.
    """
    import vllm_omni.engine.async_omni_engine as engine_mod

    class DummyStageEngineCoreClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def shutdown(self):
            return None

    class DummyOutputProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyInputProcessor:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.renderer = object()
            self.input_preprocessor = object()

    class DummyOmniInputPreprocessor:
        def __init__(self, vllm_config, renderer=None):
            self.vllm_config = vllm_config
            self.renderer = renderer

    monkeypatch.setattr(
        engine_mod.StageEngineCoreClientBase,
        "make_async_mp_client",
        staticmethod(lambda **kwargs: DummyStageEngineCoreClient(**kwargs)),
    )
    monkeypatch.setattr(engine_mod, "MultimodalOutputProcessor", DummyOutputProcessor)
    monkeypatch.setattr(engine_mod, "InputProcessor", DummyInputProcessor)
    monkeypatch.setattr(engine_mod, "OmniInputPreprocessor", DummyOmniInputPreprocessor)

    started = types.SimpleNamespace(
        stage_id=0,
        metadata=types.SimpleNamespace(stage_id=0, engine_output_type="token_ids"),
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(skip_tokenizer_init=True)),
        executor_class=object,
        engine_manager=object(),
        coordinator=object(),
        proc=None,
        addresses=types.SimpleNamespace(
            inputs=["inproc://input"],
            outputs=["inproc://output"],
            frontend_stats_publish_address=None,
        ),
    )

    engine = object.__new__(AsyncOmniEngine)

    _stage_client, _out_proc, _vllm_cfg, input_processor = engine._attach_llm_stage(started)

    assert input_processor is not None
    assert isinstance(input_processor.input_preprocessor, DummyOmniInputPreprocessor)
    assert input_processor.input_preprocessor.renderer is input_processor.renderer


def test_inject_kv_stage_info_infers_sender_tp_topology():
    from vllm_omni.engine.stage_init_utils import inject_kv_stage_info

    stage0 = types.SimpleNamespace(
        stage_id=0,
        engine_args={
            "tensor_parallel_size": 4,
            "omni_kv_config": {
                "need_send_cache": True,
                "omni_from_stage": "0",
                "omni_to_stage": "1",
            },
        },
        engine_input_source=[],
    )
    stage1 = types.SimpleNamespace(
        stage_id=1,
        engine_args={
            "parallel_config": {
                "tensor_parallel_size": 2,
                "cfg_parallel_size": 1,
            },
            "omni_kv_config": {"need_recv_cache": True},
        },
        engine_input_source=[0],
    )

    inject_kv_stage_info(stage0, 0, [stage0, stage1])

    assert stage0.engine_args["omni_kv_config"]["stage_id"] == 0
    assert stage0.engine_args["omni_kv_config"]["rank_mapping"] == {"from_tp": 4, "to_tp": 2}


def test_inject_kv_stage_info_infers_receiver_tp_topology():
    from vllm_omni.engine.stage_init_utils import inject_kv_stage_info

    stage0 = types.SimpleNamespace(
        stage_id=0,
        engine_args={
            "tensor_parallel_size": 4,
            "omni_kv_config": {"need_send_cache": True},
        },
        engine_input_source=[],
    )
    stage1 = types.SimpleNamespace(
        stage_id=1,
        engine_args={
            "parallel_config": {
                "tensor_parallel_size": 2,
                "cfg_parallel_size": 1,
            },
            "omni_kv_config": {
                "need_recv_cache": True,
                "omni_from_stage": "0",
                "omni_to_stage": "1",
            },
        },
        engine_input_source=[0],
    )

    inject_kv_stage_info(stage1, 1, [stage0, stage1])

    assert stage1.engine_args["omni_kv_config"]["stage_id"] == 1
    assert stage1.engine_args["omni_kv_config"]["engine_input_source"] == [0]
    assert stage1.engine_args["omni_kv_config"]["rank_mapping"] == {"from_tp": 4, "to_tp": 2}
