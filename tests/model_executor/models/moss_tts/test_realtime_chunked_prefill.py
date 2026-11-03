# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_talker_preserves_state_across_chunked_prefill() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )

    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 2
    talker.audio_pad_token = 1024
    talker.text_vocab_size = 16
    talker.text_pad_id = 2
    talker._batch_state = None
    talker._batch_state_spans = None

    embedded_audio: list[torch.Tensor | None] = []

    def build_input_embeds(input_ids, audio_codes):
        embedded_audio.append(audio_codes.clone() if audio_codes is not None else None)
        return torch.zeros((input_ids.shape[0], 1))

    generated: list[None] = []

    def generate_frame(*args, **kwargs):
        generated.append(None)
        return torch.ones((1, talker.n_vq), dtype=torch.long)

    talker._build_input_embeds = build_input_embeds
    talker.local_transformer = SimpleNamespace(generate_frame=generate_frame)
    talker.local_lm_heads = []

    ref_codes = torch.arange(10, dtype=torch.long).reshape(5, 2)
    info: dict[str, Any] = {
        "codes": {"ref": ref_codes},
        "ids": {"all": [7, 8]},
        "max_new_frames": [3],
    }
    chunks = [(0, 2, False), (2, 2, False), (4, 1, True)]

    state: dict[str, Any] | None = None
    for computed, length, eligible in chunks:
        _, _, update = talker.preprocess(
            torch.ones(length, dtype=torch.long),
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=computed,
            _omni_prompt_len=5,
            **info,
        )
        info.update(update)
        if state is None:
            state = info["audio_state"]
        else:
            assert info["audio_state"] is state

        output = talker.make_omni_output(
            torch.ones((length, 1)),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, length)],
            request_sample_eligible=[eligible],
        )
        if not eligible:
            assert output.multimodal_outputs == {}
        logits = talker.compute_logits(torch.ones((1, 1)))
        assert logits.argmax().item() == (7 if eligible else talker.text_pad_id)

    assert state is not None
    assert info["ref_offset"] == 5
    assert state["step"] == 1
    assert state["text_cursor"] == 1
    assert state["remaining_text"] == [7, 8]
    assert len(generated) == 1
    assert output.multimodal_outputs["codes"]["audio"][0].shape == (1, 2)
    for actual, expected in zip(embedded_audio, ref_codes.split((2, 2, 1)), strict=True):
        assert actual is not None
        torch.testing.assert_close(actual, expected)


def test_mrv2_postprocess_model_output_wires_sample_eligibility_to_realtime_talker() -> None:
    import numpy as np

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 2
    talker.audio_pad_token = 1024
    talker.text_vocab_size = 16
    talker.text_pad_id = 2
    talker.audio_eos_id = 15
    talker._batch_state = None
    talker._batch_state_spans = None
    talker.local_transformer = SimpleNamespace(
        generate_frame=lambda *args, **kwargs: torch.tensor([[3, 4]], dtype=torch.long)
    )
    talker.local_lm_heads = []

    state = OmniModelState.__new__(OmniModelState)
    state.model = talker
    state.have_multimodal_outputs = True
    state.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs=4)
    # Slot 2 is mid-prefill (computed 2 + scheduled 2 < prompt 5 -> ineligible)
    # Slot 0 is finishing prefill (computed 4 + scheduled 1 >= prompt 5 -> eligible)
    state.intermediate_buffer.buffers[2] = {
        "req_id": "req_mid",
        "audio_state": {"is_stopping": False, "step": 0, "text_cursor": 0, "remaining_text": [9]},
    }
    state.intermediate_buffer.buffers[0] = {
        "req_id": "req_done",
        "audio_state": {"is_stopping": False, "step": 0, "text_cursor": 0, "remaining_text": [10]},
    }

    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
        query_start_loc_np=np.array([0, 2, 3], dtype=np.int32),
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(np=np.array([4, 0, 2, 0], dtype=np.int32)),
        prompt_len=SimpleNamespace(np=np.array([5, 0, 5, 0], dtype=np.int32)),
    )

    hidden = torch.ones((3, 4), dtype=torch.float32)
    text_hidden, mm_outputs = state.postprocess_model_output(hidden, input_batch, req_states)
    assert text_hidden.shape == (3, 4)
    assert "codes" in mm_outputs
    audio_codes = mm_outputs["codes"]["audio"]
    assert len(audio_codes) == 2
    assert audio_codes[0].numel() == 0  # mid-prefill request skipped
    assert torch.equal(audio_codes[1], torch.tensor([[3, 4]], dtype=torch.long))
    assert state.intermediate_buffer.buffers[2]["audio_state"]["step"] == 0
    assert state.intermediate_buffer.buffers[0]["audio_state"]["step"] == 1


def test_realtime_talker_batched_decode_and_local_transformer_parity() -> None:
    import torch.nn as nn
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
        MossTTSLocalTransformerConfig,
    )
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
        MossTTSRealtimeLocalTransformer,
    )
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )

    cfg = MossTTSLocalTransformerConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        audio_vocab_size=32,
        rvq=4,
    )
    with set_current_vllm_config(VllmConfig()):
        local_tf = MossTTSRealtimeLocalTransformer(cfg).eval()
        lm_heads = nn.ModuleList([nn.Linear(64, 32, bias=False) for _ in range(4)])

    # 1. Verify history_tensors vs history_per_codebook parity on generate_frame
    hidden_1 = torch.randn((1, 64))
    acc_hist = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 6, 3, 9]], dtype=torch.long)
    hist_per_cb = [[int(acc_hist[r, cb].item()) for r in range(acc_hist.shape[0])] for cb in range(4)]

    torch.manual_seed(77)
    codes_list_path = local_tf.generate_frame(
        hidden_1,
        lm_heads,
        temperature=0.8,
        top_p=0.6,
        top_k=10,
        do_sample=True,
        repetition_penalty=1.1,
        history_per_codebook=hist_per_cb,
    )
    torch.manual_seed(77)
    codes_tensor_path = local_tf.generate_frame(
        hidden_1,
        lm_heads,
        temperature=0.8,
        top_p=0.6,
        top_k=10,
        do_sample=True,
        repetition_penalty=1.1,
        history_tensors=[acc_hist],
    )
    assert torch.equal(codes_list_path, codes_tensor_path)

    # 2. Verify preprocess_decode_batch_mrv2 matches per-request preprocess
    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    nn.Module.__init__(talker)
    talker.n_vq = 4
    talker.audio_vocab_size = 32
    talker.audio_pad_token = 28
    talker.text_vocab_size = 64
    talker.text_pad_id = 2
    talker.audio_eos_id = 3
    talker._batch_state = None
    talker._batch_state_spans = None
    talker._n_vq_idx = None
    talker._local_cuda_graph_enabled = False
    talker.embed_tokens = nn.ModuleList(
        [nn.Embedding(64, 64)] + [nn.Embedding(32, 64) for _ in range(4)]
    )
    talker._stacked_audio_emb_w = torch.stack(
        [talker.embed_tokens[i + 1].weight.detach() for i in range(4)], dim=0
    )
    talker.local_transformer = local_tf
    talker.local_lm_heads = lm_heads

    input_ids = torch.tensor([5, 12, 19], dtype=torch.long)
    req_infos = [
        {"audio_state": {"is_stopping": False, "step": 2}, "audio_codes": {"current": torch.tensor([1, 2, 3, 4])}},
        {"audio_state": {"is_stopping": False, "step": 1}, "audio_codes": {}},  # missing current -> uses pad
        {"audio_state": {"is_stopping": False, "step": 5}, "audio_codes": {"current": torch.tensor([7, 8, 9, 10])}},
    ]
    _, batch_embeds, _, _, updates = talker.preprocess_decode_batch_mrv2(
        input_ids, torch.zeros((3, 64)), req_infos
    )
    assert len(updates) == 3
    for idx, info in enumerate(req_infos):
        _, single_emb, _ = talker.preprocess(input_ids[idx : idx + 1], None, **info)
        torch.testing.assert_close(batch_embeds[idx : idx + 1], single_emb)


def test_moss_realtime_codec_mrv2_dummy_run_skips_empty_runtime_info() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
        MossTTSCodecDecoder,
    )

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(n_vq=16, model_type="moss_tts_realtime"),
            async_chunk=True,
            use_v2_model_runner=True,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    decoder = MossTTSCodecDecoder(vllm_config=config)
    decoder._codec = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    # 65536 is divisible by n_vq=16; dummy profiling on MRv2 passes [{}]
    dummy_ids = torch.zeros(65536, dtype=torch.long)
    out = decoder.forward(
        input_ids=dummy_ids,
        runtime_additional_information=[{}],
        seq_token_counts=[65536],
    )
    assert len(out.multimodal_outputs["model_outputs"]) == 1
    assert out.multimodal_outputs["model_outputs"][0].numel() == 0
    assert decoder._stream_req_slots == {}


def test_realtime_local_transformer_cuda_graph_parity_and_mixed_batch_stopping() -> None:
    import torch.nn as nn
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
        MossTTSLocalTransformerConfig,
    )
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
        MossTTSRealtimeLocalTransformer,
    )
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MossTTSLocalTransformerConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        audio_vocab_size=32,
        rvq=4,
    )
    with set_current_vllm_config(VllmConfig()):
        local_tf = MossTTSRealtimeLocalTransformer(cfg).to(device=device).eval()
        lm_heads = nn.ModuleList([nn.Linear(64, 32, bias=False) for _ in range(4)]).to(device=device)

    for bsz in (1, 3):
        hidden = torch.randn((bsz, 64), device=device)
        hist = [torch.randint(0, 28, (5, 4), device=device, dtype=torch.long) for _ in range(bsz)]
        eager_out = local_tf.generate_frame(
            hidden,
            lm_heads,
            temperature=0.0,
            top_k=10,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.1,
            history_tensors=hist,
            use_cuda_graph=False,
        )
        graph_out = local_tf.generate_frame(
            hidden,
            lm_heads,
            temperature=0.0,
            top_k=10,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.1,
            history_tensors=hist,
            use_cuda_graph=True,
        )
        assert torch.equal(eager_out, graph_out)

    # Verify mixed batch stopping (EOS, BOS, valid, already stopped) in make_omni_output
    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 4
    talker.audio_pad_token = 1024
    talker.text_vocab_size = 16
    talker.text_pad_id = 2
    talker.audio_eos_id = 15
    talker._batch_state = None
    talker._batch_state_spans = None

    scripted = [
        torch.tensor([talker.AUDIO_EOS, 1, 1, 1], dtype=torch.long),
        torch.tensor([talker.AUDIO_BOS, 2, 2, 2], dtype=torch.long),
        torch.tensor([5, 6, 7, 8], dtype=torch.long),
    ]
    call_idx = {"idx": 0}

    def fake_gen(*args, **kwargs):
        res = scripted[call_idx["idx"]].unsqueeze(0)
        call_idx["idx"] += 1
        return res

    talker.local_transformer = SimpleNamespace(generate_frame=fake_gen)
    talker.local_lm_heads = []

    infos = [
        {"audio_state": {"is_stopping": False, "step": 0}},  # emits AUDIO_EOS -> stops, empty output
        {"audio_state": {"is_stopping": True, "step": 3}},   # already stopped -> skipped
        {"audio_state": {"is_stopping": False, "step": 0}},  # emits AUDIO_BOS -> skipped
        {"audio_state": {"is_stopping": False, "step": 0}},  # emits [5,6,7,8] -> kept
    ]
    out = talker.make_omni_output(
        torch.ones((4, 8)),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2), (2, 3), (3, 4)],
        request_sample_eligible=[True, True, True, True],
    )
    codes_out = out.multimodal_outputs["codes"]["audio"]
    assert len(codes_out) == 4
    assert codes_out[0].numel() == 0 and infos[0]["audio_state"]["is_stopping"] is True
    assert codes_out[1].numel() == 0 and infos[1]["audio_state"]["step"] == 3
    assert codes_out[2].numel() == 0 and infos[2]["audio_state"]["is_stopping"] is False
    assert torch.equal(codes_out[3], torch.tensor([[5, 6, 7, 8]], dtype=torch.long))
