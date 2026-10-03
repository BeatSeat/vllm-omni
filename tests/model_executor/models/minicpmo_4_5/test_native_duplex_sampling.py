# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 native duplex sampling: batched rows, the deferred device path, candidate-space draws."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

Model = MiniCPMO45OmniForConditionalGeneration
CHUNK_EOS, LISTEN, TTS_PAD, VOCAB = 5, 6, 10, 40
_NAMES = ("unit", "chunk_eos", "listen", "tts_bos", "turn_eos", "chunk_tts_eos", "tts_pad")
TOKEN_IDS = {f"{name}_token_id": token for name, token in zip(_NAMES, (1, CHUNK_EOS, LISTEN, 7, 8, 9, TTS_PAD))}


class _CharTokenizer:
    """Token t decodes to (t % 3) + 1 copies of one CJK character; special ids decode to ""."""

    clean_up_tokenization_spaces = False
    bad_token_ids: list[int] = []
    all_special_ids = sorted(TOKEN_IDS.values())

    def __len__(self):
        return VOCAB

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if t in self.all_special_ids else chr(0x4E00 + t) * (t % 3 + 1) for t in map(int, ids))

    def batch_decode(self, batch, skip_special_tokens=False):
        return [self.decode(ids) for ids in batch]


def _scenario(seed: int, *, all_greedy: bool = False, max_chars: int = 6):
    """Six rows: sampled, listen in an open turn, speak-length cap, boundary chunk_eos, greedy, penalized."""
    torch.manual_seed(seed)
    steps = [torch.randn(6, VOCAB) * 2 for _ in range(2)]
    for logits in steps:
        logits[3, CHUNK_EOS] += 50.0
        logits[1, LISTEN] += 40.0

    def build():
        states = {
            row: SimpleNamespace(generated_tokens=[], current_turn_ended=row != 1, pending_speech_context=False)
            for row in range(6)
        }
        states[5].generated_tokens = [11, 11, 12]
        model = Model.__new__(Model)
        model.model_stage = "llm"
        model._minicpmo45_native_duplex_token_ids_cache = dict(TOKEN_IDS)
        model._minicpmo45_tokenizer_cache = _CharTokenizer()
        model._minicpmo45_duplex_state_for_row = states.get
        model._minicpmo45_duplex_payload_for_row = lambda row: None
        model._minicpmo45_duplex_row_request_max_tokens = lambda row: None
        model.max_new_speak_tokens_per_chunk = 5
        model.max_speak_chars_per_chunk = max_chars
        md = SimpleNamespace(
            all_greedy=all_greedy,
            generators={row: torch.Generator().manual_seed(100 + row) for row in range(6)},
            output_token_ids=[[11, 12], [2], [1, 2, 3, 4], [2, 3], [7, 2], [20, 21, 22]],
            temperature=torch.tensor([0.8, 1.1, 0.7, 0.7, 0.0, 0.9]),
            top_k=torch.tensor([10, 0, 100, 100, 100, 20]),
            top_p=torch.tensor([0.9, 1.0, 0.8, 0.8, 0.8, 0.85]),
        )
        return model, states, md

    return steps, build


def _run(model, md, steps) -> tuple[list[list[int]], list[bool]]:
    """Two consecutive steps; also whether each was decided on the device (left a pending commit)."""
    tokens, deferred = [], []
    for logits in steps:
        out = model._sample_minicpmo45_native_duplex_stage0(logits.clone(), md, duplex_rows=list(range(6)))
        tokens.append(out.sampled_token_ids.squeeze(-1).tolist())
        deferred.append(getattr(model, "_minicpmo45_duplex_pending_samples", None) is not None)
        model._commit_minicpmo45_duplex_pending_samples()
        for row, token in enumerate(tokens[-1]):
            md.output_token_ids[row].append(token)
    return tokens, deferred


def _reference(logits, md, states, row_params) -> list[int]:
    """Row by row: speak-length cap, boundary draw, then the stage-2 draw (no cut, no listen rewrite)."""
    out = []
    for row, (temperature, top_k, top_p) in enumerate(row_params):
        generator, recent = md.generators[row], md.output_token_ids[row]
        if len(recent) >= 4 or torch.rand((), generator=generator) < torch.softmax(logits[row], -1)[CHUNK_EOS]:
            out.append(CHUNK_EOS)
            continue
        row_logits = logits[row : row + 1].clone()
        row_logits[0, [TTS_PAD, CHUNK_EOS]] = float("-inf")
        for token in set((states[row].generated_tokens or recent)[-MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE :]):
            row_logits[0, token] /= 1.05
        if temperature <= 0:
            out.append(int(row_logits.argmax()))
            continue
        probs, ids = Model._duplex_top_k_top_p_candidates(row_logits / temperature, top_k=top_k, top_p=top_p)
        out.append(int(ids[0, torch.multinomial(probs[0], 1, generator=generator)]))
    return out


@pytest.mark.parametrize("seed", range(4))
def test_batched_rows_match_the_row_by_row_reference(seed):
    steps, build = _scenario(seed, max_chars=10**6)
    model, states, md = build()
    for state in states.values():
        state.current_turn_ended = True  # no listen rewrite
    ref_model, ref_states, ref_md = build()
    row_params = model._minicpmo45_duplex_row_params(md, 6)
    rows = model._sample_minicpmo45_native_duplex_rows(
        steps[0], md, row_idxs=list(range(6)), token_ids=TOKEN_IDS, row_params=row_params
    )
    assert rows == _reference(steps[0], ref_md, ref_states, row_params)
    for row in range(6):
        assert torch.equal(md.generators[row].get_state(), ref_md.generators[row].get_state())


@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_deferred_rows_match_the_synchronous_rows(seed, all_greedy, monkeypatch):
    steps, build = _scenario(seed, all_greedy=all_greedy)
    sync_model, sync_states, sync_md = build()
    monkeypatch.setattr(sync_model, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)
    expected, _ = _run(sync_model, sync_md, steps)
    model, states, md = build()
    assert _run(model, md, steps) == (expected, [True, True])
    assert {row: vars(state) for row, state in states.items()} == {row: vars(s) for row, s in sync_states.items()}
    for row in range(6):
        assert torch.equal(md.generators[row].get_state(), sync_md.generators[row].get_state())


def test_batched_rows_read_the_host_at_most_twice(monkeypatch):
    rows = list(range(32))
    model, _, _ = _scenario(0)[1]()
    model._minicpmo45_duplex_state_for_row = lambda row: None
    md = SimpleNamespace(all_greedy=False, generators={}, output_token_ids=[[] for _ in rows])
    reads, tolist = [], torch.Tensor.tolist

    def counting_tolist(self):
        reads.append(self)
        return tolist(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("per-row host read"))
    model._sample_minicpmo45_native_duplex_rows(
        torch.randn(32, VOCAB), md, row_idxs=rows, token_ids=TOKEN_IDS, row_params=[(0.8, 10, 0.9)] * 32
    )
    assert len(reads) <= 2


def test_batched_rows_match_per_row_reads():
    md = SimpleNamespace(temperature=torch.tensor([0.5, 0.9, 1.3]))
    rows = Model._sampling_metadata_rows(md, "temperature", 4, 0.8)
    assert rows == [Model._sampling_metadata_value(md, "temperature", row, 0.8) for row in range(4)]


def test_candidate_distribution_matches_the_filter():
    logits = torch.randn(1, 4096, generator=torch.Generator().manual_seed(0)) * 4
    filtered = torch.softmax(Model._top_k_top_p_filter(logits.clone(), top_k=20, top_p=0.85), dim=-1)
    probs, indices = Model._duplex_top_k_top_p_candidates(logits.clone(), top_k=20, top_p=0.85)
    dense = torch.zeros_like(filtered)
    dense[0].scatter_(0, indices[0], probs[0])
    support = filtered[0] > 0
    assert torch.equal(support, dense[0] > 0)
    assert torch.allclose(filtered[0][support], dense[0][support], atol=1e-5)
