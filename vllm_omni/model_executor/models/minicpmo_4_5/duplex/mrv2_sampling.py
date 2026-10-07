# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt the MiniCPM duplex policy to MRv2's request slots and sampler output."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.model_executor.duplex_sampling import DuplexSamplingHelper
from vllm_omni.worker_v2.omni_sampler import OmniSampler


class MiniCPMO45DuplexSampler(OmniSampler):
    """Keep policy RNG/session state separate from the stock MRv2 sampler."""

    def __init__(self, base_sampler: Any, model: Any) -> None:
        super().__init__(base_sampler)
        self.model = model
        self.generators: dict[str, torch.Generator] = {}
        model._mrv2_duplex_sampler = self

    def forget_requests(self, request_ids) -> None:
        for request_id in request_ids:
            self.generators.pop(request_id, None)
            getattr(self.model, "_mrv2_sampling_infos", {}).pop(request_id, None)

    def _metadata(self, input_batch, rows, infos, device):
        """Read the accepted history, excluding graph padding and uncomputed tokens.

        MRv2 stores its token ledger on device. This compatibility path copies
        the needed histories before policy evaluation; it favors correctness
        over avoiding that read. Per-request generators survive batch reorder.
        """
        seq_lens = input_batch.seq_lens[: input_batch.num_reqs].tolist()
        histories, params, generators = [], [], {}
        for local_row, row in enumerate(rows):
            slot = int(input_batch.idx_mapping_np[row.row_idx])
            prompt_len = int(self.req_states.prompt_len.np[slot])
            end = int(seq_lens[row.row_idx])
            histories.append(self.req_states.all_token_ids.gpu[slot, prompt_len:end].tolist())
            sp = infos[row.request_id]["sampling_params"]
            params.append(sp)
            if sp.seed is not None:
                generator = self.generators.get(row.request_id)
                if generator is None:
                    generator = torch.Generator(device=device).manual_seed(sp.seed)
                    self.generators[row.request_id] = generator
                generators[local_row] = generator
        return SimpleNamespace(
            output_token_ids=histories,
            generators=generators,
            temperature=torch.tensor([sp.temperature for sp in params]),
            top_k=torch.tensor([sp.top_k for sp in params]),
            top_p=torch.tensor([sp.top_p for sp in params]),
            all_greedy=all(sp.temperature <= 0 for sp in params),
        )

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> SamplerOutput:
        infos = getattr(self.model, "_mrv2_sampling_infos", {})
        helper = DuplexSamplingHelper()
        runner = SimpleNamespace(input_batch=input_batch, model_intermediate_buffer=infos)
        for request_id in input_batch.req_ids:
            helper.refresh_active_request(runner, request_id)
        rows = helper.rows(runner)
        if rows and input_batch.num_draft_tokens:
            raise NotImplementedError("MiniCPM-o MRv2 duplex sampling does not support speculative decoding")
        # Partial prefills are discarded by the runner and must not mutate the
        # policy latches or advance their generators.
        rows = tuple(
            row
            for row in rows
            if not input_batch.is_prefilling_np[row.row_idx]
            or int(input_batch.num_computed_prefill_tokens_np[row.row_idx])
            + int(input_batch.num_scheduled_tokens[row.row_idx])
            >= int(input_batch.prefill_len_np[row.row_idx])
        )
        metadata = self._metadata(input_batch, rows, infos, logits.device) if rows else None
        selected = torch.tensor([row.row_idx for row in rows], device=logits.device, dtype=torch.long)
        policy_logits = logits.index_select(0, selected)
        self.model.prepare_duplex_sampling(
            policy_logits, metadata, tuple(replace(row, row_idx=i) for i, row in enumerate(rows))
        )
        if not rows:
            return self.base_sampler(logits, input_batch)
        policy_output = self.model.sample(policy_logits, metadata)
        if policy_output is None:
            return self.base_sampler(logits, input_batch)
        if len(rows) != input_batch.num_reqs:
            # The stock sampler owns counts/logprobs for any ordinary rows.
            output = self.base_sampler(logits, input_batch)
            output.sampled_token_ids.index_copy_(0, selected, policy_output.sampled_token_ids.long())
            return output
        counts, rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=policy_output.sampled_token_ids.long(),
            logprobs_tensors=policy_output.logprobs_tensors,
            num_nans=None,
            num_sampled=counts,
            num_rejected=rejected,
        )
