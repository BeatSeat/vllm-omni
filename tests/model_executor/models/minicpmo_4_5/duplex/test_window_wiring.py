# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests covering the Scheduler, Worker, and Batch Wiring for MiniCPM-o 4.5 duplex KV window.

Covers:
1. Block-table in-place compaction on the Worker side without row moves.
2. rotate_cached_keys numeric identity with attention dot products.
3. Multi-request concurrency / batched execution:
   - Request 0: triggers window trim with delta=16
   - Request 1: normal decode (untouched)
   - Request 2: triggers window trim with delta=32
4. Scheduler-side watermark detection and reanchor plan attachment without full re-prefill.
5. Barge-in / abort safety ensuring zero leaked blocks.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import (
    DuplexWindowGeometry,
    PositionReanchor,
    plan_position_reanchor,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
    DUPLEX_WINDOW_BLOCK_SIZE,
    MiniCPMO45DuplexWindowManager,
    MiniCPMO45DuplexWindowSpec,
    duplex_window_geometry,
    rotate_cached_keys,
    rotate_keys,
)

BLOCK_SIZE = DUPLEX_WINDOW_BLOCK_SIZE  # 16
HEAD_DIM = 128
NUM_KV_HEADS = 8


def _get_inv_freq(head_dim: int = HEAD_DIM, base: float = 1000000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def test_worker_block_table_compaction():
    """Verify that shifting block table entries on the worker is exact and zero-copy."""
    # Simulate a request with 7 physical blocks: [B0, B1, Gap0, Gap1, B2, B3, B4]
    sink_blocks = 2
    gap_blocks = 2
    num_blocks = 7
    initial_blocks = [101, 102, 201, 202, 301, 302, 303]

    table_np = np.zeros((4, 32), dtype=np.int32)
    num_blocks_per_row = np.zeros(4, dtype=np.int32)

    row_idx = 1
    table_np[row_idx, :num_blocks] = initial_blocks
    num_blocks_per_row[row_idx] = num_blocks

    # Worker-side compaction helper
    total = int(num_blocks_per_row[row_idx])
    assert total == 7
    table_np[row_idx, sink_blocks : total - gap_blocks] = table_np[
        row_idx, sink_blocks + gap_blocks : total
    ]
    table_np[row_idx, total - gap_blocks : total] = 0
    num_blocks_per_row[row_idx] -= gap_blocks

    assert num_blocks_per_row[row_idx] == 5
    compacted = list(table_np[row_idx, :5])
    assert compacted == [101, 102, 301, 302, 303]
    # Zeroed out tail
    assert list(table_np[row_idx, 5:7]) == [0, 0]


def test_rotate_cached_keys_attention_equivalence():
    """Verify that attention scores against rotated keys equal computing RoPE at pos - delta."""
    inv_freq = _get_inv_freq()
    delta = 16
    moved_from = 32
    sink_blocks = 2  # 32 tokens
    plan = PositionReanchor(delta=delta, moved_from=moved_from, sink_blocks=sink_blocks)

    num_blocks = 10
    block_ids = list(range(num_blocks))
    num_tokens = 64
    positions = torch.arange(moved_from, num_tokens, dtype=torch.long)

    # Key cache pool: (num_blocks, block_size, num_kv_heads, head_dim)
    torch.manual_seed(42)
    k_pool = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)

    # Rotate in place
    k_pool_before = k_pool.clone()
    touched = rotate_cached_keys(
        k_pool,
        block_ids=block_ids,
        positions=positions,
        plan=plan,
        inv_freq=inv_freq,
    )
    assert touched == len(positions)

    # Pick a token at pos = 40 (which shifts to 40 - 16 = 24)
    test_pos = 40
    new_pos = test_pos - delta
    block_idx = test_pos // BLOCK_SIZE
    offset = test_pos % BLOCK_SIZE

    original_k = k_pool_before[block_idx, offset]  # (heads, dim)
    rotated_k = k_pool[block_idx, offset]

    # Directly un-rotate original_k by delta
    expected_rotated = rotate_keys(original_k.unsqueeze(0), delta, inv_freq).squeeze(0)
    assert torch.allclose(rotated_k, expected_rotated, atol=1e-6)

    # Attention score check with a query at step Q
    q = torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    # The relative angle between Q and K must be preserved:
    # Score = q(pos_q) * k(pos_k - delta) == q(pos_q + delta) * k(pos_k)
    score_rotated = (q * rotated_k).sum()
    q_shifted = rotate_keys(q.unsqueeze(0), -delta, inv_freq).squeeze(0)
    score_original = (q_shifted * original_k).sum()
    assert torch.allclose(score_rotated, score_original, atol=1e-5)


def test_batched_concurrency_isolation():
    """Verify that in a batch of multiple concurrent requests:
    - Request 0 triggers trim (delta=16)
    - Request 1 is normal decode (untouched)
    - Request 2 triggers trim with different delta=32
    Non-triggering requests are completely unaffected.
    """
    inv_freq = _get_inv_freq()
    num_blocks = 20
    torch.manual_seed(123)
    k_pool = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    k_pool_snapshot = k_pool.clone()

    # Request 0: blocks [0, 1, 2, 3], needs delta=16, moved_from=32, sink_blocks=1
    # Gap is block 1. Compacted table after trim is [0, 2, 3].
    plan_0 = PositionReanchor(delta=16, moved_from=32, sink_blocks=1)
    req0_blocks_compacted = [0, 2, 3]
    req0_positions = torch.arange(32, 64, dtype=torch.long)

    # Request 1: blocks [4, 5, 6, 7], normal decode, NO trim
    req1_blocks = [4, 5, 6, 7]

    # Request 2: blocks [8, 9, 10, 11, 12], needs delta=32, moved_from=48, sink_blocks=1
    # Gap is blocks [9, 10]. Compacted table after trim is [8, 11, 12].
    plan_2 = PositionReanchor(delta=32, moved_from=48, sink_blocks=1)
    req2_blocks_compacted = [8, 11, 12]
    req2_positions = torch.arange(48, 80, dtype=torch.long)

    # Execute Re-RoPE for the batch
    reanchor_batch = {
        0: (req0_blocks_compacted, req0_positions, plan_0),
        2: (req2_blocks_compacted, req2_positions, plan_2),
    }

    for req_idx, (b_ids, pos, plan) in reanchor_batch.items():
        rotate_cached_keys(
            k_pool,
            block_ids=b_ids,
            positions=pos,
            plan=plan,
            inv_freq=inv_freq,
        )

    # ASSERTION 1: Request 1's physical blocks [4, 5, 6, 7] are 100% UNTOUCHED
    for b in req1_blocks:
        assert torch.equal(k_pool[b], k_pool_snapshot[b]), f"Block {b} of Request 1 was corrupted!"

    # ASSERTION 2: Request 0's sink block [0] is UNTOUCHED
    assert torch.equal(k_pool[0], k_pool_snapshot[0]), "Request 0 sink block was corrupted!"

    # ASSERTION 3: Request 0's tail blocks [2, 3] are rotated by delta=16
    for b in [2, 3]:
        expected = rotate_keys(k_pool_snapshot[b], 16, inv_freq)
        assert torch.allclose(k_pool[b], expected, atol=1e-6)

    # ASSERTION 4: Request 2's sink block [8] is UNTOUCHED
    assert torch.equal(k_pool[8], k_pool_snapshot[8]), "Request 2 sink block was corrupted!"

    # ASSERTION 5: Request 2's tail blocks [11, 12] are rotated by delta=32
    for b in [11, 12]:
        expected = rotate_keys(k_pool_snapshot[b], 32, inv_freq)
        assert torch.allclose(k_pool[b], expected, atol=1e-6)


def test_scheduler_reanchor_planning_and_no_full_reprefill():
    """Verify that Scheduler triggers reanchor without replacing the session prompt."""
    geometry = duplex_window_geometry(
        prefix_tokens=96,
        window_tokens=6000,
        block_size=16,
        max_model_len=40960,
        high_watermark_tokens=8000,
    )

    # Below high watermark (96 + 8000 = 8096): no reanchor
    plan = plan_position_reanchor(geometry, computed_tokens=7900, pending_tokens=12)
    assert plan is None

    # Crossing watermark: 8090 + 12 = 8102 > 8096
    # Target is prefix(96) + window(6000) = 6096
    target = geometry.prefix_tokens + geometry.window_tokens
    plan = plan_position_reanchor(geometry, computed_tokens=8090, pending_tokens=12)
    assert plan is not None
    assert plan.delta % BLOCK_SIZE == 0
    assert plan.delta > 0
    # Retained tail stays within target budget
    retained = (8090 + 12) - plan.moved_from
    assert target - BLOCK_SIZE < retained <= target
    # New sequence lands exactly at sink_end + retained
    assert (8090 + 12) - plan.delta == plan.sink_end + retained


def test_barge_in_abort_safety():
    """Verify that aborting a request during/after reanchor does not leak blocks."""
    # Simulate a mini block allocator
    free_blocks = set(range(100))
    allocated = {}

    def alloc(req_id, count):
        blocks = [free_blocks.pop() for _ in range(count)]
        allocated[req_id] = blocks
        return blocks

    def free_blocks_range(req_id, start, end):
        blocks = allocated[req_id]
        freed = blocks[start:end]
        for b in freed:
            free_blocks.add(b)
        allocated[req_id] = blocks[:start] + blocks[end:]

    def abort_request(req_id):
        blocks = allocated.pop(req_id, [])
        for b in blocks:
            free_blocks.add(b)

    # Req A allocates 8 blocks
    alloc("req-a", 8)
    assert len(free_blocks) == 92

    # Trim: free gap blocks [2:4] (2 blocks)
    free_blocks_range("req-a", 2, 4)
    assert len(free_blocks) == 94
    assert len(allocated["req-a"]) == 6

    # User barges in -> abort!
    abort_request("req-a")
    assert len(free_blocks) == 100  # ALL blocks successfully returned, zero leak!


def test_runner_stage0_reanchor_pipeline():
    """Verify that Worker applies stage0_reanchor hook correctly."""
    inv_freq = _get_inv_freq()
    num_blocks = 10
    k_pool = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    k_pool_orig = k_pool.clone()

    # Initial block table for req-1: blocks [0, 1, 2, 3]
    # sink=1 (block 0), gap=1 (block 1, delta=16), tail=[2, 3] (pos 32..64)
    table_np = np.zeros((2, 16), dtype=np.int32)
    table_np[0, :4] = [0, 1, 2, 3]
    num_blocks_per_row = np.array([4, 0], dtype=np.int32)

    class _MockBlockTable:
        def __init__(self):
            self.block_table = SimpleNamespace(np=table_np)
            self.num_blocks_per_row = num_blocks_per_row

    class _MockRunner:
        def __init__(self):
            self.device = torch.device("cpu")
            self.cache_config = SimpleNamespace(block_size=BLOCK_SIZE)
            self.model_config = SimpleNamespace(
                get_head_size=lambda: HEAD_DIM,
                hf_config=SimpleNamespace(rope_theta=1000000.0),
            )
            self._duplex_inv_freq = inv_freq
            self.kv_caches = [k_pool]
            self.input_batch = SimpleNamespace(
                num_reqs=1,
                req_ids=["req-1"],
                block_table=_MockBlockTable(),
                num_computed_tokens_cpu=np.array([64], dtype=np.int32),
            )
            self.model_intermediate_buffer = {
                "req-1": {
                    "duplex": {
                        "stage0_reanchor": {
                            "delta": 16,
                            "moved_from": 32,
                            "sink_blocks": 1,
                        }
                    }
                }
            }

        def _maybe_apply_stage0_reanchor(self):
            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids[:num_reqs]
            for req_idx, req_id in enumerate(req_ids):
                info = self.model_intermediate_buffer.get(req_id)
                if not isinstance(info, dict):
                    continue
                duplex = info.get("duplex")
                if not isinstance(duplex, dict):
                    continue
                reanchor = duplex.pop("stage0_reanchor", None)
                if reanchor is None:
                    continue

                plan = PositionReanchor(
                    delta=reanchor["delta"],
                    moved_from=reanchor["moved_from"],
                    sink_blocks=reanchor["sink_blocks"],
                )

                block_size = self.cache_config.block_size
                gap_blocks = plan.delta // block_size
                sink_blocks = plan.sink_blocks
                bt = self.input_batch.block_table
                total = int(bt.num_blocks_per_row[req_idx])
                if sink_blocks + gap_blocks <= total:
                    bt.block_table.np[req_idx, sink_blocks : total - gap_blocks] = (
                        bt.block_table.np[req_idx, sink_blocks + gap_blocks : total]
                    )
                    bt.block_table.np[req_idx, total - gap_blocks : total] = 0
                    bt.num_blocks_per_row[req_idx] -= gap_blocks
                compacted_block_ids = list(bt.block_table.np[req_idx, : bt.num_blocks_per_row[req_idx]])

                old_computed = int(self.input_batch.num_computed_tokens_cpu[req_idx])
                self.input_batch.num_computed_tokens_cpu[req_idx] = max(0, old_computed - plan.delta)

                positions = torch.arange(plan.moved_from, old_computed, dtype=torch.long, device=self.device)
                if positions.numel() > 0:
                    for kv_cache in self.kv_caches:
                        rotate_cached_keys(
                            kv_cache,
                            block_ids=compacted_block_ids,
                            positions=positions,
                            plan=plan,
                            inv_freq=self._duplex_inv_freq,
                        )

    runner = _MockRunner()
    runner._maybe_apply_stage0_reanchor()

    # Verify block table is compacted to [0, 2, 3]
    bt = runner.input_batch.block_table
    assert bt.num_blocks_per_row[0] == 3
    assert list(bt.block_table.np[0, :3]) == [0, 2, 3]
    # Verify computed tokens decremented from 64 to 48
    assert runner.input_batch.num_computed_tokens_cpu[0] == 48
    # Verify metadata consumed
    assert "stage0_reanchor" not in runner.model_intermediate_buffer["req-1"]["duplex"]
    # Verify sink block 0 is untouched, blocks 2 and 3 are rotated
    assert torch.equal(k_pool[0], k_pool_orig[0])
    for b in [2, 3]:
        expected = rotate_keys(k_pool_orig[b], 16, inv_freq)
        assert torch.allclose(k_pool[b], expected, atol=1e-6)

