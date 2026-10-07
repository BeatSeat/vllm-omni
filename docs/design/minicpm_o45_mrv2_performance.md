# MiniCPM-o 4.5 MRv2 profiles

Turn mode uses the stage-level MRv2 contracts from #8184. Thinker emits live
latent metadata outside graph replay; Talker keeps codec history and EOS
control on device. Code2Wav reuses mainline Whole-Euler Flow graphs and shared
prompt state. The previous experimental block compilation, tiled attention,
channels-last and merged-CFM implementations are removed.

`minicpmo_4_5_turn_mrv2_h200.yaml` selects Talker capacity 16 and 4 GiB KV;
the generic MRv2 profile selects capacity 8 and 2 GiB KV. These configurations
require new end-to-end performance measurements after the mainline integration.
Previously reported numbers do not describe this revised codec backend.

The generic MRv2 profile and the opt-in native V1 duplex H200 throughput profile enable
ordinary TF32 for CFM DiT GEMMs within Code2Wav forward/capture only
(`torch.backends.cuda.matmul.allow_tf32` on dense QKV/MLP, Triton
`input_precision="tf32"` on tiled attention). This is not compensated TF32x3.
The previous process matmul policy is restored afterwards. cuDNN's TF32 policy
is independent. HiFT stays IEEE FP32. TF32 changes rounding.

The shared asynchronous output snapshot and batched Talker preprocessing also
apply to V1 when async chunking/scheduling is enabled. Default V1 behavior must
therefore be included in end-to-end regression validation.

CFM graph execution defaults to the fused DiT body on CUDA when TF32 is
allowed. NVIDIA SM80+ uses tiled TF32 attention; older CUDA devices use the
fused body's SDPA fallback. Unsupported DiT layouts or non-FP32 attention
caches retain the original body. Other platforms retain their existing default.
Set `cfm_fused_body: false` in connector `extra` to restore the original body.
Disabling TF32 with `token2wav_allow_tf32: false` or
`MINICPMO_CODE2WAV_TF32=off` also disables the default fusion choice; an explicit
`cfm_fused_body: true` overrides that choice and may use TF32 attention.
Slot pooling and row-offset merging remain opt-in. Fusion changes floating-point
rounding and does not promise identical waveforms.

## Native Duplex Deployment Profiles on H200

We structure native duplex deployment on H200 into distinct profile sets matching deployment priorities:

1. **`minicpmo_4_5_duplex_h200.yaml` (Max Throughput Profile)**:
   - Enables `cfm_fused_body: true`, `cfm_slot_pool: true`, and `cfm_row_offset_merge: true`.
   - Retains FP32 attention cache (`code2wav_bfloat16_attention_cache: false`), which is required by `cfm_fused_body` and tiled attention on NVIDIA SM80+.
   - Combines with Stage 0 incremental Fbank, SigLIP fused layers & CUDA graphs, and streaming Whisper encoder CUDA graph.
2. **`minicpmo_4_5_duplex_bf16_h200.yaml` (Max Concurrency / Memory-Saving Profile)**:
   - Uses compressed BF16 DiT attention cache (`code2wav_bfloat16_attention_cache: true`) to halve DiT attention memory consumption.
   - Disables `cfm_fused_body` and `cfm_slot_pool` to avoid silent fallback warnings, using WholeEuler standard arena allocation instead.
3. **`minicpmo_4_5_duplex_mrv2.yaml` (MRv2 Unified Profile)**:
   - Unifies all 3 stages under Model Runner V2 (`model_runner: v2`). It inherits the H200 capacity numbers directly from `minicpmo_4_5.yaml` (Stage 1 `max_num_seqs: 16` and the CUDA `kv_cache_memory_bytes: 4 GiB` overlay), so a separate `*_h200.yaml` MRv2 overlay is redundant and intentionally not provided.

The default duplex profile keeps the mainline V1 session path. Turn results do
not establish duplex performance or interruption correctness.

## Duplex mode on MRv2

Full-duplex serving (`session_mode: duplex`) is adapted to Model Runner V2
via `minicpmo_4_5_duplex_mrv2.yaml`, which already carries the H200 Stage-1
capacity (16 sessions / 4 GiB KV) inherited from `minicpmo_4_5.yaml`.
Thinker (stage 0), Talker (stage 1) and Code2Wav (stage 2) execute through MRv2 runners.
The audio encoder graph buckets follow `duplex_session.max_sessions`, avoiding
unreachable larger batches that consume Code2Wav's graph-memory headroom.

- Thinker: reuses native Stage-0 duplex preprocessing, window KV compaction
  and reanchor (`MiniCPMO45DuplexWorkerHelper`), and the latest batched/deferred
  sampling policy via the model-registered `MiniCPMO45DuplexSampler`. Outputs `duplex_prompt_token_ids` and
  special boundary token metadata within `make_omni_output_mrv2` for `llm2tts`.
- Talker: runs on `OmniARModelRunner` with native duplex metadata propagation
  (`duplex_epoch`, `duplex_turn_id`, `llm_output_text_utf8`, `turn_end`).
  It preserves the V1 codec floor and turn-end cadence EOS mask. Streaming
  conditions share the V1 prompt-window recipe: full attention recomputes at
  capacity, while `sliding_recompute` rebuilds on every condition boundary.
- Code2Wav: receives full-payload duplex metadata on MRv2 generation runner.

The MRv2 adapter reads request sampling parameters from the intermediate buffer,
maintains seeded generators by request ID, and copies accepted token histories
from the device ledger before policy evaluation. Partial prefills do not advance
the policy state or generators. Empty duplex row sets are published every step;
pending samples are committed before the next append or request cleanup.

The overlay retains the base deployment's TF32 policy and Talker KV budget.
Candidate distributions are compared against dense top-k/top-p filtering in
FP32 and BF16. Deferred and synchronous sampling are checked for identical
tokens, session state and per-row RNG state on CPU and CUDA. These comparisons
use the latest Stage-0 sampling algorithm: its Bernoulli boundary draw and
candidate-space sampling do not promise the old full-vocabulary sampler's
seeded token sequence.
