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

The generic MRv2 profile and the opt-in native V1 duplex H200 profile enable
ordinary TF32 for CFM DiT GEMMs within Code2Wav forward/capture only
(`torch.backends.cuda.matmul.allow_tf32` on dense QKV/MLP, Triton
`input_precision="tf32"` on tiled attention). This is not compensated TF32x3.
The previous process matmul policy is restored afterwards. cuDNN's TF32 policy
is independent. HiFT stays IEEE FP32. TF32 changes rounding.
Duplex keeps the mainline V1 session path. Turn results do not establish duplex
performance or interruption correctness.

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

## Full-duplex MRv2

`minicpmo_4_5_duplex_mrv2.yaml` opts all three CUDA stages into MRv2.
It inherits the default duplex profile's sampling, codec chunk size, cache
precision, TF32 policy, 16-session capacity and 4 GiB Talker KV budget.
Non-CUDA platforms retain V1. No separate H200 or BF16-cache profile is needed.

The required overrides are Stage 0 `async_chunk: false` for the completed
Thinker-to-Talker handoff and Stage 1/2 `async_scheduling: false` to serialize
segment completion with subsequent input. Stage 0's audio graph bucket sizing
is optional memory tuning; it follows the session capacity.

The Talker reuses the existing streaming prompt recipe: full attention extends
its KV prefix until capacity, while sliding recompute rebuilds the previous
condition, confirmed codec ids and current condition. Codec history survives
condition boundaries for the 16-frame penalty. The MRv2 host-side sampled EOS
flushes a segment; `turn_end` alone never closes it prematurely.

Configuration and unit tests do not establish audio quality or speedup.
Compare V1 and V2 on matched resolved configurations, input traces, model and
ASR revisions, concurrency and repeated seeds. Record per-item WER, output
lengths, errors, first-audio and inter-chunk latency, RTF definition and peak
memory. The runners use different random-number streams, so equal seeds do
not guarantee identical codec tokens. Include long AV sessions, residual-input
commit, turn-end drain, interruption and context rollover in model E2E tests.
