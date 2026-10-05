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

## Duplex mode on MRv2

Full-duplex serving (`session_mode: duplex`) is adapted to Model Runner V2
via `minicpmo_4_5_duplex_mrv2.yaml`. Thinker (stage 0), Talker (stage 1) and
Code2Wav (stage 2) execute through MRv2 runners.

- Thinker: reuses native Stage-0 duplex preprocessing, window KV compaction
  and reanchor (`MiniCPMO45DuplexWorkerHelper`), and device-side duplex sampling
  via `DuplexSamplingRunnerMixin`. Outputs `duplex_prompt_token_ids` and
  special boundary token metadata within `make_omni_output_mrv2` for `llm2tts`.
- Talker: runs on `OmniARModelRunner` with native duplex metadata propagation
  (`duplex_epoch`, `duplex_turn_id`, `native_duplex_segment_text`, `turn_end`).
- Code2Wav: receives full-payload duplex metadata on MRv2 generation runner.

