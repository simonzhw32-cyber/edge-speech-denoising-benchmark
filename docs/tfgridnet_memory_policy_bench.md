# TF-GridNet training-memory policy benchmark

Measured 2026-11-12 on the local Windows host. This is a **measurement harness**,
not a benchmark-protocol result: one random 4 s mixture, batch 1, 6-layer
TF-GridNet (`8,381,504` parameters), one warm-up step, then 20 measured AdamW
steps in a fresh process. The model is randomly initialized, so the numbers
characterize memory and step time, not enhancement quality.

Hardware/software: RTX 5060 Ti 16 GB, driver 617.42, Python 3.14.9,
PyTorch 2.11.0+cu128. Peak allocated/reserved are from
`torch.cuda.max_memory_allocated/reserved`; reserved includes allocator
fragmentation and is therefore the safer fit criterion for a 16 GB card.

Command:

```powershell
.\.conda\python.exe scripts\bench_tfgridnet_policies.py --steps 2 `
  --output results\paper_runs\tfgridnet_policy_bench.json
```

Initial two-step scoping table (absolute times are superseded by the 20-step
table below; the memory columns remain representative):

| Hop | Policy | Peak alloc GiB | Peak reserved GiB | Step s | Fits 16 GB |
|---|---|---:|---:|---:|---|
| 8 ms | FP32, no checkpoint | 12.72 | 13.03 | OOM (+2.86 GiB request) | no |
| 8 ms | FP32, checkpoint 4 blocks | 8.53 | 13.02 | 7.52 | marginal |
| 8 ms | FP32, checkpoint 6 blocks | 5.77 | 10.13 | 9.10 | yes |
| 8 ms | FP16 autocast, no checkpoint | 11.12 | 13.04 | 4.54 | marginal |
| 8 ms | FP16 + checkpoint 4 | 4.77 | 6.80 | 7.21 | yes |
| 8 ms | FP16 + checkpoint 6 | 3.21 | 5.36 | 8.59 | yes |
| 16 ms | FP32, no checkpoint | 9.90 | 10.48 | 2.52 | yes |
| 16 ms | FP16, no checkpoint | 5.63 | 5.92 | 2.39 | yes |

Reading the table:

- The paper-faithful 8 ms hop cannot train the 6-layer model at batch 1 in
  FP32 without checkpointing on this card. FP16 alone removes only ~1.6 GiB
  of the peak and still leaves the allocator close to the limit.
- Checkpointing four blocks is the fastest 8 ms policy that gets under the
  card's limit in FP16; checkpointing all six blocks costs another ~1.4 s per
  step for ~1.6 GiB less peak allocation.
- The 16 ms fallback is ~3× faster per step and fits without checkpointing,
  but it is not the paper's 8 ms STFT configuration.

The paired quality runs below check the optimization consequence directly:
200 steps with `whole 0–2` versus `attention + whole 0–2`, and 100 steps with
no checkpoint versus attention-only. Both pairs produce bit-identical loss
columns (max abs diff 0.0) and the same PESQ trajectory, so the selected
policies do not change the optimization, not merely the forward pass.

## What this does and does not establish

Established: the memory/time trade-off of uniform block checkpointing, attention
sub-block checkpointing and FP16 on one consumer GPU, with a reproducible
harness and a clear 8 ms feasibility boundary. Also established: the selected
policies are loss-bit-identical to their controls over 200 and 100 steps.

Not established: final PESQ/STOI equivalence after full training, a learned
policy beyond the measured greedy ordering, or multi-GPU behavior. Those
require the longer quality run and the second model/GPU tracked in
`docs/tfgridnet_paper_plan.md`.

## Sub-block checkpointing: attention is the cheap first step

Whole-block checkpointing treats every block as an indivisible unit. The block
also contains an attention section whose recompute cost is much smaller than a
full block, so checkpointing only that section can save memory at nearly no
step-time cost. `speech_denoising.training.tfgridnet_policy.install_subblock_checkpointing`
patches each block's forward into intra-RNN, inter-RNN, and attention sections;
each can be checkpointed independently.

Measured on the same harness, 8 ms, FP16, batch 1, 4 s, one warm-up plus 20
measured steps per fresh process (peak allocated / reserved GiB, seconds):

| Policy | Peak alloc | Peak reserved | Step s |
|---|---:|---:|---:|
| FP16, no checkpoint | 11.12 | 13.04 | 3.89 |
| FP16, attention only | 9.83 | 11.16 | 3.92 |
| FP16, whole block 0 | 9.54 | 11.14 | 4.47 |
| FP16, attention + whole 0 | 8.50 | 11.11 | 4.49 |
| FP16, whole blocks 0–1 | 7.95 | 8.25 | 5.03 |
| FP16, whole blocks 0–2 | 6.36 | 8.24 | 5.61 |
| FP16, attention + whole 0–2 | 5.84 | 8.24 | 5.62 |
| FP16, whole blocks 0–3 | 4.77 | 6.80 | 6.17 |
| FP16, attention + whole 0–3 | 4.51 | 6.80 | 6.19 |
| FP16, whole blocks 0–4 | 3.19 | 5.36 | 6.75 |
| FP16, all sub-blocks | 2.78 | 3.92 | 7.29 |

The useful result is not that every sub-block helps. Checkpointing the intra
or inter RNN is expensive; the all-sub-block point reaches 2.78 GiB but costs
7.29 s/step. Attention is different: it saves 1.29 GiB for **0.03 s**, within
run-to-run noise.

That makes attention checkpointing a cheap first move. Adding it to the
whole-block policy shifts the frontier:

- `attention + whole 0` uses **1.04 GiB less** than `whole 0` for +0.02 s
  (8.50 vs 9.54 GiB, 4.49 vs 4.47 s).
- `attention + whole 0–2` uses **0.52 GiB less** than `whole 0–2` for +0.01 s
  (5.84 vs 6.36 GiB).
- `attention + whole 0–3` uses **0.26 GiB less** than `whole 0–3` for +0.02 s
  (4.51 vs 4.77 GiB).

So the paper's policy is **cheap-first hierarchical checkpointing**: measure
the recompute cost per GiB of each sub-block, checkpoint the cheapest section
(here attention) in every block, then spend the remaining memory budget on
whole blocks. This is a Pareto improvement over uniform whole-block
checkpointing at the low- and mid-memory end, and it degrades gracefully to
the uniform policy when attention checkpointing is not enough.

20-step Pareto (8 ms, FP16, one warm-up + 20 measured steps per fresh process):

| Policy | Peak alloc GiB | Peak reserved GiB | Step s |
|---|---:|---:|---:|
| no checkpoint | 11.12 | 13.04 | 3.89 |
| attention only | 9.83 | 11.16 | 3.92 |
| whole block 0 | 9.54 | 11.14 | 4.47 |
| attention + whole 0 | 8.50 | 11.11 | 4.49 |
| whole blocks 0–1 | 7.95 | 8.25 | 5.03 |
| whole blocks 0–2 | 6.36 | 8.24 | 5.61 |
| attention + whole 0–2 | 5.84 | 8.24 | 5.62 |
| whole blocks 0–3 | 4.77 | 6.80 | 6.17 |
| attention + whole 0–3 | 4.51 | 6.80 | 6.19 |
| whole blocks 0–4 | 3.19 | 5.36 | 6.75 |
| all sub-blocks | 2.78 | 3.92 | 7.29 |

Caveats: these are single-GPU, random-input measurements; each row is a fresh
process with one warm-up and 20 measured steps, and absolute times include
per-process kernel setup. Compare rows within this table, not against
wall-clock training runs. The quality-equivalence check is a separate 200-step
paired run on real VoiceBank audio; it is short-horizon evidence, not a
final-quality comparison.

## Model-size check: 4-layer DNS profile

The 4-layer DNS profile (`configs/tfgridnet_dns.yaml`, 2,552,790 parameters)
fits at 8 ms without checkpointing. At 8 ms/FP16, batch 1, 20 steps:
`none` 2.17 GiB allocated / 2.76 reserved / 0.173 s per step; attention-only
1.63 GiB / 2.27 reserved / 0.189 s per step. Attention saves 0.54 GiB for
0.016 s (+9 %), with identical loss columns
(`results/paper_runs/tfgridnet_dns4_8ms.json`). The cheap-first order survives,
but the relative cost is higher than the six-layer model's 0.03 s (+0.7 %).

Command used:

```powershell
.\.conda\python.exe scripts\bench_tfgridnet_policies.py --child `
  --config configs\tfgridnet_dns.yaml --stride 128 --amp fp16 --steps 20 `
  --policy dns4_attn --checkpoint-blocks 0 --subblock-attention
```

## Optimization equivalence (200-step paired run)

Two 200-step runs were trained from the same seed, data order, and batch
sequence at 8 ms/FP16: `whole 0-2` (`p3_whole3_200_paired`) and `attention +
whole 0-2` (`p4_attn_whole3_200_paired`). Their per-step losses are
**bit-for-bit identical for all 200 steps** (maximum absolute difference 0.0;
both loss columns hash to `64f94acf24446d2c…`). Both learn: validation PESQ is
1.616 → 1.692 → 1.813 → 1.783 at steps 50/100/150/200, and the two policies
agree to within 2e-6 PESQ at every checkpoint. The 20-utterance validation
subset is fixed by the runner.

The paired 200-step run and the completed 1,000-step uniform run are plotted in
`results/paper_runs/tfgridnet_quality.png` (generated by
`scripts/plot_tfgridnet_quality.py`). The uniform arm completed 1,000 steps at
PESQ **2.12296** (STOI 0.89618, SI-SNR 14.7984, improvement 7.6004 dB); the
trajectory is noisy on the 20-utterance subset (1.89–2.13), so this is a
learning/equivalence check, not a final quality claim.

Two controlled A/B/A timing checks (30 steps each, same seed/data) give
`whole 0-2` 5.601/5.611 s and attention 5.624 s in a fast session, then
7.297/7.406 s and attention 6.868 s in a slow session. The attention arm is
not slower in either; the 0.52 GiB memory saving (6.36 → 5.84 GiB allocated)
has no consistent step-time penalty. The 1,000-step arm itself alternated
between 7.66 and 5.65 s/step with no code change, so absolute step time is
environment-sensitive. The paper claims memory and trajectory equivalence,
not a step-time speedup. See `timing_controls.json`.

The first 150-step pair (`p3_whole3_150`, `p4_attn_whole3_150`) cropped noisy
and clean with independent random offsets. That made the proxy learn nothing
(loss ~36–55, PESQ ~1.05); it is retained only as a historical
bit-identical-loss check, not as quality evidence. `_crop_pair` now draws one
offset for both signals; the 200-step pair above is the corrected comparison.

## Precision check: FP16 versus FP32 at 16 ms

The 8 ms model does not fit in FP32, so the precision pair is measured at the
16 ms profile where both arms fit. Both runs use 200 steps, no checkpointing,
the same 400-utterance pool, batch 1, seed 42, and the same 20-utterance
validation subset:

| Precision | Peak alloc GiB | Peak reserved GiB | Mean step s | PESQ 50 / 100 / 150 / 200 |
|---|---:|---:|---:|---|
| FP32 | 9.90 | 12.90 | 2.73 | 1.663 / 1.727 / 1.809 / 1.891 |
| FP16 | 5.63 | 12.08 | 2.59 | 1.640 / 1.758 / 1.824 / 1.889 |

Reserved includes the 20-utterance validation pass; the training-step reserved
values are lower (initial table above). FP16 saves **4.27 GiB** of peak allocation and is slightly faster. The final
PESQ difference is **-0.003** and STOI **+0.001**; the loss trajectories are
not bit-identical (mean absolute loss difference 0.37), and the final SI-SNR
is 1.14 dB lower in FP16, so this is a quality-neutral result within proxy
noise, not an exact-equivalence claim. At 8 ms, FP16 is required: the FP32
no-checkpoint run OOMs.

A second 100-step pair tests attention-only at the no-checkpoint level:
`q_none_100` vs `q_attn_100_cache`. Losses are bit-identical (hash
`466cba038606ea37…`), PESQ is 1.616 / 1.692 at steps 50/100 in both, mean step
time is 3.862 s vs 3.858 s, and peak allocated memory is 11.12 vs 9.83 GiB.
Without the conditional cache release the attention run degrades to 9.125
s/step after validation reserves 14.58 GiB (`q_attn_100`), the validation-cache
cliff noted above.


One caveat: the training-step peak and the end-to-end peak differ. The
training step of `whole 0-2` reserves about 8.24 GiB, while full-utterance
validation pushes the process peak to about 13.5 GiB for both policies. The
Pareto table reports the training-step measurement; a production fit check
must include validation memory or chunk validation.

A sharper end-to-end case: attention-only training allocates 9.83 GiB, but
after a 20-utterance validation the allocator held 14.58 GiB reserved and
step time collapsed from 3.9 s to 14.3 s. Releasing cached blocks when
reserved memory exceeds 14 GiB restores 3.9 s/step. Attention-only is
therefore not a safe standalone policy on a 16 GB card; the attention +
whole-block hybrid at 13.5 GiB reserved does not hit this cliff. The policy
trainer now performs the conditional cache release after validation.
