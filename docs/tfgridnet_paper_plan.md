# Paper plan: memory-efficient 8 ms TF-GridNet training on consumer GPUs

Status: active research plan, 2026-11-12. This document is a plan and an
experiment map, not a claim of completed results.

## Working title

**Budget-Aware Activation Checkpointing and Precision for Paper-Faithful
8 ms TF-GridNet Training on a 16 GB Consumer GPU**

Alternative framing if the selective policy does not beat uniform
checkpointing: *A reproducible memory/time Pareto study of TF-GridNet training
configurations* (benchmark/systems venue rather than method venue).

## Problem

The TF-GridNet paper uses a 32 ms window and an **8 ms hop**. The repository's
local profile uses a 16 ms hop because the 8 ms configuration exceeds the
16 GB RTX 5060 Ti during training. On this card the 8 ms, 6-layer model at
batch 1 peaks at 12.72 GiB allocated and then requests another 2.86 GiB,
so the paper-faithful configuration cannot be trained without changing the
compute policy. The question is whether checkpointing and precision can make
it fit **without changing the model or the objective**, and at what step-time
cost.

## Contributions (target)

1. A measured **VRAM / step-time / numerical-equivalence Pareto frontier** for
   paper-faithful 8 ms TF-GridNet on a 16 GB consumer GPU, covering uniform
   block checkpointing, FP16 autocast, and their combinations.
2. A **budget-aware selective checkpointing policy**: choose which blocks to
   recompute from a per-block memory-saving/recompute-cost model, rather than
   checkpointing the first *k* blocks uniformly. The policy is a small
   knapsack/greedy solver over six blocks; its cost model is calibrated from
   measured block activations.
3. A **quality-proxy protocol** at equal optimizer-step and equal-data budgets
   that verifies the policy does not change the learning trajectory beyond
   numerical noise, plus a short PESQ/STOI comparison.

The novelty is not "we used `torch.utils.checkpoint`"; it is the budgeted
selection and the reproducible Pareto/evaluation protocol. Reviewers will
reject a paper whose only method is the built-in checkpoint API.

## Method sketch

For each block `b`, measure:
- `M_b`: peak activation memory retained by the block when it is not
  checkpointed (from a forward hook / saved-tensor inspection);
- `C_b`: recompute time of the block in backward;
- `S_b`: memory saved by checkpointing it.

Given a target peak `B` (e.g., 10 GiB reserved), solve
`max sum(S_b x_b)` subject to `sum(M_b (1-x_b)) <= B` and a step-time
penalty, or minimize predicted step time subject to the memory cap. With six
blocks the exact enumeration is 64 policies; a greedy margin-per-second rule
is the scalable variant. The paper reports both the oracle (enumeration) and
the greedy policy.

Precision is a second axis: FP16 autocast lowers activation memory and step
time, but complex-half support is experimental; the paper must report the
warning and verify loss/gradient equivalence.

## Experiment matrix

All quality-proxy runs use the verified VoiceBank-DEMAND-16k train manifest,
4 s crops, batch 1, AdamW 1e-3, clip 1.0, seed 42, and an equal step budget
(200 steps for the whole-block pair, 100 for the attention-only pair, 200 for the precision pair).
Validation uses a fixed 20-utterance subset and the shared
PESQ/STOI/SI-SNR implementation.

| ID | Hop | Precision | Checkpoint | Role |
|---|---|---|---|---|
| P0 | 8 ms | FP32 | none | infeasible reference (OOM) |
| P1 | 8 ms | FP32 | 4 blocks | fitting FP32 policy |
| P2 | 8 ms | FP32 | 6 blocks | memory-minimal FP32 |
| P3 | 8 ms | FP16 | none | precision-only |
| P4 | 8 ms | FP16 | attention + whole 0–2 | **proposed default (cheap-first)** |
| P5 | 8 ms | FP16 | 6 blocks | memory-minimal |
| P6 | 16 ms | FP32 | none | repository reference |
| P7 | 16 ms | FP16 | none | fast reference |
| P8 | 8 ms | FP16 | attention only | cheap-first ablation |
| P9 | 8 ms | FP16 | attention + whole 0–1 | mid-memory point |
| P10 | 16 ms | FP32 vs FP16 | none | precision equivalence at a fitting hop |

Primary table: peak allocated/reserved, step seconds, projected epoch time,
whether it fits, validation PESQ/STOI/SI-SNR after the fixed budget.

## Results already measured

The policy benchmark is in
[`docs/tfgridnet_memory_policy_bench.md`](tfgridnet_memory_policy_bench.md)
and `results/paper_runs/tfgridnet_pareto_8ms_fp16.json` (20 measured steps per
fresh process):
- 8 ms FP32 no checkpoint: OOM at 12.72 GiB allocated / 13.03 GiB reserved.
- 8 ms FP16 no checkpoint: 11.12 GiB, 13.04 GiB reserved, 3.89 s/step.
- 8 ms FP16 attention only: 9.83 GiB, 11.16 GiB reserved, 3.92 s/step.
- 8 ms FP16 whole 0–2: 6.36 GiB, 8.24 GiB reserved, 5.61 s/step.
- 8 ms FP16 attention + whole 0–2: 5.84 GiB, 8.24 GiB reserved, 5.62 s/step.
- 8 ms FP16 whole 0–3: 4.77 GiB, 6.80 GiB reserved, 6.17 s/step.
- 8 ms FP16 attention + whole 0–3: 4.51 GiB, 6.80 GiB reserved, 6.19 s/step.
- 16 ms FP32 no checkpoint: 9.90 GiB, 10.48 GiB reserved, 2.52 s/step.
- 16 ms FP16 no checkpoint: 5.63 GiB, 5.92 GiB reserved, 2.39 s/step.

Quality equivalence is measured, not assumed:
- 200 steps, `whole 0–2` vs `attention + whole 0–2`: losses bit-identical
  (max abs diff 0.0), PESQ 1.616 → 1.692 → 1.813 → 1.783.
- 100 steps, no checkpoint vs attention only: losses bit-identical, PESQ
  1.616 / 1.692; 3.862 vs 3.858 s/step with cache release; 9.125 s/step
  without it after validation reserves 14.58 GiB.
- 200 steps, 16 ms FP32 vs FP16: FP16 5.63 vs 9.90 GiB allocated, 2.59 vs
  2.73 s/step, final PESQ 1.889 vs 1.891; loss trajectories not bit-identical
  (mean abs diff 0.37), so this is quality-neutral within proxy noise, not
  exact equivalence.
A 100-epoch 8 ms run is not planned on this GPU; the projected cost is tens of
days, and the paper's claim is feasibility/Pareto, not a new SOTA score.

## Threats to validity

- Single GPU, single model size, single dataset; no multi-GPU or A100 check.
- The 200-step, 100-step, and precision proxies are short-horizon quality checks, not a
  full training
  reproduction. It can detect divergence or a large quality gap, not a
  0.01-PESQ difference.
- FP16 complex operations are experimental; the numerical-equivalence check
  must include loss and gradient norms, not only "it runs".
- `checkpoint` recomputation changes floating-point ordering; our 200-step and
  100-step pairs show bit-identical loss values, but gradient-level bitwise
  equality is not claimed.
- The quality evidence is a short-horizon proxy trajectory (200/1,000 steps on
  a 20-utterance subset), not a final PESQ comparison; a full-recipe or
  pretrained run is still needed for a venue submission.
- The current model uses the repository's negative SI-SNR objective, not the
  paper's full Wav+Mag+MC objective. The compute-policy comparison is
  objective-agnostic, but the paper must state this deviation.

## Venue and framing

- Method/systems: ICASSP/EUSIPCO/IEEE SPL, or an efficiency workshop. The
  selective policy and Pareto protocol are the core.
- Benchmark/reproducibility: Speech Communication or IEEE Access. The
  contribution is the measured frontier and open harness.
- Do **not** frame the paper as "we trained TF-GridNet at 8 ms and got a
  better PESQ"; the evidence does not support a SOTA claim.

## Artifact map

- `scripts/bench_tfgridnet_policies.py` — isolated per-policy memory/time
  harness; writes JSON.
- `scripts/bench_tfgridnet_subblocks.py` — sub-block sweep.
- `speech_denoising/training/tfgridnet_policy.py` — quality-proxy trainer
  (checkpoint/FP16, equal-step budget, validation, conditional cache release).
- `scripts/train_tfgridnet_policy.py` — CLI for the proxy.
- `results/paper_runs/tfgridnet_pareto_8ms_fp16.json` — consolidated 20-step
  Pareto table.
- `results/paper_runs/tfgridnet_dns4_8ms.json` — 4-layer boundary check.
- `results/paper_runs/tfgridnet_policy_proxy/` — proxy runs and
  `quality_equivalence_200.json` / `quality_equivalence_100.json`.
- `docs/tfgridnet_memory_policy_bench.md` and
  `docs/tfgridnet_paper_draft.md` — measured table and draft.
- `docs/tfgridnet_submission_checklist.md` — claim/evidence matrix and
  pre-submission blockers.

## Next concrete steps

Done: sub-block checkpointing is implemented and measured; 200-step
`whole 0-2` vs `attention + whole 0-2` losses are bit-identical while both
learn (PESQ 1.62 → 1.81); 100-step `none` vs `attention only` is also
bit-identical with a 1.29 GiB saving at 3.86 s/step; the A/B/A control shows
no step-time penalty. The 4-layer DNS check fits at 8 ms and saves 0.54 GiB
for 9 % step time, so the "free" claim is six-layer-specific. The Pareto
figure and draft are in
`results/paper_runs/tfgridnet_pareto_8ms_fp16.{svg,png}`,
`docs/tfgridnet_memory_policy_bench.md`, and
`docs/tfgridnet_paper_draft.md`.

1. Both 1,000-step arms are complete (PESQ 2.123, losses identical). The
   post-run A/B/A controls found no attention penalty in either a fast or a
   slow session; absolute step time is environment-sensitive (see
   `timing_controls.json`).
2. Add a second GPU to test whether cheap-first generalizes beyond one device.
3. Decide venue: method/systems workshop vs benchmark track, based on the
   longer quality result.
