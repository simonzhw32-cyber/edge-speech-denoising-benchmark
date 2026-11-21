# TF-GridNet memory-policy paper — submission checklist

Status: working draft v0.6, 2026-11-13. Not submission-ready until E5 is
resolved or the paper is explicitly scoped to one GPU.

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| 8 ms FP32 does not fit 16 GB at batch 1 | OOM after 12.72 GiB allocated; 2.86 GiB request | measured |
| FP16 + checkpointing makes 8 ms fit | 5.84 GiB training step for attention + whole 0–2 | measured |
| Attention checkpointing is cheap | 1.29 GiB for 0.03 s at 20 steps; 3.858 vs 3.862 s at 100 steps | measured |
| Hybrid shifts the Pareto | 6.36 → 5.84 GiB; no consistent step-time penalty across two A/B/A controls | measured |
| Optimization is unchanged | 200-step and 100-step paired losses bit-identical | measured |
| Longer learning signal | 1,000-step uniform arm PESQ 2.123 | measured |
| Attention arm also learns at 1,000 steps | p4 endpoint PESQ 2.122954; all 10 checkpoints loss-identical | measured |
| RNN sub-block checkpointing is dominated | all-subblock 2.78 GiB / 7.29 s | measured |
| 4-layer boundary | 0.54 GiB for +9 % step time | measured |
| FP16 is quality-neutral at 16 ms | 5.63 vs 9.90 GiB; final PESQ −0.003; loss trajectories not bit-identical | measured |
| Generalizes to a second GPU | — | missing (E5) |

## Artifacts to attach

- `results/paper_runs/tfgridnet_pareto_8ms_fp16.json`
- `results/paper_runs/tfgridnet_pareto_8ms_fp16.svg` / `.png`
- `results/paper_runs/tfgridnet_quality.svg` / `.png`
- `results/paper_runs/tfgridnet_dns4_8ms.json`
- `results/paper_runs/tfgridnet_policy_proxy/quality_equivalence_200.json`
- `results/paper_runs/tfgridnet_policy_proxy/quality_equivalence_100.json`
- `results/paper_runs/tfgridnet_policy_proxy/quality_equivalence_1000.json`
- `results/paper_runs/tfgridnet_policy_proxy/timing_analysis.json`
- `results/paper_runs/tfgridnet_policy_proxy/timing_controls.json`
- `results/paper_runs/tfgridnet_policy_proxy/precision_equivalence_200.json`
- `results/paper_runs/tfgridnet_policy_proxy/final_summary.md`
- `results/paper_runs/tfgridnet_policy_proxy/artifact_manifest.json`
- `scripts/bench_tfgridnet_policies.py`
- `scripts/bench_tfgridnet_subblocks.py`
- `scripts/plot_tfgridnet_pareto.py`
- `scripts/plot_tfgridnet_quality.py`
- `speech_denoising/training/tfgridnet_policy.py`
- `docs/tfgridnet_memory_policy_bench.md`
- `docs/tfgridnet_paper_draft.md`
- `docs/tfgridnet_experiment_matrix.md`
- `results/paper_runs/tfgridnet_policy_proxy/experiment_matrix.json`

## Before submission

1. p4 complete; loss hash matches p3 and both endpoints are PESQ 2.123.
2. Attention-arm endpoint and E7 precision result are in the paper, abstract,
   and checklist.
3. Run E5 on a 24 GB GPU, or explicitly scope the paper to one 16 GB card.
4. Re-run the Pareto on the submission machine and report GPU/driver versions.
5. Replace the 20-utterance proxy with a larger validation set if time allows.
6. Timing controls complete; no consistent attention penalty. Absolute times
   remain environment-sensitive, so report them descriptively rather than as
   a benchmark.
7. Add author/affiliation, venue template, and references.
8. Archive code and raw JSON with a DOI (Zenodo) if the venue requires it.

## Known deviations to state

- Repository negative SI-SNR objective, not TF-GridNet's full Wav+Mag+MC.
- 16 ms fallback is not paper-faithful; the paper studies 8 ms.
- Single GPU, one model family, short proxy horizons.
- FP16 complex operations emit a PyTorch experimental warning; the 16 ms
  precision pair is quality-neutral (final PESQ −0.003) but not bit-identical.
