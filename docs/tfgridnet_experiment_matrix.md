# TF-GridNet memory-policy experiment matrix

Generated 2026-11-13 from the run artifacts; see the JSON files for raw values.

| ID | Experiment | Setting | Result | Status | Artifact |
|---|---|---|---|---|---|
| E1 | 8 ms FP16 Pareto | 11 policies, 20 steps, fresh process | peak 11.12 → 2.78 GiB; step 3.89 → 7.29 s | done | [tfgridnet_pareto_8ms_fp16.json](../results/paper_runs/tfgridnet_pareto_8ms_fp16.json) |
| E2 | whole-3 vs attention + whole-3 | 200 steps, seed 42, same data order | loss max abs diff 0.0; PESQ 1.616/1.692/1.813/1.783 | done | [quality_equivalence_200.json](../results/paper_runs/tfgridnet_policy_proxy/quality_equivalence_200.json) |
| E3 | none vs attention-only | 100 steps, seed 42 | loss max abs diff 0.0; 11.12 vs 9.83 GiB; 3.862 vs 3.858 s | done | [quality_equivalence_100.json](../results/paper_runs/tfgridnet_policy_proxy/quality_equivalence_100.json) |
| E4 | 4-layer DNS boundary | 8 ms FP16, 20 steps | 2.17 vs 1.63 GiB; 0.173 vs 0.189 s (+9%) | done | [tfgridnet_dns4_8ms.json](../results/paper_runs/tfgridnet_dns4_8ms.json) |
| E5 | second GPU | 24 GB / data-center GPU | not run | pending | — |
| E6 | 1,000-step quality pair | 8 ms FP16, 1,000 steps, 20-utt proxy | PESQ 2.122956 (uniform) vs 2.122954 (attention); loss hash equal | done | [quality_equivalence_1000.json](../results/paper_runs/tfgridnet_policy_proxy/quality_equivalence_1000.json) |
| E7 | 16 ms FP32 vs FP16 | 200 steps, no checkpoint, seed 42 | 9.90 vs 5.63 GiB; final PESQ 1.891 vs 1.889; loss not bit-identical | done | [precision_equivalence_200.json](../results/paper_runs/tfgridnet_policy_proxy/precision_equivalence_200.json) |

E5 is the only missing experiment. The current draft is explicitly scoped to one 16 GB GPU; a 24 GB run would test cross-GPU generalization, not the within-GPU policy ordering.
