# TF-GridNet 8 ms memory-policy study — final measurement summary

Date: 2026-11-13. Hardware: RTX 5060 Ti 16 GB, driver 617.42, PyTorch
2.11.0+cu128. Model: six-layer TF-GridNet, 8,381,504 parameters, 8 ms hop
(512/128), 4 s segments, batch 1.

## Main result

FP16 + cheap-first hierarchical checkpointing fits the paper-faithful 8 ms
configuration in 5.84 GiB allocated / 13.50 GiB reserved. A 200-step paired
run and a 100-step attention-only run are loss-bit-identical to the uniform
baseline. Both 1,000-step arms reach PESQ 2.123 on the fixed 20-utterance
proxy with identical losses (max abs diff 0.0, SHA-256
`fceab86d02eb5021...`).

## Pareto (8 ms, FP16, 20 measured steps)

| Policy | Alloc GiB | Reserved GiB | Step s |
|---|---:|---:|---:|
| none | 11.12 | 13.04 | 3.889 |
| attention | 9.83 | 11.16 | 3.917 |
| whole 0 | 9.54 | 11.14 | 4.465 |
| attention + whole 0 | 8.50 | 11.11 | 4.487 |
| whole 0–1 | 7.95 | 8.25 | 5.032 |
| whole 0–2 | 6.36 | 8.24 | 5.606 |
| attention + whole 0–2 | **5.84** | **8.24** | **5.620** |
| whole 0–3 | 4.77 | 6.80 | 6.173 |
| attention + whole 0–3 | 4.51 | 6.80 | 6.187 |
| whole 0–4 | 3.19 | 5.36 | 6.747 |
| all sub-blocks | 2.78 | 3.92 | 7.289 |

## Timing controls

Two A/B/A controls (30 steps each): fast session uniform/attention/uniform
5.601 / 5.624 / 5.611 s; slow session 7.297 / 6.868 / 7.406 s. The attention
arm is not slower in either. Absolute step time is environment-sensitive;
the paper claims memory and trajectory equivalence, not a step-time speedup.

## Precision control (16 ms, 200 steps, no checkpointing)

| Precision | Alloc GiB | Reserved GiB | Mean step s | Final PESQ |
|---|---:|---:|---:|---:|
| FP32 | 9.90 | 12.90 | 2.73 | 1.891 |
| FP16 | 5.63 | 12.08 | 2.59 | 1.889 |

FP16 saves 4.27 GiB and is slightly faster; final PESQ differs by -0.003.
Losses are not bit-identical (mean abs diff 0.37; final SI-SNR 1.14 dB lower),
so this is quality-neutral within proxy noise, not exact equivalence. At 8 ms
FP32 OOMs; FP16 is required.

## Artifacts

- `tfgridnet_pareto_8ms_fp16.json` / `.svg` / `.png`
- `tfgridnet_policy_proxy/quality_equivalence_200.json`
- `tfgridnet_policy_proxy/quality_equivalence_100.json`
- `tfgridnet_policy_proxy/quality_equivalence_1000.json`
- `tfgridnet_policy_proxy/timing_analysis.json`
- `tfgridnet_policy_proxy/timing_controls.json`
- `tfgridnet_policy_proxy/precision_equivalence_200.json`
- `tfgridnet_dns4_8ms.json`
- `docs/tfgridnet_paper_draft.md`
- `docs/tfgridnet_memory_policy_bench.md`
- `docs/tfgridnet_reproduce.md`
