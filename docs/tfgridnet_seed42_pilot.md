# TF-GridNet seed-42 five-epoch pilot

Completed five real training epochs and five independent validations.
Random initialization, seed 42, local_6layer profile, CUDA FP32.
Microbatch size 1 with gradient accumulation over up to four utterances.
Training: 10,778 utterances; held-out validation: 794 utterances.
Final optimizer steps: 13,475.
Validation selected epoch 5; mean negative SI-SNR: -15.801807321138886.

The selected checkpoint was evaluated on all 824 fixed test utterances.

| Model | Training epochs | Selected epoch | PESQ | STOI | SI-SNR (dB) | SI-SNR improvement (dB) |
|---|---:|---:|---:|---:|---:|---:|
| GTCRN | 25 | 24 | 2.572465 | 0.934276 | 18.703016 | 10.257480 |
| LiSenNet | 25 | 21 | 2.572288 | 0.934428 | 18.631195 | 10.185659 |
| TF-GridNet | 5 | 5 | 2.585125 | 0.940414 | 17.764235 | 9.318699 |

These are single-seed pilots with different training budgets.
They do not establish an overall model ranking or reproduce the papers'
complete training recipes. Edge-device performance remains unverified.

Epoch 5 training took 5,703 seconds; validation took 241 seconds.
These are timings for that epoch, not averages over the complete run.

Evidence: training_reports/tfgridnet_random_seed42_accum4/
Weights and execution-source snapshot: Release tfgridnet-seed42-pilot-5ep.
The archive excludes the audio dataset and predates this publication note.
Restore into a separate directory and preserve the original run metadata.
