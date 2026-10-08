# LiSenNet seed-42 pilot — 2026-10-08

Completed 25 real training epochs and 25 independent validations.
Random initialization, seed 42, native_default profile.
Training: 10,778 utterances; validation: 794 held-out train utterances.
Final optimizer steps: 67,375.
Validation selected epoch 21, mean negative SI-SNR loss: -14.784733074062718.
Selected checkpoint: checkpoints/epoch_0021.pt
Checkpoint SHA-256: 470a6f3be2d77355afca21e99cd3ab01232389c30bc1d14cd8307323d2153a80

The selected checkpoint was evaluated on all 824 fixed test utterances.

| Metric (higher is better) | GTCRN seed 42, epoch 24 | LiSenNet seed 42, epoch 21 |
|---|---:|---:|
| PESQ-WB | 2.572464525554944 | 2.5722881885408198 |
| STOI | 0.9342757059390325 | 0.9344279064590634 |
| SI-SNR (dB) | 18.703016442807773 | 18.631195078462376 |
| SI-SNR improvement (dB) | 10.257480317426278 | 10.185658953080884 |

Both pilots use the same split, seed and 25-epoch budget.
These are single-seed results under the draft shared protocol, not exact
reproductions of the papers' training recipes. The small differences do
not establish a stable winner. Edge-device efficiency remains unverified.

Evidence: training_reports/lisennet_random_seed42/
Weights and execution-source snapshot: Release lisennet-seed42-pilot-25ep.
The archive predates this publication documentation update and excludes
the audio dataset. See run.json for source fingerprints and run identity.

Restore the archive into a separate directory. The selected epoch 21
checkpoint is for inference; epoch 25 is the final training checkpoint.
Preserve the original run metadata when restoring or resuming.
