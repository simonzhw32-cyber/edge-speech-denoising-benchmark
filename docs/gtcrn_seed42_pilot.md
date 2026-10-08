# GTCRN random seed-42 pilot — 2026-10-05

Completed 25 training epochs and 25 independent validations.
Final optimizer steps: 67,375. Validation selected epoch 24 (64,680 steps).
Random initialization, CUDA FP32, AdamW, negative zero-mean SI-SNR loss.
Training: 10,778 utterances. Validation: 794 utterances from train.
Fixed test: 824 utterances, all four metrics complete, zero failures.

| Metric | Result |
|---|---:|
| PESQ-WB | 2.57246453 |
| STOI | 0.93427571 |
| SI-SNR | 18.70301644 dB |
| SI-SNR improvement | 10.25748032 dB |

RTX 5090 RTF: 0.001884925; includes STFT/network/iSTFT, excludes
I/O, transfers and metric computation. This is not edge-device latency.

This is a single-seed pilot under a draft unified protocol, not an exact
reproduction of the official training recipe or a matched three-model comparison.
The run stopped at epoch 25; do not automatically continue to 100 epochs.
Do not use test results for checkpoint selection or tuning.

Evidence: ../training_reports/gtcrn_random_seed42/
The original JSON reports and checkpoint identities are preserved.
learning_curve.csv contains all 25 training and validation losses.
Training uses crops; validation uses complete utterances.

Install full evaluation dependencies:
python -m pip install -e . -r requirements-evaluation.txt
PESQ may require a C compiler.

Evaluate restored checkpoints using scripts.evaluate_trained, not the official
pretrained evaluation entry point. The original report's training_run.global_step
refers to the selected epoch, not the final training epoch.

The experiment archive is uploaded separately as a Release asset.
Verify archive.sha256 before extracting it into the repository root.
Data and external nohup logs are not included in that archive.

Training and evaluation used uncommitted changes based on 6c2c965.
Source hashes in the historical records identify those exact files.
Committing does not change those historical records. The original Release-tag resume logic
also compares Git metadata, so resuming after a commit may be rejected.
The updated runner ignores Git-only changes, but still rejects changed training
sources; it does not silently migrate this historical run.
Do not edit old run metadata to bypass this check.

Next: extend the real training/evaluation workflow to LiSenNet and six-layer
TF-GridNet, with an agreed experimental budget.

## Restore the original evaluation

Use a separate clean checkout of `gtcrn-seed42-pilot-25ep` and install the desired
Torch build first. Download the Release archive and `archive.sha256` to that
checkout. In a Linux terminal:

```sh
sha256sum -c archive.sha256
tar -xzf gtcrn_seed42_epoch25_20261005_204146.tar.gz
python -m pip install -e . -r requirements-evaluation.txt
python -m scripts.prepare_data --split test
python -m scripts.evaluate_trained \
  --run-dir results/training_runs/gtcrn_random_seed42 \
  --manifest data/voicebank-demand-16k/test_manifest.json \
  --output results/trained_reports/gtcrn_seed42_replay_test824.json \
  --device cuda --metrics pesq stoi si_snr si_snr_improvement --threads 4
```

The archive excludes audio. Keep existing historical test reports; the replay
writes a new filename. Exact numeric reproducibility also depends on the recorded
runtime/backend, not just the checkpoint hash.
