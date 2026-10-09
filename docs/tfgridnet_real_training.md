# TF-GridNet six-layer pilot

The original local six-layer, mono 16 kHz architecture is supported for random
initialization only. It is offline and uses future frames; it is not the four-layer
DNS pretrained profile and is not a streaming edge model.

## Hardware evidence

RTX 5090, FP32, four-second crops: batch 4 failed with CUDA OOM.
Diagnostic batch 1 completed forward/backward (peak allocated 10,553,112,576 bytes).
The longest validation utterance, 185,760 samples, also completed inference
(peak allocated 5,183,784,448 bytes). These are bounded checks without updates,
not evidence that a complete epoch has already run.

## Execution settings

The shared draft plan remains effective batch 4. TF-GridNet explicitly records
micro_batch_size=1, gradient_accumulation_steps=4 and
accumulation_reduction=actual_group_utterance_mean in run.json.
Each utterance loss is divided by its group's actual count; gradients are clipped
once and AdamW steps once per group. The final two utterances are divided by two,
not four. A complete 10,778-item epoch has 2,695 optimizer updates.
This execution change is not guaranteed numerically equivalent to direct batch 4.
FP32, seed, crop selection, data split and validation selection are retained.
The training loss remains the shared negative SI-SNR pilot objective, not an exact
reproduction of the paper's training recipe.

## Commands

First run `python -m scripts.smoke_tfgridnet_training` and the existing
`python -m scripts.smoke_real_training`. All audio is generated and test metrics
are mocked in these fixtures; no real quality result is claimed.

Train one complete epoch:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m scripts.train --real \
  --model tfgridnet --manifest data/voicebank-demand-16k/train_manifest.json \
  --preflight results/training_preflight/gtcrn_audio.json \
  --run-dir results/training_runs/tfgridnet_random_seed42_accum4 \
  --epoch 1 --seed 42 --device cuda --initialization random
```

Then run independent validation with scripts.validate --real, the same manifest,
preflight, run-dir, epoch and device. Fixed test audio is not used for selection.
The first complete epoch is an execution check; later training budget should be
based on measured runtime and cost. It does not automatically launch 25 epochs.

## Existing experiments

The installer changes shared execution source fingerprints. Historical GTCRN
and LiSenNet reports and checkpoint files are not edited. Replay those runs in a
separate checkout of their original Release source snapshot. Git-only changes can
resume a run; changes to execution sources or accumulation settings cannot.
Interrupted epochs are not resumable within an epoch. Keep partial run artifacts
for diagnosis; do not rewrite metadata or restart a launcher blindly.
