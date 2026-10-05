# Verified real-data GTCRN runner

The real runner connects the existing verified VoiceBank train view to one GTCRN
epoch, epoch-boundary state persistence, and a separate complete held-out
validation command. This is an implementation milestone, not evidence that a
real experiment has run or that the draft 100-epoch / three-seed proposal has
been approved.

## Inputs and boundaries

`scripts.train --real` requires the complete prepared train manifest, its current
preflight report, a new run directory below `results/training_runs/`, an epoch,
seed, device and explicit initialization mode. `random` starts from newly seeded
GTCRN parameters. `pretrained` is fine-tuning and additionally requires the
checksum-pinned official VCTK-DEMAND GTCRN checkpoint for epoch one. The two modes
produce different immutable run identities and must never share a run directory.

The runner recomputes the canonical train-only speaker split, verifies audio bytes
as items are read, uses deterministic paired crops, updates with AdamW and negative
SI-SNR, clips finite gradients, and writes a local checkpoint containing model,
optimizer, RNG and progress state. It does not read the fixed 824-item test set,
compute test metrics, download data, overwrite checkpoints or edit historical
reports.

`scripts.validate --real` is a separate process. It restores the named epoch,
checks its hash and progress, and evaluates every planned held-out train utterance
at full length with batch size one. The resulting validation report updates an
owned selection index. Training the next epoch is rejected until this validation
step is complete.

## Command shape

Random initialization:

```sh
python -m scripts.train --real --model gtcrn \
  --manifest data/voicebank-demand-16k/train_manifest.json \
  --preflight results/training_preflight/gtcrn_audio.json \
  --run-dir results/training_runs/gtcrn_random_seed42 \
  --epoch 1 --seed 42 --device cuda --initialization random
```

Pretrained fine-tuning uses a different run directory and appends
`--initialization pretrained --initial-checkpoint checkpoints/gtcrn_vctk.tar`.
Validate the completed epoch with the same manifest, preflight, run directory and
device:

```sh
python -m scripts.validate --real \
  --manifest data/voicebank-demand-16k/train_manifest.json \
  --preflight results/training_preflight/gtcrn_audio.json \
  --run-dir results/training_runs/gtcrn_random_seed42 \
  --epoch 1 --device cuda
```

Before launching either real initialization mode, choose it explicitly and agree
the first-run resource budget. A one-epoch integration run does not commit the
project to the draft 100 epochs or all three draft seeds. Do not use the fixed
test set for checkpoint selection.

The metadata-only guard check needs no audio or model execution:

```sh
python -m scripts.smoke_real_training --metadata-only
```

With the project dependencies installed, omit `--metadata-only` to exercise the
same orchestration on generated tensors. This does not read VoiceBank or claim a
real training result.

## Saved evidence

`run.json` records the plan, initialization/checkpoint provenance, runtime, Git
state and source fingerprints. Each epoch adds an immutable checkpoint and
training record. Independent validation adds its report and atomically refreshed
selection index. Resume requires the same runtime/device and rejects changed
source, inputs, state hashes, skipped epochs, duplicate epochs and incomplete
prior validation.
