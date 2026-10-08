# Verified real-data GTCRN and LiSenNet runner

The real runner connects the existing verified VoiceBank train view to one GTCRN or LiSenNet
epoch, epoch-boundary state persistence, and a separate complete held-out
validation command. GTCRN has completed the [25-epoch seed-42 pilot](gtcrn_seed42_pilot.md).
LiSenNet real-data execution is pending; the draft 100-epoch / three-seed proposal
is not an accepted final budget.

## Inputs and boundaries

`scripts.train --real` requires the complete prepared train manifest, its current
preflight report, a new run directory below `results/training_runs/`, an epoch,
seed, device and explicit initialization mode. `random` starts from newly seeded
parameters for the explicitly selected model. LiSenNet accepts only `random`.
`pretrained` is fine-tuning and additionally requires the
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

## LiSenNet first real epoch

After the full synthetic smoke check succeeds on the training machine, use a
new run directory. The GTCRN audio preflight may be reused when its current
source, manifest, protocol and split bindings match; the model plan is recomputed
for LiSenNet. Never reuse the GTCRN run directory.

```sh
python -m scripts.smoke_real_training
python -u -m scripts.train --real --model lisennet \
  --manifest data/voicebank-demand-16k/train_manifest.json \
  --preflight results/training_preflight/gtcrn_audio.json \
  --run-dir results/training_runs/lisennet_random_seed42 \
  --epoch 1 --seed 42 --device cuda --initialization random
python -u -m scripts.validate --real \
  --manifest data/voicebank-demand-16k/train_manifest.json \
  --preflight results/training_preflight/gtcrn_audio.json \
  --run-dir results/training_runs/lisennet_random_seed42 \
  --epoch 1 --device cuda
```

The saved run determines validation's model. For subsequent epochs, repeat the
separate training and validation commands with the same epoch number in each.
A complete 25-epoch pilot is a proposed next-stage budget, not proof of convergence
or official-recipe reproduction. Keep TF-GridNet separate until its six-layer
configuration is supported and the resource budget is recorded.

LiSenNet retains its upstream-style normalization and two Griffin-Lim iterations.
Phase reconstruction detaches the predicted magnitude; waveform-loss gradients
still flow through the reconstructed magnitude. This shared negative SI-SNR
experiment is not LiSenNet's original multi-loss training recipe.

After the agreed pilot ends, selection uses validation alone. Evaluate a new
report with the shared trained-model entry point:

```sh
python -u -m scripts.evaluate_trained \
  --run-dir results/training_runs/lisennet_random_seed42 \
  --manifest data/voicebank-demand-16k/test_manifest.json \
  --output results/trained_reports/lisennet_random_seed42_test824.json \
  --device cuda --metrics pesq stoi si_snr si_snr_improvement --threads 4
```

Install `requirements-evaluation.txt` with the same interpreter first. The test
manifest must cover all 824 fixed items; the test must not guide training choices.

## Resume and historical sources

Git commit/worktree metadata is recorded at run creation as provenance. A
Git-only change is now allowed when every other run field remains identical,
including source fingerprints, manifest, preflight, plan, initialization and
runtime. The saved original Git metadata is never rewritten.

This is not permission to resume through training-code changes. In particular,
the archived GTCRN run predates these source edits. Use the
`gtcrn-seed42-pilot-25ep` checkout for its original replay/evaluation; use a new run
for the updated implementation. Do not alter historical `run.json` hashes to
force an old run through a new checkout.


## LiSenNet pilot completed — 2026-10-08

LiSenNet random seed 42 completed 25 real training epochs and 25 independent validations. Validation selected epoch 21; all 824 fixed test utterances were evaluated with PESQ, STOI, SI-SNR and SI-SNR improvement. See [experiment results](lisennet_seed42_pilot.md).

This update supersedes earlier statements that LiSenNet real execution is pending. TF-GridNet real training and a complete three-model comparison remain pending; edge-device efficiency is not established.
