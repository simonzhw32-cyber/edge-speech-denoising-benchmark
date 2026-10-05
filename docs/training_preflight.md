# Training-data preflight

This stage checks an **existing** prepared train manifest and its local audio.
It does not download missing data, construct a model, update parameters or run
validation scores. Its report is a required input to the separate guarded GTCRN
real-training and validation entry points.

## What is bound

`load_training_inputs()` reads the current pinned dataset configuration, draft
training protocol and random-weight model configuration, then rebuilds the
[metadata plan](training_protocol.md). The 11,572 official train-source items
are divided using the existing speaker hash ranking (split seed 42, two held-out
speakers). Speaker names and subset counts come from the manifest. TF-GridNet
uses the six-layer local profile; the four-layer DNS checkpoint remains a
separate pretrained baseline.

`TrainingSubset(manifest_path, plan, split)` accepts only `train` or `validation`.
It requires the entire plan to match the freshly rebuilt canonical plan, even
if an edited split has a new internally consistent hash. Its indices preserve
planned ID order and partition the parent train source without overlap.

The subset returns complete mono FP32 pairs through the existing
`VoiceBankDataset` with file verification enabled. The runner owns paired
cropping and padding; validation continues to receive full utterances. File
hashes are checked on access, so an older preflight report cannot authorize
changed local audio. The fixed 824-item test is never opened by this tool and
cannot be used as a parent training manifest.

## Audio checks and report

Every noisy/clean pair must have:

- Bytes matching the SHA-256 recorded in the manifest; decoding uses those same bytes.
- Mono 16 kHz audio, matching recorded sample lengths, with at least two samples.
- Finite FP32 decoded samples and a nonconstant full clean reference with finite
  centered energy above the draft loss epsilon.
- Files inside the manifest directory, with no symlinks in their audio paths.

Any failure aborts the scan without publishing a success report. Input manifest
bytes and configuration/source fingerprints are rechecked before publication.
The tool does not rewrite WAV files, manifest rows, plans or benchmark reports.
It verifies audio against prepared metadata; it does not re-extract parquet or
independently prove the manifest's declared upstream provenance.

Reports are new `.json` files below `results/training_preflight/`, which is already
ignored by Git. Publication uses a same-directory temporary file and hard link:
complete bytes become visible at once, and existing outputs cannot be replaced.
Use a new filename for another check. Input and historical-report paths are
protected. A filesystem without hard-link support will fail rather than fall
back to a partial or overwriting write.

The report records the embedded draft plan and its hash, parent rows, source,
protocol and split hashes, verified counts/durations and source fingerprints.
The metadata plan still says `audio_files_verified: false`; it is the unchanged
planner product. The enclosing preflight report records the completed audio
scan separately. Neither object becomes a training checkpoint or evaluator
manifest.

A successful scan is a point-in-time observation, not a lock over the dataset.
Avoid editing data or source during a scan; runtime dataset access must continue
to verify files. No device, memory, model stability or timing claims are made.
Full-utterance clean-reference energy is checked in FP32, but this is not a proof
that every deterministic crop is suitable for SI-SNR. Crop feasibility and
hardware selection remain pending, so `execution_ready` remains **false**.
The draft protocol and its budgets have not been frozen.

## Commands

From the repository root, check the new logic with generated local metadata/WAVs:

```sh
python -m scripts.smoke_training_preflight
```

This needs the existing NumPy, SoundFile, PyYAML and PyTorch dependencies.
It does not load a speech model or take optimizer steps. Tiny fixture metadata
and loader overrides are confined to the test; production CLI inputs retain
the pinned 11,572-item requirement. Symlink rejection is exercised when the
operating system permits creating the fixture symlink.

For environments without audio/Torch dependencies:

```sh
python -m scripts.smoke_training_preflight --metadata-only
```

This narrower mode checks plan binding, shared splits, input/output protection
and CLI rejection. It does not test WAV decoding or tensor dataset views.

Only when the complete train data is already prepared locally, run:

```sh
python -m scripts.preflight_training --model gtcrn --manifest data/voicebank-demand-16k/train_manifest.json --output results/training_preflight/gtcrn_audio.json
```

The command checks all 11,572 pairs with no limit or fallback download. It will
fail if train data is absent. `--model lisennet` and `--model tfgridnet` select
their draft constructor configuration while retaining the same split. They do
not execute those models. A separate report filename is required for each run.

The production GTCRN entry points bind this report to the plan and experiment
identity, record the selected device/runtime and retain separate epoch/validation
history. A successful preflight alone is not permission to consume shared GPU
resources; initialization and a small first-run budget must still be confirmed.
