# Epoch state and validation components

These components support the [draft training protocol](training_protocol.md).
The separate `scripts.train` and `scripts.validate` commands now exercise a
[synthetic entry-point fixture](training_runner.md), with persisted reports and
selection history. The [real GTCRN runner](real_training.md) now wires these
components to verified train data; hardware feasibility and a frozen budget are
still needed.

## Experiment identity

`experiment_identity(plan, seed)` in `training/selection.py` binds an experiment
to its model/profile, training seed, protocol, speaker split and model settings.
The identity checks plan fingerprints but does not verify audio. TF-GridNet
training identities accept the local six-layer profile; the four-layer DNS
pretrained profile is rejected. The protocol remains a draft with an unselected
device. An identity does not make it execution-ready.

## State format

`save_training_state(path, model, optimizer, identity, completed_epoch=..., global_step=...)`
saves a local epoch-boundary format, distinct from upstream pretrained weights:

- Model parameters and buffers, per-module train/eval modes.
- AdamW parameter groups and moments, with named parameter ordering.
- Python, NumPy and PyTorch CPU/CUDA random states.
- Completed epoch, optimizer-step count, experiment identity and runtime versions.

Save after the last update of an epoch and before validation. Gradients are not
saved. A future loop must clear gradients and reconstruct the next epoch's
shuffle/crops from its seed and epoch. This does **not** resume a partial epoch,
a DataLoader iterator, worker-local random streams, a scheduler or AMP scaler.
The draft currently specifies FP32 with no scheduler.

The writer stages a complete file in its destination directory, flushes it and
publishes it with a hard link, refusing overwrite. The destination filesystem
must support hard links (for example NTFS); failure leaves an existing target
unchanged. Callers should use new epoch paths under their own run directory in
`checkpoints/`, never overwrite pretrained weights.

`load_training_state(path, model, optimizer, identity)` uses `weights_only=True`,
checks identity, model tensor signatures, optimizer type/order/moment shapes and
finite state, then restores the state strictly. A failed restoration rolls back
model, optimizer, RNG and modes. On success, gradients are cleared. The return
value identifies the completed epoch, next epoch, step count and file SHA-256.
Only load local experiment files you trust; this is a format check, not a
cryptographic authenticity guarantee.

This stage requires the same Python, NumPy, PyTorch and device configuration for
restoration, including CUDA RNG topology. The fixture tests CPU continuation
within one process. It does not prove bitwise CUDA reproducibility, portability
between computers or equivalence across software versions. Hardware, deterministic
kernel settings and loader policy still need recording in the future runner.

## Independent validation

`run_validation()` in `training/validation.py` runs outside the training loop:
complete FP32 mono utterances at 16 kHz, batch one, inference mode, and the shared
negative SI-SNR loss averaged per utterance. No cropping or padded batches are
accepted. It restores per-module modes and random states, and leaves gradients
alone. It does not compute PESQ/STOI or change benchmark timing conditions.

For production use, the dataset must be the plan-bound `TrainingSubset` (or an
equivalent `torch.utils.data.Subset`) of a verified complete train
`VoiceBankDataset`. The supplied plan is recomputed against its
parent metadata and compared to the identity and ordered validation IDs. Audio
bytes are checked when each item is read. Test data, arbitrary datasets and
incomplete validation sets are rejected. This component consumes existing local
data; it does not prepare or download it.

The caller must first restore the named checkpoint and pass its returned hash.
A report records that hash, epoch, identity, ordered-ID fingerprint, count and
macro loss. The component cannot independently prove that an arbitrary in-memory
model came from the supplied hash; both fixture and real validation entry points
own that loading/report-writing sequence. Model adapters are expected not to modify
buffers in eval mode; the component does not snapshot arbitrary forward side effects.

Synthetic datasets require `synthetic=True`; their report scope is
`synthetic_fixture`, never a real validation score.

## Selection

`select_checkpoint(best, candidate, identity, validation_ids)` validates report
coverage, scope, experiment identity and checkpoint hash. It selects the lowest
finite macro loss, with the earliest epoch breaking exact ties. Conflicting
reports for one epoch are rejected. It returns a copy rather than modifying a
caller-owned report. It neither deletes files nor tunes against the fixed test.

The fixture and real GTCRN orchestrators persist complete validation reports,
fold them through this rule and name the retained checkpoint. Saving/loading training state
alone does not persist the selection history. Fixture reports can only be selected
with the explicit `synthetic=True` flag.

## Check

```sh
python -m scripts.smoke_training_state --metadata-only
python -m scripts.smoke_training_state
python -m scripts.smoke_training_plan
python -m scripts.smoke_training_components
```

The first command checks identities and selection without importing PyTorch.
The full check performs a few **toy** AdamW updates in a temporary directory,
compares uninterrupted continuation with checkpoint restoration, and tests
rollback, overwrite rejection, malformed optimizer state, complete-utterance
validation and mode/RNG/gradient preservation. It does not update any of the
three speech models, load pretrained assets, read VoiceBank audio or generate
quality scores. The existing component test still covers all three adapters.
