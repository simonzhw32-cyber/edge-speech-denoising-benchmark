# Training and validation entry-point fixture

The separate `scripts.train` and `scripts.validate` commands now exercise an
end-to-end **synthetic** experiment. Real-data execution is still disabled. The
[training protocol](training_protocol.md) remains a draft; this fixture does not
freeze its proposed budget or select target hardware.

## What runs

The fixture uses the existing random-weight GTCRN adapter on CPU FP32, one thread
and deterministic algorithms. Four generated training waveforms are cropped or
padded to 1,024 samples, in batches of two; two complete generated utterances are
used for validation. Each epoch has two AdamW updates, with at most two epochs.
The optimizer settings, clipping and negative SI-SNR loss come from the draft,
but these tiny lengths, batch sizes and epoch limits are **fixture overrides**,
not the real 4-second / batch-4 / 100-epoch training recipe.

There is no pretrained initialization, VoiceBank audio, asset download or quality
benchmark. Fixture IDs are not VoiceBank IDs. Loss values here are software checks
and must not be added to the pretrained result table. The model/profile label is
GTCRN, and the run spec explicitly records synthetic scope and fixture settings.
LiSenNet and six-layer TF-GridNet retain their existing shared-loss backward
checks; this entry-point fixture does not establish that their training runs work.

`train_one_epoch()` in `training/runner.py` implements paired crops, deterministic
epoch ordering, retained partial batches, FP32 forward/backward, finite gradient
clipping and AdamW updates. Its returned mean is weighted by utterance count,
including the last partial batch. It computes no validation. This routine is a
building block; its caller remains responsible for using a verified train-only
plan before any future real-data execution.

## Separate commands

Describe the current training profile without importing PyTorch or opening data:

```sh
python -m scripts.train --describe --model gtcrn
python -m scripts.train --describe --model tfgridnet
```

To manually inspect a fixture, use a **new** run name:

```sh
python -m scripts.train --fixture --run-dir results/training_fixture/example --epoch 1
python -m scripts.validate --fixture --run-dir results/training_fixture/example --epoch 1
python -m scripts.train --fixture --run-dir results/training_fixture/example --epoch 2
python -m scripts.validate --fixture --run-dir results/training_fixture/example --epoch 2
```

These commands perform actual optimizer updates on generated waveforms. They are
optional; the smoke test already exercises this sequence in temporary run folders.
Do not run them as a request to train on VoiceBank. There is no `--manifest` or
real-data execution option at this stage. Training supports the GTCRN fixture
only and rejects skipped/repeated epochs, changed seeds/specs and a pending
previous validation. Epoch 3 is rejected. Both CLI entry points require an
explicit execution mode; the former configuration-only invocation is replaced
by `--describe`.

## Saved files and recovery

Each new run lives below the Git-ignored `results/training_fixture/`:

- `run.json`: immutable settings, identity, synthetic IDs and normalized source
  fingerprints. The current code/config must match on later commands.
- `checkpoints/epoch_0001.pt`: [epoch state](training_state.md), including buffers,
  optimizer moments, RNG and progress. New epoch paths are never overwritten.
- `training/epoch_0001.json`: complete epoch count, macro training loss, cumulative
  step count and checkpoint SHA-256.
- `validation/epoch_0001.json`: a complete independent synthetic validation report,
  bound to the checkpoint actually restored by `scripts.validate`.
- `selection.json`: history fingerprint, number of validated epochs, best report
  and relative checkpoint path. Lowest finite loss wins; exact ties choose the
  earlier epoch. The index is recomputed from saved reports, not caller memory.

The validation command checks the recorded checkpoint hash and restored progress,
then uses full-utterance, batch-one validation. Training cannot advance until the
previous report and its selection index are complete. Repeating an identical
validation is idempotent; a conflicting report is rejected.

JSON is staged in the destination directory before publication. Immutable files
use hard links to prevent replacement; only the owned selection index is replaced
atomically. The filesystem must support hard links. The files together are not a
single filesystem transaction: if publication stops after a report but before
selection, rerun validation to repair the index. A checkpoint without its training
record, a history gap, unknown file or changed hash is rejected and kept for
inspection; do not silently discard it or continue updates. Use a fresh fixture
run after investigating a failed training publication. Symlink paths and outputs
outside the dedicated fixture area are rejected. Execute only one command at a
time for each run; this fixture is not a concurrent job coordinator.

Resume uses the saved epoch state in the same runtime/device environment. Shuffle
order derives from seed and epoch; paired crop offsets retain the existing
utterance-based algorithm. No workers or mid-epoch cursor are used. This does not
prove portable or CUDA reproducibility. No benchmark timing settings are changed.

## Checks and remaining work

```sh
python -m scripts.smoke_training_runner --metadata-only
python -m scripts.smoke_training_runner
python -m scripts.smoke_training_state
python -m scripts.smoke_training_components
python -m scripts.smoke_evaluation_cli
```

The metadata-only mode exercises output protection, source/spec identity, report
replay, checkpoint hash rejection and recovery after a simulated selection-write
failure. Its dummy checkpoint bytes are never deserialized. It also checks the
no-PyTorch descriptions and explicit CLI execution modes.

The full check performs two uninterrupted synthetic GTCRN epochs in memory and
the equivalent train/validate sequence in separate subprocesses. It compares
parameters, buffers, optimizer moments, RNG, progress and epoch losses exactly,
checks frozen parameters and partial-batch weighting, verifies checkpoint selection and rejects skipped,
repeated or unvalidated epochs. Temporary files are removed afterwards. These
updates provide no evidence of speech quality, convergence or real-data memory
feasibility.

Next work is the verified full-train dataset/plan binding for the real trainer,
production validation/report entry-point wiring, an explicit hardware feasibility
run and protocol freezing. No full training or fixed-test evaluation should begin
from this fixture command.
