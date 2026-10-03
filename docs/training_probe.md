# Bounded training-resource probe

This is a device and waveform feasibility check. It constructs one random-weight
model from the current draft config, checks one training batch with backward,
then checks one complete validation utterance without backward. It creates no
optimizer, takes no parameter updates and writes no checkpoint or quality score.
The train/validate entry points retain their existing synthetic-only behavior.

## Inputs and selection

An existing [complete audio preflight](training_preflight.md) is required. Its
manifest byte hash, canonical embedded plan, source/config fingerprints, full
audio counts and shared split must still match the checkout. A GTCRN preflight
can be reused for LiSenNet and TF-GridNet because the audio and speaker split are
shared; the selected model's config is separately rebuilt and recorded. The
preflight is local evidence, not independently authenticated dataset provenance.
Selected WAV hashes are checked again through `TrainingSubset` on access.

Selection is deliberately small and fixed:

- The first four IDs in the sorted planned training subset form one full draft
  batch. They are cropped together at their existing deterministic per-ID offsets,
  using seed 42, epoch 1 and 64,000 samples; valid-length masking is unchanged.
- The longest planned validation utterance is read in full, with lexical ID order
  breaking ties. Its forward/loss check uses batch size one and eval mode.
- TF-GridNet uses the **six-layer local random-weight profile**, not the four-layer
  DNS pretrained model. No checkpoint or fixed test audio is opened.

These are probe samples, not the future runner's shuffled epoch. Passing does not
prove that every crop, seed or utterance will be stable. A constant selected clean
crop is an error under the existing loss; it is not skipped or replaced.

## Conditions and report

The device must be explicit (`cpu` or `cuda`); there is no fallback. The model and
audio use FP32 tensors, with autocast disabled. Existing TF32, cuDNN benchmark,
deterministic-algorithm and thread settings are recorded rather than changed.
Thus FP32 tensor dtype is not a guarantee about every backend's internal
arithmetic. This probe does not freeze the formal training runtime or enable
deterministic GPU training.

Parameters must remain unchanged. Buffers (including BatchNorm running state),
module modes and RNG are restored, and gradients are discarded. Validation starts
from the original random model state, not buffers updated by the train-mode pass.
CPU snapshots also include nonpersistent buffers. They consume host memory but
are excluded from GPU peak allocation; no optimizer moments are allocated.

New JSON reports are published atomically below `results/training_probe/`, without
overwriting inputs or existing outputs. Reports record selection IDs, crop offsets,
valid lengths, model config/parameter counts, source fingerprints, preflight hash,
runtime and stage diagnostics. No numeric loss is presented as a quality score.
CUDA timings synchronize around each case; they include transfers, first-call
setup, finite checks and the parameter-preservation check. They are **not RTF** and
must not be compared with the historical CPU benchmark timings. Reserved-memory
peaks include allocator cache retained from earlier work in the same process.

An out-of-memory or finite-value failure produces a diagnostic report and a
nonzero exit status when normal cleanup/publication is possible. The probe stops
at the failing case; it does not shrink batches, switch profiles or continue to
validation. Fatal runtime/device errors may prevent reporting. Input/source
changes during execution also prevent publication. Use a new filename for retry.

Even a passing probe leaves `execution_ready: false`. Optimizer state, repeated
epochs, all validation utterances, GPU resume reproducibility and the production
training/validation wiring remain to be checked. In particular, the current
production validation component accepts `Subset(VoiceBankDataset)`; integrating
the newer plan-bound `TrainingSubset` is still pending. This command does not
call that component or claim a complete validation/selection result.

## Windows commands

From the repository root, run the synthetic guards and toy backward check first:

```bat
venv\gpu\Scripts\python.exe -m scripts.smoke_training_probe
```

It runs a CPU toy model, including injected OOM/nonfinite/parameter-change cases,
and a CLI fixture with optimizer construction forbidden. Loader overrides and
synthetic preflight records exist only inside the test. A narrower
`--metadata-only` mode needs NumPy/PyYAML but neither Torch nor SoundFile.

With the existing complete train data and preflight report, run one model:

```bat
venv\gpu\Scripts\python.exe -m scripts.probe_training --model gtcrn --device cuda --manifest data/voicebank-demand-16k/train_manifest.json --preflight results/training_preflight/gtcrn_audio.json --output results/training_probe/gtcrn_cuda.json
```

After inspecting that report, use `--model lisennet` or `--model tfgridnet` with a
different output filename to check the other draft architectures. Each command
is a separate process; none enables real-data training. Source/checkpoint assets
are never downloaded by this tool.

Generated reports are ignored by Git. Keep a copy of the preflight and probe
reports outside a temporary net-café session if they need to survive changing
computers. These resource observations belong to this machine/runtime, unlike
the portable manifest and split hashes.
