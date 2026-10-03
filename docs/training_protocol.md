# Unified training draft

Phase 3.1 adds a planning tool, not a trainer. The settings in
[training_protocol.yaml](../configs/training_protocol.yaml) are proposed shared
settings, not tuned hyperparameters or a reproduction of the authors' recipes.
`scripts.train` and `scripts.validate` now have an explicit
[synthetic entry-point fixture](training_runner.md); real-data execution remains disabled. The shared waveform
loss and crop/collator implementation is described in [components](training_components.md).
Existing pretrained baselines and the fixed test protocol are unchanged.

## Split policy

Use only the pinned complete **11,572-item train manifest**. Extract the speaker
prefix from each `pNNN_...` utterance ID. Rank speakers by SHA-256 of
`voicebank-speaker-split-v1:{seed}:{speaker}`, with a lexical tie-break. Hold out
the first two speakers, using split seed 42, and keep all their utterances in
validation. This is this project's proposed policy, not an official validation
split. Counts and speaker names are computed from the real manifest, not assumed.

The remaining train speakers and validation speakers must be disjoint, and their
utterance union must equal the parent train set. Split seed stays the same for
all models and all training seeds; changing it creates a different experiment.
The 824 fixed test utterances are never read by the planner and cannot be supplied
as its parent manifest. Test remains reserved for the final selected checkpoint.

The tool checks metadata counts, source/parquet provenance, row fingerprints,
sorted unique IDs, sample counts, hash syntax and train-path structure. It does
not open WAV files or verify the bytes of local audio or parquet. A later trainer
must use the verified dataset loader and verify the files before learning.
The JSON output is a **plan**, not a dataset manifest accepted by the evaluator.

## Proposed common budget

| Setting | Draft |
|---|---|
| Initialization | Random weights for every model |
| Seeds | 42, 43, 44 |
| Training segments | 64,000 samples (4 seconds), paired crop |
| Short input | Right zero-padding; padding excluded from loss |
| Batch size / epochs | 4 / 100; retain final partial batch |
| Optimizer | AdamW, LR 0.001, weight decay 0, betas 0.9/0.999 |
| Gradient clipping | Norm 5 |
| Precision / schedule | FP32, no scheduler or early stopping |
| Validation | Complete utterances, batch 1, every epoch |
| Selection | Lowest macro validation loss; earliest epoch on a tie |

Loss is proposed as negative zero-mean SI-SNR with epsilon `1e-12`, averaged per
utterance over valid samples. Silent clean references produce an error. This is
implemented in `EnhancementLoss`; evaluation math remains unchanged.
Future crop/shuffle randomness must be controlled by the training seed and saved
with resumable optimizer/RNG state. Training and validation stay separate entry
points; the orchestration layer invokes validation and applies the selection rule.

GTCRN and LiSenNet use their current constructor settings. TF-GridNet uses the
original **six-layer local profile**, trained from scratch. The four-layer DNS
pretrained profile stays in the external-pretrained results and is rejected by
this planner. The current frontends remain part of each model; this protocol
does not make them causal or identical.

Device is deliberately unselected. The proposed batch size, memory requirements,
wall time and training stability need a short feasibility check on the target
hardware before a final protocol is frozen. Equal epochs and data exposure are
not equal compute budgets. These defaults are not a request to start 100 epochs.

## Commands

The planner uses PyYAML from the base project requirements to read configuration.
It does not import PyTorch or any audio backend.

Describe the draft without data, weights, model execution or downloads:

```sh
python -m scripts.plan_training --model gtcrn --describe
python -m scripts.plan_training --model lisennet --describe
python -m scripts.plan_training --model tfgridnet --describe
python -m scripts.smoke_training_plan
```

The smoke test builds synthetic metadata and checks the split, rejection rules,
profile distinction, deterministic fingerprints and output protection. It does
not calculate audio quality or prove that a training loop works.

When a real train manifest has already been prepared, write a plan:

```sh
python -m scripts.plan_training --model gtcrn --manifest data/voicebank-demand-16k/train_manifest.json --output results/gtcrn_training_plan.json
```

Do not download train data solely to run the synthetic check. Planning never
fetches missing data. Only new `.json` files below `results/` may be created;
existing files and historical reports cannot be overwritten. All model plans
should have the same `split_sha256` and `protocol_sha256`. Model constructor
settings and their source-config fingerprint are recorded separately.

## Remaining implementation

Epoch state, independent validation and the selection rule are implemented as
[reusable components](training_state.md). The
[entry-point fixture](training_runner.md) connects seeded synthetic updates,
independent validation and saved selection history. Binding the real training
data/plan and enabling production entry-point execution remain to be implemented.
Verify that runner on synthetic fixtures before a small hardware feasibility run.
Only after that should the training protocol be frozen and full training begin.
