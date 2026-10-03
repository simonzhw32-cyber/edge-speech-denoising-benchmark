# Requirements and acceptance status

Review baseline: `b8aee8eed15f8f326d26576aa2f554ac3b169fe5`, 2026-10-04
(Asia/Shanghai). This review reads the actual repository, including committed
benchmark and training-resource reports. No `AGENTS.md` or teacher-authored
acceptance checklist is present in this snapshot. The requirements below are
the user's project instructions and relayed group workflow. Draft hyperparameters
and engineering tests are not additional requirements attributed to the group.

## Coverage and acceptance criteria

| ID | Requirement | Status | Evidence / completion criterion |
|---|---|---|---|
| R1 | One benchmark repository with extracted model code and source attribution | Implemented | Shared package, three model directories, `SOURCE.md` and upstream licenses; no three independent model forks |
| R2 | GTCRN, LiSenNet and TF-GridNet share a mono 16 kHz waveform interface | Implemented | `BaseEnhancementModel`, registry adapters, reference-equivalence/alignment/gradient smoke checks |
| R3 | Pinned public dataset, fixed test and reproducible validation from train | Implemented in preparation/planning | `configs/voicebank.json`, manifests, speaker-disjoint plan and complete train-audio preflight |
| R4 | Common PESQ-WB, STOI, SI-SNR, improvement, RTF and parameter counts | Complete for two pretrained baselines | Both full 824-item JSON reports include four quality metrics, full coverage and zero failures |
| R5 | Explain three-model baseline availability and training provenance | Partial | LiSenNet checkpoint provenance remains unresolved; do not assign a random-weight score as its pretrained baseline |
| R6 | Separate training and validation entry points with resumable state | Synthetic execution implemented; real-data integration pending | `scripts/train.py`, `scripts/validate.py`, state/selection components; real epoch and held-out validation must still be connected |
| R7 | Controlled comparison under a recorded shared training protocol | Pending | Random initialization, shared split/loss/exposure and seeds; selected checkpoints and full test reports after real runs |
| R8 | On-board profiling / edge deployment evidence | Pending | Record agreed target hardware, runtime, model/profile, precision, workload, latency and memory; desktop RTF is not on-board profiling |
| R9 | Repository visibility follows group agreement | Needs confirmation | Previously requested private; this review can access a public repository after the user opened it. If private remains required, change it back after review |

R7's proposed settings live in [training_protocol.md](training_protocol.md) and
`configs/training_protocol.yaml`: seeds 42/43/44, four-second crops, batch four,
100 epochs, FP32, AdamW and negative SI-SNR. They remain a draft, not an accepted
group budget or a reproduction of the papers' recipes. Equal epochs are not
equal compute cost. Any later protocol change must create a separately identified
experiment and explain what can still be compared.

## Evidence boundaries

The two full reports use the same fixed test source and ordered 824 IDs; their
per-utterance macro means agree with the summaries. They are **external pretrained
baselines**, with different training datasets, budgets and selection provenance.
They do not satisfy R7. Existing RTF runs were made on different computers and
must not be ranked as a matched runtime comparison. See [results](results.md).

| Pretrained baseline | PESQ-WB | STOI | SI-SNR (dB) | Improvement (dB) |
|---|---:|---:|---:|---:|
| GTCRN / VCTK-DEMAND | 2.86969 | 0.94029 | 18.79606 | 10.35052 |
| TF-GridNet / DNS | 2.77327 | 0.94310 | 19.40151 | 10.95598 |

The original six-layer local TF-GridNet has 8,381,504 parameters. The separate
four-layer DNS pretrained profile has 2,552,790 parameters. Do not substitute DNS
for the local model silently to make the unified experiment fit in memory.

The archived train preflight verified 11,572 pairs and holds out speakers p256
and p259: **10,778 train / 794 validation**. The fixed 824-item test is excluded
from training, validation, tuning and checkpoint selection.

GTCRN and LiSenNet passed a bounded real-audio batch of four 64,000-sample crops
with backward and one full validation utterance. Reports record zero optimizer
steps. This is resource evidence, not a completed training epoch or full validation.

TF-GridNet failed under a 50% CUDA allocator cap (3.98 GiB), both at draft batch
four and diagnostic batch one. The archived trace locates the latter failure at
`network.blocks.1.inter_rnn.forward`, during train forward, input
`[257, 248, 192]`. Loss, backward and validation were not reached. These failures
do not establish whether unrestricted 8 GiB training can fit. Preserve the cap,
diagnostic batch and trace conditions when interpreting those reports.

## Dependency roles

| File / dependency | Purpose |
|---|---|
| `requirements.txt` | Editable base package; `pyproject.toml` supplies Torch, NumPy, SoundFile and PyYAML |
| `requirements-benchmark.txt` | Parquet data preparation and STOI evaluation; SoundFile is also declared here |
| `requirements-tfgridnet.txt` | Historical TF-GridNet reference support (`torch-complex`), NumPy minimum and packaging |
| `pesq==0.0.4` | Explicit optional compiled backend; required for full four-metric evaluation |
| Cython, wheel, C++ build tools | PESQ build prerequisites when no suitable wheel is available |

There is no missing PyYAML dependency: it is declared in `pyproject.toml`.
CPU/CUDA selection belongs to environment setup, not a universal CUDA pin in
base requirements. Install the chosen Torch build first, then base/extras with
the **same interpreter**. `pip freeze` snapshots are evidence of prior runs,
not cross-platform installation recipes. See [Windows recovery](windows_restore.md)
for the recorded GPU environment and new-machine setup.

## Continue from this baseline

1. Restore the checkout and the intended interpreter. Restore local train data
   only if absent; old absolute paths in archived reports are not portable.
2. Connect `TrainingSubset`, batching, loss, optimizer, run persistence and state
   restoration to real-data training; bind the split, source and runtime to the
   run identity. Keep validation in its independent entry point.
3. Run **one real GTCRN epoch**, then all 794 held-out validation utterances.
   Save epoch state, optimizer/RNG state, loss history and checkpoint selection.
   Check epoch-boundary resume before starting the longer run. This first epoch
   is an integration/stability check, not a final baseline result.
4. Continue GTCRN and integrate LiSenNet using the recorded common protocol.
   Resolve TF-GridNet resource feasibility separately; it does not block starting
   GTCRN. Record any model-specific deviations before comparing results.
5. After training, choose checkpoints using validation alone and evaluate each
   selected checkpoint on the fixed test with complete quality coverage. Store
   trained-model reports separately from existing pretrained reports.
6. Perform matched-hardware runtime comparison and agreed on-board profiling.

A real-training acceptance record must identify model/profile, initialization,
training seed, source commit, dependency versions, hardware, manifest/split/protocol
fingerprints, optimizer settings, precision/backend flags, completed epochs and
steps, selection evidence and checkpoint hash. Preserve partial/failed runs as
such; do not merge them into a completed experiment.

The current `train.py` and `validate.py` still expose only synthetic execution.
No real training command, trained checkpoint, convergence curve or final
controlled comparison exists at this baseline.

## Materials to retain

Keep source licenses/attribution, checkpoint cards and hashes, audit records,
reference-equivalence tests, full and partial benchmark JSON reports and their
environment snapshots, and the diagnostic JSON/launchers under
`training_reports/phase36_windows/`. The long preflight file records actual IDs
and verification evidence; its size alone is not a reason to discard it.

Git preserves those tracked materials. It does not preserve ignored audio,
environments, weights, generated results or future training checkpoints. Once
training starts, back up learned weights separately from code and archive hashes
and run metadata with the result. Copies of environments must be recreated on
new machines; archived probe launchers remain historical diagnostics.
