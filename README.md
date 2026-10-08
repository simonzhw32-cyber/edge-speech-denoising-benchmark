# Edge Speech Denoising Benchmark

> **Current milestone:** GTCRN random seed-42 completed 25 epochs; epoch 24 was selected and evaluated on all 824 test items. See [experiment results](docs/gtcrn_seed42_pilot.md). GTCRN and LiSenNet have separate real training/validation entry points; LiSenNet real-data results are pending.


GTCRN, LiSenNet and original TF-GridNet in one codebase, with a shared
16 kHz waveform interface and VoiceBank-DEMAND evaluation.

## Models

| Model | Implementation | Pretrained evaluation |
|---|---|---|
| GTCRN | Lightweight real-time architecture; whole-utterance adapter | VCTK-DEMAND checkpoint, full 824-item test complete |
| LiSenNet | Lightweight sub-band and dual-path architecture; offline phase reconstruction | Pending accepted checkpoint provenance |
| TF-GridNet | Original offline ESPnet architecture | Four-layer DNS checkpoint, full 824-item test complete |

All adapters map mono `[batch, samples]` tensors to the same shape.
`build_model()` creates random weights. TF-GridNet's registry default is the
six-layer local profile (8,381,504 parameters); the DNS loader uses the separate
four-layer `dns_ins20_epoch33` profile (2,552,790 parameters).

Training and validation have separate entry points. In addition to the synthetic
GTCRN fixture, an explicit `--real` GTCRN/LiSenNet path binds a verified VoiceBank train
manifest and preflight report, updates parameters, saves epoch state, and requires
independent complete held-out validation before the next epoch. It supports either
random initialization or the checksum-pinned upstream GTCRN checkpoint and records
that choice in the run identity. GTCRN seed 42 completed a 25-epoch pilot with 25 independent validations;
epoch 24 was evaluated on the complete test split. The protocol and final
three-model comparison budget remain a draft. LiSenNet supports random
initialization; accepted upstream fine-tuning remains GTCRN-only.
[Training-data preflight](docs/training_preflight.md) verifies existing local
train audio and binds the draft speaker split; it does not enable training.

## Recorded pretrained results

Both reports cover the fixed 824-item test split, with all four quality metrics,
zero failures, matching utterance IDs and the same manifest fingerprint.
Values below come from the committed JSON summaries; SI-SNR columns are in dB.

| Model / checkpoint training data | PESQ-WB | STOI | SI-SNR | SI-SNR improvement | Parameters: all / trainable |
|---|---:|---:|---:|---:|---:|
| GTCRN (VCTK-DEMAND) | 2.86969 | 0.94029 | 18.79606 | 10.35052 | 48,245 / 23,669 |
| TF-GridNet (DNS) | 2.77327 | 0.94310 | 19.40151 | 10.95598 | 2,552,790 / 2,552,790 |

Sources: [GTCRN report](benchmark_reports/gtcrn_pretrained_voicebank_test824_windows.json)
and [TF-GridNet DNS report](benchmark_reports/tfgridnet_dns_pretrained_voicebank_test824_windows.json).
LiSenNet has no accepted pretrained score; its checkpoint decision is documented
in the [audit](docs/lisennet_checkpoint_audit.md).

These checkpoints have different training provenance and budgets. The results
are descriptive pretrained baselines, not a controlled training comparison.
Exact DNS training manifests and selection logs are unavailable. RTF is listed
with its run conditions in [results](docs/results.md); the runs were made on
different computers, so their runtimes are not ranked here.

## Locally trained GTCRN pilot

| Initialization / selected epoch | PESQ-WB | STOI | SI-SNR (dB) | Improvement (dB) |
|---|---:|---:|---:|---:|
| Random seed 42 / epoch 24 of 25 | 2.57246 | 0.93428 | 18.70302 | 10.25748 |

This pilot trained on 10,778 utterances and validated on 794 held-out train
utterances. It is separate from the official pretrained results above.

Weights are **Release assets**, not part of `git clone`. Download
[`gtcrn_seed42_epoch25_20261005_204146.tar.gz`](https://github.com/simonzhw32-cyber/edge-speech-denoising-benchmark/releases/download/gtcrn-seed42-pilot-25ep/gtcrn_seed42_epoch25_20261005_204146.tar.gz)
and [`archive.sha256`](https://github.com/simonzhw32-cyber/edge-speech-denoising-benchmark/releases/download/gtcrn-seed42-pilot-25ep/archive.sha256)
from the [pilot Release](https://github.com/simonzhw32-cyber/edge-speech-denoising-benchmark/releases/tag/gtcrn-seed42-pilot-25ep).
Verify the archive hash before extracting it. The selected checkpoint is
`results/training_runs/gtcrn_random_seed42/checkpoints/epoch_0024.pt`;
epoch 25 retains the final optimizer/RNG state.

For this historical experiment, use the Release tag checkout and its archived
run. Its strict source fingerprints intentionally reject later training-code
changes. See [restore and evaluate](docs/gtcrn_seed42_pilot.md).
For new LiSenNet work, see [the real runner commands](docs/real_training.md).

## Dataset

[VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k)
is a standard speech enhancement benchmark: **11,572 train** and **824 test**
utterances. The test split is fixed by the dataset protocol. Validation
comes from train, with no test-based tuning or checkpoint selection.

[configs/voicebank.json](configs/voicebank.json) pins the source revision,
parquet sizes and hashes. Preparation validates audio pairs and preserves their
encoded bytes. Data and generated output stay outside Git.

## Install

Use Python 3.10+ and install PyTorch for your platform first. The recorded Windows
runs used PyTorch 2.14.1. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m pip install -r requirements-benchmark.txt -r requirements-tfgridnet.txt
python -m pip install Cython wheel
python -m pip install pesq==0.0.4 --no-build-isolation
```

PESQ needs a C compiler when built from source. For Windows tool installation,
Git authentication and recovery on a new computer, see [Windows setup](docs/windows_restore.md).
The `torch-complex` extra supports the historical TF-GridNet reference check;
production inference uses the local adapter.

## Evaluate

Fetch the checkpoint for the model you want to evaluate, then prepare test data:

```sh
python -m scripts.fetch_gtcrn_checkpoint
python -m scripts.fetch_tfgridnet_checkpoint
python -m scripts.prepare_data --split test
```

Each fetcher verifies pinned weights. Data preparation downloads only the
132 MB test parquet by default. `--parquet PATH` accepts the same pinned file
from local storage; `--split all` additionally prepares train.

Check five utterances before a full run:

```sh
python -m scripts.evaluate --model gtcrn --metrics pesq stoi si_snr si_snr_improvement --limit 5 --output results/gtcrn_test5_all_metrics.json
python -m scripts.evaluate --model tfgridnet --profile dns_ins20_epoch33 --metrics pesq stoi si_snr si_snr_improvement --limit 5 --output results/tfgridnet_dns_test5_all_metrics.json
```

A five-item report is a subset check. For all 824 utterances, remove `--limit 5`
and choose a separate output filename. Missing requested backends abort; to omit
PESQ explicitly, request `--metrics stoi si_snr si_snr_improvement`. The report
then records partial metric coverage.

One `evaluate(model, dataset)` function supplies PESQ-WB, classical STOI,
SI-SNR, SI-SNR improvement, RTF and parameter counts. The default timing protocol
is CPU FP32, batch=1, one thread, one warmup and three timed runs per utterance.
RTF includes adapter processing and the network, with file I/O, transfers and
metrics excluded. See the [benchmark protocol](docs/benchmark_protocol.md).

For single-WAV inference, see the [GTCRN instructions](speech_denoising/models/gtcrn/PRETRAINED.md)
and [TF-GridNet instructions](speech_denoising/models/tfgridnet/PRETRAINED.md).

## Development

No-download fixture checks:

```sh
python -m scripts.smoke_benchmark
python -m scripts.smoke_evaluation_cli
python -m scripts.smoke_gtcrn
python -m scripts.smoke_lisennet
python -m scripts.smoke_tfgridnet
```

These include metric/CLI contracts, random-weight adapters and embedded upstream
reference checks. The pretrained checks additionally require the corresponding
local weights: `scripts.smoke_gtcrn_pretrained` and `scripts.smoke_tfgridnet_pretrained`.
Synthetic checks are separate from dataset quality evaluation.

Next work is one LiSenNet real epoch plus complete independent validation,
followed by a recorded pilot budget. Six-layer TF-GridNet real training remains
pending. The 25-epoch GTCRN pilot does not establish a completed three-model comparison. Source revisions and original license
notices live beside each model; [history](docs/history.md) records earlier milestones.

The [training draft](docs/training_protocol.md) adds a speaker-disjoint split
planner and synthetic checks. It does not execute training or verify local audio.
The [waveform components](docs/training_components.md) provide loss and batch
checks through `python -m scripts.smoke_training_components`.
[State and validation components](docs/training_state.md) add an epoch-boundary
resume fixture and selection checks through `python -m scripts.smoke_training_state`.
The [entry-point fixture](docs/training_runner.md) checks train/validate persistence
and resumed synthetic updates with `python -m scripts.smoke_training_runner`.
The [data preflight](docs/training_preflight.md) has a synthetic WAV/subset check
through `python -m scripts.smoke_training_preflight`; it performs no model updates.
The [bounded resource probe](docs/training_probe.md) checks one configured train
batch and one full validation utterance on an explicit device, without optimizer
steps. A passing probe does not yet enable real-data training.
The [real GTCRN runner](docs/real_training.md) documents the separate train and
validation commands, initialization provenance, saved artifacts and safety gates.

Framework reference: [edge-kws-benchmark](https://github.com/kittytinglee/edge-kws-benchmark).

See [requirements and acceptance status](docs/project_requirements.md) for the
remaining work, dependency roles and the next real-training milestone.


## LiSenNet pilot completed — 2026-10-08

25 real training epochs and independent validations completed; epoch 21 selected and evaluated on the fixed 824-utterance test set. See [results and GTCRN comparison](docs/lisennet_seed42_pilot.md). Weights are provided in the `lisennet-seed42-pilot-25ep` Release.
