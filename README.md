# Edge Speech Denoising Benchmark

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

Training and validation remain separate configuration-only entry points.
A shared valid-length waveform loss and paired crop/batch utilities are available;
the optimizer loop and validation orchestration are not implemented.

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

## Dataset

[VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k)
is a standard speech enhancement benchmark: **11,572 train** and **824 test**
utterances. The test split is fixed by the dataset protocol. Future validation
must come from train, with no test-based tuning or checkpoint selection.

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

Next work is a unified training pipeline with a reproducible train/validation
split, loss, budget, seeds and checkpoint-selection rule, followed by a controlled
comparison. Keep the repository private. Source revisions and original license
notices live beside each model; [history](docs/history.md) records earlier milestones.

The [training draft](docs/training_protocol.md) adds a speaker-disjoint split
planner and synthetic checks. It does not execute training or verify local audio.
The [waveform components](docs/training_components.md) provide loss and batch
checks through `python -m scripts.smoke_training_components`.

Framework reference: [edge-kws-benchmark](https://github.com/kittytinglee/edge-kws-benchmark).
