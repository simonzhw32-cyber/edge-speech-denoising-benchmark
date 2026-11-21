# Edge Speech Denoising Benchmark

> **Current milestone (2026-11-12):** the LiSenNet paper-recipe run completed
> 110 epochs (full-validation PESQ 2.70486, 824-item test PESQ 3.04565); the
> 100-epoch artifacts are preserved beside it. The GTCRN paper-track run
> completed 100 epochs; its fixed-test 824 result is PESQ 2.79621 / STOI 0.94055
> / SI-SNR 18.74095 (epoch 87 peaked at PESQ 2.85092). TF-GridNet's 8 ms-hop
> probe OOMed on the 16 GB GPU, so its paper-hop path remains blocked.
> See [paper alignment](docs/paper_alignment.md) and the
> [GTCRN](docs/gtcrn_seed42_pilot.md), [LiSenNet](docs/lisennet_seed42_pilot.md)
> and [TF-GridNet](docs/tfgridnet_seed42_25ep.md) pilot reports.


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

## Three-model pilot comparison

The three 25-epoch pilots below use the same speaker-disjoint split
(10,778 train / 794 validation / 824 test), 4 s crops, batch 4 and the unified
negative-SI-SNR objective. They are comparable to each other, not to the
pretrained baselines or to LiSenNet's paper recipe.

| Model / initialization | Selected epoch | PESQ-WB | STOI | SI-SNR (dB) | Improvement (dB) |
|---|---|---:|---:|---:|---:|
| GTCRN / random seed 42 | 24 of 25 | 2.57246 | 0.93428 | 18.70302 | 10.25748 |
| LiSenNet / random seed 42 | 21 of 25 | 2.57229 | 0.93443 | 18.63120 | 10.18566 |
| TF-GridNet 6-layer / random seed 42 | 23 of 25 | 2.79501 | 0.94825 | 18.32292 | 9.87739 |

TF-GridNet used gradient accumulation 4 with micro-batch 1 (effective batch 4);
its RTF is therefore not comparable to the other rows. Sources:
[GTCRN](docs/gtcrn_seed42_pilot.md), [LiSenNet](docs/lisennet_seed42_pilot.md),
[TF-GridNet](docs/tfgridnet_seed42_25ep.md).

## Paper recipes

The unified benchmark deliberately does not reproduce each paper's optimizer,
schedule or loss. The separate [paper-alignment record](docs/paper_alignment.md)
tabulates paper values against the repository for all three models and documents
what is executable:

- **LiSenNet** — 100 epochs is defined by the paper. The paper-derived run
  (2 s crops, AdamW 5e-4, betas (0.8, 0.99), StepLR 0.98, PESQ-conditioned GAN
  loss) completed 110 epochs under `results/paper_runs/lisennet_paper_seed42_100ep`;
  full-validation PESQ 2.70486 at epoch 107 and 824-test PESQ 3.04565. Deviations:
  824-item speaker-disjoint test rather than 872, a 200-utterance PESQ-selection
  subset, wrap-padded shorts, and non-scale-invariant validation PESQ; see
  `docs/paper_alignment.md`.
- **GTCRN** — the paper states STFT, batch 4, Adam 1e-3 with plateau halving and
  a `0.01/0.7/0.3` SI-SNR/magnitude/RI loss, but **no VCTK epoch count**. A
  100-epoch budget-parity run completed under
  `results/paper_runs/gtcrn_paper_seed42_100ep`: epoch 100 test PESQ 2.79621,
  epoch 87 peaked at 2.85092. `paper_full` has no held-out split, so the last
  epoch is the protocol result; see `docs/paper_alignment.md`.
- **TF-GridNet** — the paper has no VoiceBank-DEMAND recipe; its 8 ms hop,
  sqrt-Hann window and Wav+Mag+MC loss are structural differences from the local
  16 ms-hop, plain-Hann, negative-SI-SNR profile. The 8 ms, six-layer model
  cannot train in FP32 on the 16 GB card. The [memory-policy study](docs/tfgridnet_memory_policy_bench.md)
  measures a 20-step Pareto and proposes cheap-first hierarchical checkpointing;
  a 200-step paired run and a 100-step attention-only run are loss-bit-identical,
  so the saving changes memory but not the optimization trajectory; the
  1,000-step uniform and attention arms both reached PESQ 2.123 with identical
  losses. A 200-step 16 ms precision pair shows FP16 saves 4.27 GiB with final
  PESQ within 0.003. The 4-layer DNS check fits at 8 ms and shows the same ordering at a 9 % step-time
  cost. The
  [experiment matrix](docs/tfgridnet_experiment_matrix.md) lists E1–E7 status and artifacts. The
  [paper draft](docs/tfgridnet_paper_draft.md) records the current claims and
  limitations (single GPU, short proxy, 16 ms fallback), the
  [submission checklist](docs/tfgridnet_submission_checklist.md) tracks the
  claim/evidence gaps, the [reviewer FAQ](docs/tfgridnet_reviewer_faq.md)
  records expected objections, and the
  [reproduction note](docs/tfgridnet_reproduce.md) lists exact commands.

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
runs used Python 3.14.9 and `torch 2.11.0+cu128`. From the repository root:

```sh
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -r requirements-benchmark.txt -r requirements-tfgridnet.txt
python -m pip install Cython wheel
python -m pip install pesq==0.0.4 --no-build-isolation
```

PESQ needs a C compiler when built from source. For Windows tool installation,
Git authentication and recovery on a new computer, see [Windows setup](docs/windows_restore.md).
The `torch-complex` extra supports the historical TF-GridNet reference check;
production inference uses the local adapter.

## Deploy GTCRN

The GTCRN paper-track checkpoint exports to a 408 KB spectrum-domain ONNX graph.
On the host i5-12600KF it runs at RTF **0.0073** with four ONNX Runtime threads
and **0.0100** with one thread (9.77 s utterance), with PyTorch/ONNX parity
**3.6e-6**. See [GTCRN ONNX deployment](docs/gtcrn_deployment.md) for export,
benchmark commands, and the phone/board next steps.

```sh
python -m scripts.export_gtcrn_onnx
python -m scripts.deploy_gtcrn_onnx --threads 1 \
  --audio data/voicebank-demand-16k/audio/test/noisy/p232_023.wav \
  --output results/deploy/gtcrn_p232_023_enhanced.wav
```

## Deploy TF-GridNet

The pretrained DNS TF-GridNet (`dns_ins20_epoch33`, 2.55M parameters) runs
offline on CPU and CUDA. On the host i5-12600KF it takes **1.20 s** for a 4 s
segment with four threads (RTF 0.299) and **83.8 ms** on the RTX 5060 Ti
(RTF 0.021); 4 s PESQ improves from 1.3359 to 2.8297. See
[TF-GridNet deployment](docs/tfgridnet_deployment.md) for the full table and
caveats.

```sh
python -m scripts.fetch_tfgridnet_checkpoint
python -m scripts.bench_tfgridnet_inference --device cpu --threads 4 --seconds 4
python -m scripts.bench_tfgridnet_inference --device cuda --seconds 4
```

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
python -m scripts.smoke_lisennet_paper
python -m scripts.smoke_gtcrn_paper
python -m scripts.smoke_tfgridnet
```

These include metric/CLI contracts, random-weight adapters and embedded upstream
reference checks. The pretrained checks additionally require the corresponding
local weights: `scripts.smoke_gtcrn_pretrained` and `scripts.smoke_tfgridnet_pretrained`.
Synthetic checks are separate from dataset quality evaluation.

The one-seed three-model 25-epoch pilot comparison is complete; the multi-seed
controlled comparison (R7) remains open. Current work is the
LiSenNet paper-recipe 100-epoch run and a post-run full validation plus 824-item
test; the unified protocol remains the only controlled cross-model comparison.
GTCRN has no paper-defined VCTK epoch count, and TF-GridNet's paper does not
train on VoiceBank-DEMAND, so neither gets a literal “paper 100 epochs” run. See
[paper alignment](docs/paper_alignment.md) for the full matrix. Source revisions
and original license notices live beside each model;
[history](docs/history.md) records earlier milestones.

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

## LiSenNet paper-protocol run

The LiSenNet paper recipe was trained to 110 epochs on the repository's
speaker-disjoint 10,778/794 split. Full validation selected epoch 107; the fixed
824-item test split was then evaluated once.

| Split / selected epoch | PESQ-WB | STOI | SI-SNR (dB) | Improvement (dB) |
|---|---:|---:|---:|---:|
| 794 held-out / epoch 107 | 2.70486 | 0.88075 | 9.62549 | — |
| 824 test / epoch 107 | 3.04565 | 0.93616 | 12.96042 | 4.51489 |

Recipe: 2 s crop, batch 4, AdamW 5e-4, betas (0.8, 0.99), StepLR 0.98/epoch,
clip 5.0, PESQ-conditioned GAN loss. The paper reports PESQ 3.07 on its own
872-utterance split; this run uses the repository's 824-utterance split, so the
result is aligned but not an identical reproduction. Details:
[paper alignment](docs/paper_alignment.md).
