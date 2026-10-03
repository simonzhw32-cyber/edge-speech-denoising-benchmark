# Benchmark protocol

## Dataset and split

Source: [JacobLinCool/VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k),
revision `20879f4f9aab3d0b9263993667e7711a3ae1416d`.
The source card declares CC-BY-4.0. There are 11,572 train and 824 fixed test
utterances at 16 kHz. [configs/voicebank.json](../configs/voicebank.json) pins
all six parquet file sizes and SHA-256 hashes. The test shard is 132,200,447
bytes; training shards total about 2.15 GB.

Preparation checks parquet hashes, unique IDs, filenames, finite mono 16 kHz
signals and equal clean/noisy lengths. Original audio bytes are preserved.
Sorted manifest rows and individual files have hashes; the loader verifies them
and rejects paths outside the manifest root. Full manifests must match the
pinned source and split count. Future validation comes from train; test is never
used for tuning or checkpoint selection.

## Quality metrics

Aligned complete utterances are scored without independent gain normalization,
clipping, resampling, silence trimming or score-dependent truncation.

| Metric | Definition |
|---|---|
| PESQ | `pesq`, 16 kHz wideband MOS-LQO |
| STOI | Classical `pystoi`, `extended=False` |
| SI-SNR | Zero-mean projection onto the clean reference, in dB |
| SI-SNR improvement | Enhanced SI-SNR minus noisy SI-SNR |

SI-SNR uses `10*log10((projected_energy + 1e-12)/(residual_energy + 1e-12))`.
A silent clean reference is undefined and produces an explicit error. Quality
means are per-utterance macro means. The noisy baseline uses the same functions.
Errors retain null scores and named failures, with coverage reported separately.
Data/integrity errors abort rather than dropping utterances.

Missing requested backends stop evaluation. Explicitly omitted metrics appear
in `not_computed`; nothing is substituted for PESQ. A report that processes all
824 utterances but omits PESQ still has partial quality coverage.

## Runtime and parameters

Default timing is CPU FP32, batch=1, one thread, one warmup and three timed
runs per utterance. RTF is the sum of per-utterance median inference seconds
divided by total audio duration. Adapter STFT/network/iSTFT are included;
file decoding, transfers and metric calculation are excluded. CUDA timing
synchronizes when a CUDA model is supplied to the programmatic evaluator.

Reports record platform, device, precision, threads, repeats and package versions.
Compare runtime only under matched hardware, software and measurement conditions.
Whole-utterance centered-STFT timing does not measure streaming latency.
Parameter count includes all parameters, including GTCRN's frozen ERB parameters;
trainable parameters are counted separately.

## Profiles and report scope

`scripts.evaluate` strictly loads GTCRN VCTK-DEMAND or the explicit TF-GridNet
`dns_ins20_epoch33` profile, then calls the same `evaluate(model, dataset)`.
TF-GridNet's six-layer random default is not selected by this CLI branch.
LiSenNet is held back pending accepted checkpoint provenance.

Five-item checks and synthetic fixtures are labeled `subset_or_fixture`.
Only the complete verified test split is `full_fixed_test`. Full quality coverage
also requires all four quality metrics, zero failures and 824 valid scores per
metric. Reports retain per-utterance records, dataset/checkpoint provenance and
measurement conditions; newer CLI reports also record source fingerprints and
Git context.

The recorded scores are external pretrained baselines with different training
provenance. Exact DNS training manifests are unavailable, so these results do
not establish a controlled training comparison.

Commands: [README](../README.md), [Windows setup](windows_restore.md) and
[DNS evaluation](tfgridnet_dns_evaluation.md). Existing records are indexed in
[results](results.md).
