# Edge Speech Denoising Benchmark

One private repository for GTCRN, LiSenNet and TF-GridNet under a shared
waveform-to-waveform interface. No separate model forks or repositories.

## Current status

GTCRN network integration, official VCTK-DEMAND pretrained loading, single-WAV
inference, pinned data preparation and shared evaluation are implemented.
LiSenNet/TF-GridNet remain placeholders. Training, validation and losses remain
Phase 3 placeholders; train.py and validate.py are independent entry points.
No training has been performed. Pretrained scores are not controlled-training
comparisons. Keep the repository private in GitHub settings.

## Dataset

**VoiceBank-DEMAND-16k**, a standard speech enhancement benchmark.
Train split: **11572**. Test split: **824**, fixed following dataset protocol.

HF source: https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k
Revision: `20879f4f9aab3d0b9263993667e7711a3ae1416d`.
The source card labels it CC-BY-4.0. `configs/voicebank.json` pins file hashes.
Data remains outside Git. Preparation checks all audio pairs and writes sorted
manifests with audio hashes. Validation must later be split from train; test
cannot be used for tuning or checkpoint selection.

## Baseline models

- **GTCRN**: lightweight real-time speech enhancement model. The current
  adapter uses whole-utterance STFT; this is not a stateful streaming claim.
- **LiSenNet**: lightweight sub-band and dual-path speech enhancement model.
- **TF-GridNet**: time-frequency modeling based speech enhancement model.

All adapters accept and return aligned mono `[batch, samples]` at 16 kHz.
Native feature processing belongs inside each adapter. `build_model("gtcrn")`
constructs a random-weight model. Pretrained weights must be explicitly loaded
with `load_gtcrn_checkpoint`; single-file and evaluation scripts do this.
GTCRN attribution and MIT license are retained beside its network source.

## Setup and evaluation

Python 3.10+ and PyTorch are required. For the evaluation extras:

```bash
python -m pip install -r requirements-benchmark.txt
python -m scripts.smoke_benchmark
python -m scripts.prepare_data --split test
python -m scripts.evaluate --metrics stoi si_snr si_snr_improvement --limit 5 --output results/gtcrn_test5.json
```

Preparation downloads only the 132 MB test shard by default. Use `--split all`
only when both train and test are needed (about 2.28 GB of parquet downloads).
`--parquet PATH` accepts a manually downloaded pinned test shard.

The five-utterance run is an integration check, not the full benchmark. For all
824 test utterances with the selected metrics, remove `--limit 5`:

```bash
python -m scripts.evaluate --metrics stoi si_snr si_snr_improvement
```

PESQ uses the standard `pesq` backend; it may require a C compiler to install.
After installing it, `python -m scripts.evaluate` computes all quality metrics.
Missing backends abort the requested run; explicitly omitted metrics are
recorded as not computed, without substitute or fabricated scores.

## Evaluation

One `evaluate(model, dataset)` implementation supplies all model adapters:

- PESQ (16 kHz wideband)
- STOI (classical)
- SI-SNR (dB)
- SI-SNR improvement (enhanced minus noisy SI-SNR, dB)
- RTF (inference seconds / audio duration)
- Parameter count (all, with trainable count reported separately)

Reports include noisy baseline, per-utterance scores, metric coverage, source
revision, manifest fingerprint, checkpoint provenance and timing conditions.
Default efficiency protocol: CPU FP32, batch=1, one thread, one warmup and
three timed runs per utterance. RTF includes STFT/network/iSTFT, excludes file
I/O, transfers and metric computation. Full protocol: `docs/benchmark_protocol.md`.

## Future plan

1. **Phase 1: Framework setup** — complete.
2. **Phase 2: Integrate pretrained checkpoints** — GTCRN complete; integrate
   LiSenNet and TF-GridNet next, with license/revision and output checks.
3. **Phase 3: Unified training pipeline** — shared data splits, loss, budget,
   reproducible seeds and selection rule; validation stays separate from train.
4. **Phase 4: Controlled benchmark comparison** — same fixed test manifest,
   metric code and hardware under matched training/measurement conditions.

Extract only necessary definitions/training/inference/evaluation logic; do not
copy complete upstream repositories. Record source revisions, required notices,
checkpoint training provenance and architectural capabilities for each adapter.

Architecture reference: https://github.com/kittytinglee/edge-kws-benchmark
