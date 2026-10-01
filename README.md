# Edge Speech Denoising Benchmark

One private repository for GTCRN, LiSenNet and TF-GridNet under a shared
waveform-to-waveform interface. No separate model forks or repositories.

## Current status

GTCRN network integration, official VCTK-DEMAND pretrained loading, single-WAV
inference, pinned data preparation and shared evaluation are implemented.
LiSenNet network and waveform adapter are integrated and verified against the
pinned upstream generator, with random weights only. Original TF-GridNet
network and offline mono waveform adapter are also integrated with random weights. Training, validation and losses remain Phase 3 placeholders;
train.py and validate.py are independent entry points.
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
  The native two-iteration Griffin-Lim path is retained for offline inference.
  Network/adapter checks pass; no pretrained weights or quality scores yet.
- **TF-GridNet**: time-frequency modeling based speech enhancement model.
  Original ESPnet variant, with bidirectional LSTMs and global attention; offline.
  Local 16 kHz mono single-output profile; no pretrained weights or quality scores.

All adapters accept and return aligned mono `[batch, samples]` at 16 kHz.
Native feature processing belongs inside each adapter. `build_model("gtcrn")`
constructs a random-weight model. Pretrained weights must be explicitly loaded
with `load_gtcrn_checkpoint`; single-file and evaluation scripts do this.
GTCRN and LiSenNet attribution and original MIT notices are retained beside
their network sources. LiSenNet normalization and phase reconstruction are
documented in `speech_denoising/models/lisennet/SOURCE.md`. TF-GridNet keeps the
ESPnet Apache-2.0 license and pinned provenance in its own `SOURCE.md`.

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

## Verified integration status

GTCRN pretrained evaluation on the full fixed 824-utterance test split is saved
in `benchmark_reports/gtcrn_pretrained_voicebank_test824_windows.json`.
That Windows run reports PESQ-WB=2.86969, STOI=0.94029, SI-SNR=18.79606 dB,
SI-SNR improvement=10.35052 dB, with zero metric failures. RTF=0.01557 applies
only to that recorded machine/offline protocol, not to a target edge device.
The dependency snapshot is `benchmark_reports/gtcrn_windows_environment.txt`.
This is a pretrained baseline, not a controlled-training model comparison.

For LiSenNet architecture/adapter verification:

```bash
python -m scripts.smoke_lisennet
```

This uses random weights, an embedded pinned reference and a synthetic fixture.
It performs no training, downloads or dataset quality evaluation. Parameters:
36,783 total/trainable. The shared `evaluate(model, dataset)` accepts the adapter;
the CLI still rejects LiSenNet until audited pretrained loading is implemented.

## LiSenNet checkpoint audit

The official architecture is integrated, but its main pretrained baseline is
pending. A third-party GRU checkpoint strictly matches all 102 generator state
entries; its published recipe uses test for validation and g_best selection.
Checkpoint-specific selection logs are absent, so it is excluded from the main
held-out comparison pending independent provenance. See
`docs/lisennet_checkpoint_audit.md` and its JSON evidence. No LiSenNet quality
score has been generated.

## TF-GridNet architecture verification

```bash
python -m scripts.smoke_tfgridnet
```

Uses random weights and a pinned embedded ESPnet reference, including its original
STFT frontend. Verifies strict state loading, waveform equivalence, alignment,
short/silent audio, gradients and the shared evaluator with a synthetic fixture.
No checkpoint, optimizer step or dataset quality report is generated.
Parameters: 8,381,504 total/trainable; 540 network state entries. The local
profile uses FFT=512/hop=256, six blocks, hidden=192 and embedding=48; it is not a
published pretrained recipe. `configs/tfgridnet.yaml` records the full profile.
The CLI remains restricted to GTCRN until other checkpoint provenance is audited.

## Future plan

1. **Phase 1: Framework setup** — complete.
2. **Phase 2: Integrate pretrained checkpoints** — all three architectures
   integrated; GTCRN pretrained loading complete. LiSenNet and TF-GridNet
   pretrained baselines pending provenance and compatible configuration audits.
3. **Phase 3: Unified training pipeline** — shared data splits, loss, budget,
   reproducible seeds and selection rule; validation stays separate from train.
4. **Phase 4: Controlled benchmark comparison** — same fixed test manifest,
   metric code and hardware under matched training/measurement conditions.

Extract only necessary definitions/training/inference/evaluation logic; do not
copy complete upstream repositories. Record source revisions, required notices,
checkpoint training provenance and architectural capabilities for each adapter.

Architecture reference: https://github.com/kittytinglee/edge-kws-benchmark
