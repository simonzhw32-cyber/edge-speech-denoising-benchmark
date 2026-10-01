# Edge Speech Denoising Benchmark

One private repository for a unified speech denoising benchmark.
Phase 1 provides interfaces and configuration only: no model code migration,
training, dataset downloads, checkpoints, or benchmark scores.

## Dataset

**VoiceBank-DEMAND-16k**, a standard speech enhancement benchmark.

- Train split: **11572** paired utterances.
- Test split: **824** paired utterances.
- The test split is fixed following the dataset protocol.
- Validation will be a reproducible subset of the training split; never use
  the test split for validation, checkpoint selection, or tuning.
- Future data preparation must pin the source/revision and record paired IDs,
  sample rate, file hashes, and split membership in manifests.
- Data paths are intentionally unset. Data stays outside Git.

## Baseline models

- **GTCRN**: lightweight real-time speech enhancement model.
- **LiSenNet**: lightweight sub-band and dual-path speech enhancement model.
- **TF-GridNet**: time-frequency modeling based speech enhancement model.

Each adapter accepts a floating-point mono waveform `[batch, samples]` at
16 kHz and returns an aligned waveform of the same shape, sample rate,
device, and dtype. Native STFT, padding, and inverse transforms belong inside
adapters. Streaming/causality capabilities must be documented separately;
sharing an interface does not imply equal latency or streaming support.
The three adapters currently raise `NotImplementedError`.

## Evaluation

All models will use `evaluate(model, dataset)` and one shared metric module:

- PESQ (16 kHz wideband)
- STOI
- SI-SNR (dB)
- SI-SNR improvement (enhanced SI-SNR minus noisy SI-SNR, dB)
- RTF (inference seconds / audio duration seconds)
- Parameter count (all registered parameters)

Metrics are interfaces only in Phase 1; no scores are fabricated. Future RTF
measurements must record hardware, device, precision, threads, batch size,
warmup, synchronization, and whether frontend/transfer time is included.
Quality scores will be averaged per utterance over the full fixed test set.

## Setup and entry points

Python 3.10 or newer:

```bash
python -m venv .venv
# Activate the environment using your operating system's command.
python -m pip install -r requirements.txt
python -m scripts.prepare_data --config configs/default.yaml
python -m scripts.train --config configs/gtcrn.yaml
python -m scripts.validate --config configs/gtcrn.yaml
python -m scripts.inference --config configs/gtcrn.yaml
python -m scripts.evaluate --config configs/gtcrn.yaml
```

Entry points validate the configuration and exit with an explicit Phase 1
placeholder message. They do not download data or weights or execute models.
Training and validation have separate entry points. Model configs inherit
`default.yaml`; paths are resolved relative to the repository root.

## Future plan

1. **Phase 1: Framework setup** — shared interfaces, configs, separate scripts.
2. **Phase 2: Integrate pretrained checkpoints** — audit upstream licenses and
   revisions; extract only necessary definitions/inference code; map weights;
   verify waveform alignment and compare with upstream outputs. Record native
   checkpoint training settings; report these as pretrained comparisons.
3. **Phase 3: Unified training pipeline** — implement paired manifests, a fixed
   train/validation split, losses, optimizer, seeds, and resumable training.
   `train.py` trains only; `validate.py` handles validation independently.
4. **Phase 4: Controlled benchmark comparison** — fix data, training budget,
   selection rule, metric implementation, and measurement hardware; evaluate
   all three adapters on the same 824 test utterances.

## Integration order and provenance

Integrate GTCRN first, LiSenNet second, TF-GridNet third. For each, retain only
necessary model definitions, training, inference, and evaluation logic behind
these interfaces. Do not fork or copy entire upstream repositories or create
separate model repositories. Preserve required attribution and license texts
when code is integrated. Do not assign an upstream license to this new code
without a deliberate licensing decision.

Architecture reference: https://github.com/kittytinglee/edge-kws-benchmark
(shared waveform interface and separation of preliminary and controlled results).
The repository must remain private; configuration files cannot enforce GitHub
visibility, which must be set on GitHub.
