# Edge Speech Denoising Benchmark

One private repository for GTCRN, LiSenNet and TF-GridNet under a shared
waveform-to-waveform interface. No separate model forks or repositories.

## Current status

GTCRN network integration, official VCTK-DEMAND pretrained loading, single-WAV
inference, pinned data preparation and shared evaluation are implemented.
LiSenNet network and waveform adapter are integrated and verified against the
pinned upstream generator, with random weights only. Original TF-GridNet
network and offline mono waveform adapter are integrated. A distinct DNS pretrained
profile now has strict loading and version-matched reference checks. Training,
validation and losses remain Phase 3 placeholders;
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
  Local six-layer random profile plus a separate four-layer DNS pretrained profile.
  Pretrained loading/inference checks passed; full fixed-test evaluation is pending.

All adapters accept and return aligned mono `[batch, samples]` at 16 kHz.
Native feature processing belongs inside each adapter. `build_model("gtcrn")`
constructs a random-weight model. Pretrained weights must be explicitly loaded
with `load_gtcrn_checkpoint`; single-file and evaluation scripts do this.
TF-GridNet DNS weights are explicitly loaded with `load_tfgridnet_dns_checkpoint`
into `build_tfgridnet_dns_model()`; the registry default stays six-layer/random.
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
The evaluation CLI accepts the distinct DNS pretrained profile in Phase 2.10;
the six-layer random profile remains excluded from dataset evaluation.

## TF-GridNet checkpoint audit

Phase 2.8 records source provenance, hashes and tensor metadata in
`docs/tfgridnet_checkpoint_audit.md` and its JSON evidence. The preferred next
candidate is the original TF-GridNet DNS checkpoint linked by the ESPnet recipe.
Its 4-layer / hidden=128 / embedding=32 profile has 2,552,790 parameters and
362 matching tensor entries; it cannot load into the current 6-layer default.
The Phase 2.8 audit is historical. Phase 2.9 adds strict DNS pretrained loading
and equivalence checks against the pinned ESPnet 202308 release reference.
No full TF-GridNet VoiceBank report is recorded yet.
Any subsequent score is an external pretrained baseline, not a controlled
training comparison. V3 checkpoints remain separate architectural variants.

For a new Windows computer, see `docs/windows_restore.md`. Recording the Phase 2.8 audit needs only Python and Git.
Phase 2.9 inference checks additionally require PyTorch and the dependencies below.

## TF-GridNet DNS pretrained integration (Phase 2.9)

The explicit profile is `configs/tfgridnet_dns.yaml`: four layers / hidden=128 /
embedding=32 / embedding stride=4, 2,552,790 parameters and 362 state entries.
The six-layer default config and all three original network adapters are preserved.

```bash
python -m pip install -r requirements-tfgridnet.txt
python -m scripts.fetch_tfgridnet_checkpoint
python -m scripts.smoke_tfgridnet_pretrained
```

Install PyTorch separately. The fetcher pins HF revision, byte count and SHA-256,
using HTTPS and an atomic download. The loader validates every tensor before
strict assignment. The smoke check uses actual weights and synthetic waveforms;
it writes `results/tfgridnet_dns_pretrained_check.json`, not a quality report.

Reference: official ESPnet v.202308 release, matching the uploaded version label.
The training recipe's reported exact commit was not publicly retrievable. The
release's legacy iSTFT needs a representation bridge on modern PyTorch, documented
in `speech_denoising/models/tfgridnet/PRETRAINED.md`. These checks do not reproduce
the training environment or published quality scores.

Optional single-WAV inference (actual mono 16k PCM16 file required):

```bash
python -m scripts.inference_tfgridnet --input path/to/noisy.wav --output results/tfgridnet_demo/enhanced.wav
```

The 16k integration follows the official recipe default; exact training manifests
remain unavailable. Keep this as an external DNS-pretrained track, separate from
controlled training. Phase 2.9 Windows strict loading and release-reference checks passed.
Phase 2.10 routes this profile through the shared `evaluate(model, dataset)`;
see the commands below and `docs/tfgridnet_dns_evaluation.md`.

## TF-GridNet DNS unified evaluation (Phase 2.10)

From the repository root, install benchmark extras and recover the fixed test
split on a new computer. Existing verified DNS weights are reused.

```bash
python -m pip install -r requirements-benchmark.txt
python -m scripts.smoke_evaluation_cli
python -m scripts.prepare_data --split test
python -m scripts.evaluate --model tfgridnet --profile dns_ins20_epoch33 --metrics stoi si_snr si_snr_improvement --limit 5 --output results/tfgridnet_dns_test5.json
```

The five-utterance report is a subset check. After it succeeds, remove `--limit 5`
and use `--output results/tfgridnet_dns_test824.json` for the full fixed split.
PESQ is explicitly omitted above; after installing its backend, request
`--metrics pesq stoi si_snr si_snr_improvement` for all quality metrics.
Full instructions and report-retention steps: `docs/tfgridnet_dns_evaluation.md`.

The same dataset and metric/RTF function serve GTCRN and TF-GridNet. CLI reports
record the selected profile, pinned checkpoint and implementation fingerprints.
TF-GridNet DNS is an external pretrained track; it is not a controlled-training
comparison. Old GTCRN RTF and new-computer RTF must not be ranked together.
No TF-GridNet VoiceBank quality score is claimed before an actual dataset run.

## Future plan

1. **Phase 1: Framework setup** — complete.
2. **Phase 2: Integrate pretrained checkpoints** — all three architectures
   integrated; GTCRN pretrained evaluation complete. TF-GridNet DNS loading and
   release-reference checks verified on Windows; unified evaluation CLI integrated,
   with dataset quality evaluation pending execution. LiSenNet pretrained baseline remains pending provenance.
3. **Phase 3: Unified training pipeline** — shared data splits, loss, budget,
   reproducible seeds and selection rule; validation stays separate from train.
4. **Phase 4: Controlled benchmark comparison** — same fixed test manifest,
   metric code and hardware under matched training/measurement conditions.

Extract only necessary definitions/training/inference/evaluation logic; do not
copy complete upstream repositories. Record source revisions, required notices,
checkpoint training provenance and architectural capabilities for each adapter.

Architecture reference: https://github.com/kittytinglee/edge-kws-benchmark
