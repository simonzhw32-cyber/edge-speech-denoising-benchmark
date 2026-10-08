# Project history

This is a navigation record for completed integration work. Earlier audits
retain the decisions, source hashes and limitations known at their review dates.
Their references to a pending next step describe that historical state.

| Stage | Outcome | Evidence |
|---|---|---|
| Framework | Shared waveform interface, configuration scaffold and separate train/validate placeholders | Base interface and script entry points |
| GTCRN architecture | Pinned network and waveform adapter, checked against the upstream frontend | `speech_denoising/models/gtcrn/SOURCE.md`, `scripts/smoke_gtcrn.py` |
| GTCRN pretrained | Strict VCTK-DEMAND loading, single-WAV inference and full fixed-test evaluation | GTCRN `PRETRAINED.md` and committed full report |
| Unified evaluation | Pinned VoiceBank preparation, per-utterance metrics and timing protocol | `configs/voicebank.json`, `docs/benchmark_protocol.md` |
| LiSenNet | Network/adapter reference checks; pretrained candidate held back by checkpoint-selection provenance | LiSenNet `SOURCE.md` and checkpoint audit |
| TF-GridNet architecture | Original offline six-layer local profile and embedded reference checks | TF-GridNet `SOURCE.md`, `scripts/smoke_tfgridnet.py` |
| TF-GridNet audit (2.8) | Identified a distinct four-layer DNS checkpoint and its provenance limits | `docs/tfgridnet_checkpoint_audit.md` and JSON |
| DNS pretrained (2.9) | Pinned strict loading and ESPnet 202308 release-reference checks | DNS `PRETRAINED.md` and integration JSON |
| DNS evaluation (2.10) | Shared CLI, 824-item three-metric run, then complete four-metric run | `docs/results.md`, retained full and partial reports |

The maintenance baseline is archive commit
`5eee7f12dd15f41326b1b988efecc9c70ddf50f7`. It contains both full-test reports.
This cleanup changes current documentation, ignore rules and presentation text;
it does not retrain models or revise historical evaluation records.

Early migrations used external `phase22_...` through `phase210_...` installers.
They are not repository entry points and are not needed after cloning current
source. Old installers check old source fingerprints and may reject later
versions. Use the current fetchers and [Windows setup](windows_restore.md)
when restoring a computer.

Subsequent work implements a draft training protocol, speaker split, length-aware
SI-SNR loss, paired waveform batching, epoch state restoration, independent
validation components and a synthetic GTCRN runner. Real train-data preflight,
bounded resource probes and guarded real GTCRN train/validation entry points are
also available. No selected real-data checkpoint exists.

| Stage | Outcome | Evidence |
|---|---|---|
| Training components (3.1–3.3) | Planning, loss/batching, state restoration and validation selection | `docs/training_protocol.md`, `docs/training_components.md`, `docs/training_state.md` |
| Synthetic runner (3.4) | Separate train/validate CLI and epoch-boundary resume fixture | `docs/training_runner.md` |
| Train data (3.5) | Complete local audio preflight and plan-bound dataset views | `docs/training_preflight.md`, archived `gtcrn_audio.json` |
| Resource probes (3.6) | GTCRN/LiSenNet bounded checks passed; capped TF-GridNet checks failed with OOM | `docs/training_probe.md`, `training_reports/phase36_windows/` |
| Real GTCRN runner | Verified train updates, immutable epoch state and separate complete held-out validation are wired; no real run claimed | `docs/real_training.md`, `speech_denoising/training/real_runner.py` |
| Memory trace | Batch-one TF-GridNet failure located in second block inter-RNN during forward | `b8aee8e`, archived trace JSON and launcher |

[Requirements and acceptance status](project_requirements.md) separates this
implementation progress from completed research experiments.

## GTCRN pilot and LiSenNet execution follow-up

`ae846dd` archived 25 GTCRN training epochs, 25 held-out validations and the
complete 824-item four-metric test. Validation chose epoch 24; final training
progress reached 67,375 optimizer steps. Weights are Release assets.

The follow-up implements random LiSenNet real training, independent validation
and selected-checkpoint test routing. Git-only provenance changes no longer block
new-run resume; source/runtime/data changes remain rejected. Historical reports
and weights are preserved. LiSenNet real-data execution and a matched three-model
comparison remain pending.


## LiSenNet pilot completed — 2026-10-08

LiSenNet random seed 42 completed 25 real training epochs and 25 independent validations. Validation selected epoch 21; all 824 fixed test utterances were evaluated with PESQ, STOI, SI-SNR and SI-SNR improvement. See [experiment results](lisennet_seed42_pilot.md).

This update supersedes earlier statements that LiSenNet real execution is pending. TF-GridNet real training and a complete three-model comparison remain pending; edge-device efficiency is not established.
