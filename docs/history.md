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

## Paper alignment and three-model pilots — 2026-11-09

TF-GridNet completed its 25-epoch unified-recipe pilot (`f8e4489`), so all three
models now have a seed-42 25-epoch result on the same speaker-disjoint split.
The [paper alignment](paper_alignment.md) record tabulates the original papers
against the repository: LiSenNet is the only model with a paper-defined
VoiceBank recipe and epoch count (100), GTCRN's paper is silent on the VCTK epoch
budget, and TF-GridNet's paper does not train on VoiceBank-DEMAND. A separate
LiSenNet paper-recipe 100-epoch run was launched on 2026-11-09 under
`results/paper_runs/lisennet_paper_seed42_100ep`. This supersedes earlier
statements that a three-model comparison is pending; it does not convert the
pilot into a paper reproduction.

## Paper-track audit corrections — 2026-11-09

Two independent read-only audits checked the LiSenNet and TF-GridNet paper
claims. The LiSenNet audit confirmed the crop-then-normalize order, 2 s / batch 4
/ AdamW 5e-4 / betas (0.8, 0.99) / StepLR 0.98 / clip 5.0 / 0.9-0.1-0.05 loss,
and the 36,783-parameter generator. It found two real defects: the two
discriminator head layers had lost their spectral normalization, and the
200-utterance selection subset was a truncated stride that never scored the last
24.7 % of validation. Both were corrected, along with the official `[1]` learnable-sigmoid shape and
retention of the final partial batch; earlier attempts are archived under
`_aborted_lisennet_full_utterance_norm`, `_aborted_lisennet_plain_d_linear` and
`_aborted_lisennet_tail_batch`.
The current run uses SHA-256-ranked validation selection, logs the LR actually
used, records real git-dirty state, and hashes the evaluation path as well as the
training code.

The TF-GridNet audit confirmed the 8 ms vs 16 ms hop mismatch changes activation
shapes, MACs (2.00x) and memory, but not parameter count (8,381,504). Existing
checkpoints would load silently under stride 128, so that remains a separate,
reviewed change; the docs now also record that the repo uses plain Hann rather
than the paper's sqrt-Hann. No training run was affected by either audit.

## GTCRN paper track implemented — 2026-11-09

Added `configs/gtcrn_paper.yaml`,
`speech_denoising/training/paper_gtcrn.py`,
`scripts/train_gtcrn_paper.py` and `scripts/smoke_gtcrn_paper.py`. The track
keeps `GTCRNNetwork` and the official sqrt-Hann frontend, with a spectrum-level
loss matching the official `30/70/1` weighting, Adam 1e-3 and
`ReduceLROnPlateau(min, 0.5, patience=5)`. Fifteen synthetic CPU checks pass (15/15)
and a 3-step fixture writes a valid run. No real run was launched: the paper
does not define a VCTK-DEMAND epoch count, so `--epochs` and
`--segment-seconds` remain required operator inputs and are recorded as such.
