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

Training, validation and loss remain unimplemented. The YAML scaffold still uses
`phase: 1` for its configuration-only entry points; the TF-GridNet YAML files
record architectural profiles instead. Unifying training configuration belongs
to the next implementation stage.
