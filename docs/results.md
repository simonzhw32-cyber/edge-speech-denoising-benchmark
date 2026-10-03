# Recorded pretrained results

The tables use the committed full-test JSON reports. Both contain 824 unique
utterances, full metric coverage and no failed utterances. Recomputing the
per-utterance macro means from those records agrees with each summary.
The noisy baseline summaries, utterance order and dataset metadata agree.

Dataset revision: `20879f4f9aab3d0b9263993667e7711a3ae1416d`.
Manifest rows SHA-256: `57111c2eb4ce11466e4e489c557a0a05f37a7f873f9800e26ef0c0555a2d978e`.

## Quality and parameter count

| Model / checkpoint training data | PESQ-WB | STOI | SI-SNR (dB) | SI-SNR improvement (dB) | Parameters: all / trainable |
|---|---:|---:|---:|---:|---:|
| GTCRN (VCTK-DEMAND) | 2.86969 | 0.94029 | 18.79606 | 10.35052 | 48,245 / 23,669 |
| TF-GridNet (DNS) | 2.77327 | 0.94310 | 19.40151 | 10.95598 | 2,552,790 / 2,552,790 |

- [GTCRN full report](../benchmark_reports/gtcrn_pretrained_voicebank_test824_windows.json)
- [TF-GridNet DNS full report](../benchmark_reports/tfgridnet_dns_pretrained_voicebank_test824_windows.json)
- [LiSenNet checkpoint decision](lisennet_checkpoint_audit.md): no accepted score.

The GTCRN checkpoint is trained on VCTK-DEMAND; TF-GridNet uses an external
DNS 2020 recipe. Training data, budgets and selection procedures are not matched.
Exact DNS training manifests remain unavailable, including independent proof of
non-overlap with VoiceBank. Treat these as descriptive pretrained baselines.

## Runtime records

Both reports specify CPU FP32, batch=1, one thread, one warmup and three timed
runs per utterance. The platform/processor strings and main metric dependencies
match, but do not identify a physical machine or establish matched run conditions.
The runs were made on different computers; no speed ranking is inferred.

| Recorded run | RTF | Windows environment snapshot |
|---|---:|---|
| GTCRN | 0.01557 | [packages](../benchmark_reports/gtcrn_windows_environment.txt) |
| TF-GridNet DNS | 0.29897 | [packages](../benchmark_reports/tfgridnet_dns_all_metrics_windows_environment.txt) |

RTF is total median inference time divided by total audio duration. It measures
whole-utterance inference, not streaming latency or deployment on an edge device.
A runtime comparison requires rerunning the models under matched hardware,
software and timing conditions.

## Retained records

The [earlier DNS three-metric report](../benchmark_reports/tfgridnet_dns_pretrained_voicebank_test824_windows_without_pesq.json)
explicitly omits PESQ and has its own [environment snapshot](../benchmark_reports/tfgridnet_dns_evaluation_windows_environment.txt).
The [DNS pretrained smoke report](../benchmark_reports/tfgridnet_dns_pretrained_check_windows.json)
contains synthetic reference/load checks, not VoiceBank quality scores.

The DNS full report records evaluation code commit `5e0aa5820c210586fef502a6bd20924faa3bde9e`
and source fingerprints. `5eee7f1` adds that result to Git; it is not the code
commit under which the evaluation itself ran. Keep these historical reports
unchanged when maintaining current documentation or source.
