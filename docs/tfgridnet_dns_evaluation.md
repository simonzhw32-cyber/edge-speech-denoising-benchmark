# TF-GridNet DNS evaluation

Use the original offline four-layer `dns_ins20_epoch33` profile: 2,552,790
parameters, 362 state entries. The six-layer registry default is a separate
random-weight architecture. The CLI requires an explicit DNS profile and strictly
loads the pinned checkpoint; load failures abort evaluation.

The production adapter and GTCRN share the same dataset/metric runner. Follow
[benchmark protocol](benchmark_protocol.md) for score and timing definitions,
[Windows setup](windows_restore.md) for dependencies and data recovery, and
[PRETRAINED.md](../speech_denoising/models/tfgridnet/PRETRAINED.md) for provenance.

## Run from the repository root

Prepare the pinned checkpoint and test data if they are not already local:

```sh
python -m scripts.fetch_tfgridnet_checkpoint
python -m scripts.prepare_data --split test
```

Check five utterances with the four quality metrics:

```sh
python -m scripts.evaluate --model tfgridnet --profile dns_ins20_epoch33 --metrics pesq stoi si_snr si_snr_improvement --limit 5 --output results/tfgridnet_dns_test5_all_metrics.json
```

Expect `subset_or_fixture`, `5 / 824`, no omitted metrics and zero failures.
For the complete fixed split:

```sh
python -m scripts.evaluate --model tfgridnet --profile dns_ins20_epoch33 --metrics pesq stoi si_snr si_snr_improvement --output results/tfgridnet_dns_test824_all_metrics.json
```

Accept the report as complete only when it has `full_fixed_test`, `824 / 824`,
no failures, full coverage and `all_quality_metrics_complete: true`.
Without PESQ, explicitly request `--metrics stoi si_snr si_snr_improvement`
and keep a distinct output filename. That report has partial metric coverage.

Timing defaults are CPU FP32, batch=1, one thread, one warmup and three repeats.
Keep them fixed for a matched runtime experiment. Do not select profiles or
checkpoints based on the fixed test scores.

## Retain results

Review dataset fingerprints, checkpoint SHA, selected profile, coverage and
measurement conditions before copying a report out of ignored `results/`.
For a newly completed Windows full-metric run:

```bat
copy /Y results\tfgridnet_dns_test824_all_metrics.json benchmark_reports\tfgridnet_dns_pretrained_voicebank_test824_windows.json
python -m pip freeze > benchmark_reports\tfgridnet_dns_all_metrics_windows_environment.txt
```

Those canonical files already contain the recorded complete run. Use new
filenames for additional experiments instead of replacing historical evidence.
The earlier `*_without_pesq.json` report and its environment snapshot are retained;
the synthetic pretrained check report is a separate kind of evidence.

The DNS checkpoint is an external pretrained track. Exact training manifests
and selection logs are absent. GTCRN uses another training source, and its
recorded RTF came from another computer. See [results](results.md) for the
committed scores and comparison limits.
