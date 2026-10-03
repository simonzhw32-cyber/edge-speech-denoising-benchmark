# TF-GridNet DNS unified evaluation — Phase 2.10

The accepted profile is `dns_ins20_epoch33`: original offline TF-GridNet,
four blocks, 2,552,790 parameters, 362 checkpoint state entries. Phase 2.9
Windows checks strictly loaded the pinned checkpoint and matched the
version-matched reference on synthetic waveforms (maximum difference 0.0).
Those checks are not VoiceBank quality scores.

## One runner, explicit pretrained selection

`scripts.evaluate` supports the GTCRN VCTK-DEMAND track and TF-GridNet DNS
track. TF-GridNet requires both `--model tfgridnet` and
`--profile dns_ins20_epoch33`; the registry's six-block random default is
never used by this evaluation branch. Missing, incompatible or corrupt
weights abort, with no random-weight fallback. LiSenNet remains rejected
pending accepted checkpoint provenance. GTCRN commands retain their default
checkpoint and output. TF-GridNet uses separate defaults.

Both branches call the unchanged `evaluate(model, dataset)` in
`speech_denoising/metrics/audio_metrics.py`. Dataset preparation, hashes,
fixed test split, waveform handling, metric definitions and RTF protocol
are shared. See `docs/benchmark_protocol.md`. The CLI adds profile,
checkpoint provenance, source fingerprints and Git context to each report.

## New Windows computer

Run from the repository root. Copy commands only, not terminal prompts or
output. PyTorch, TF-GridNet dependencies and the DNS checkpoint must already
be present; Phase 2.9 restore commands are in `docs/windows_restore.md`.

```bat
python -m pip install -r requirements-benchmark.txt --index-url https://pypi.org/simple
python -m scripts.smoke_benchmark
python -m scripts.smoke_evaluation_cli
python -m scripts.prepare_data --split test
```

Preparation downloads only the pinned 132 MB test parquet, checks its
size/hash, and writes all 824 paired utterances plus a manifest under
`data/voicebank-demand-16k/`. No train split is downloaded. To reuse a
manually downloaded pinned test shard, pass `--parquet PATH`. No audio
resampling or protocol changes are needed.

First run five utterances to check wiring:

```bat
python -m scripts.evaluate --model tfgridnet --profile dns_ins20_epoch33 --metrics stoi si_snr si_snr_improvement --limit 5 --output results/tfgridnet_dns_test5.json
```

Expect `subset_or_fixture`, `5 / 824`, 2,552,790 parameters and zero metric
failures. This is not the full benchmark. Without PESQ, the report explicitly
lists `pesq` in `not_computed`; no substitute metric is used.

After the five-utterance run succeeds, process all 824 with the same settings:

```bat
python -m scripts.evaluate --model tfgridnet --profile dns_ins20_epoch33 --metrics stoi si_snr si_snr_improvement --output results/tfgridnet_dns_test824.json
```

Expect `full_fixed_test`, `824 / 824` and zero failures. Metric coverage is
still partial when PESQ is omitted. Defaults remain CPU FP32, batch=1,
one thread, one warmup and three timed runs per utterance. Do not change
the profile or select checkpoints based on these fixed test scores.

## PESQ and complete quality coverage

PESQ is optional at installation and mandatory for all-four-metric coverage.
On Windows its source build requires a working C/C++ toolchain. NumPy is
already installed for this project. Attempt installation separately:

```bat
python -m pip install Cython wheel --index-url https://pypi.org/simple
python -m pip install pesq==0.0.4 --no-build-isolation --index-url https://pypi.org/simple
python -c "from pesq import pesq; print('PASS: PESQ import')"
```

If compilation fails, retain the error and resolve the compiler setup before
requesting PESQ. Do not label the three-metric report complete in quality
coverage. After successful installation, check five utterances with all four
metrics, then run the full split:

```bat
python -m scripts.evaluate --model tfgridnet --profile dns_ins20_epoch33 --metrics pesq stoi si_snr si_snr_improvement --limit 5 --output results/tfgridnet_dns_test5_all_metrics.json
python -m scripts.evaluate --model tfgridnet --profile dns_ins20_epoch33 --metrics pesq stoi si_snr si_snr_improvement --output results/tfgridnet_dns_test824_all_metrics.json
```

## Retain reports and interpret comparisons

`results/`, data and checkpoint files are ignored by Git. After confirming
824 utterances, zero failures and the intended metric coverage, copy the
chosen report to `benchmark_reports/` and capture the environment. For the
all-four-metric run:

```bat
copy /Y results\tfgridnet_dns_test824_all_metrics.json benchmark_reports\tfgridnet_dns_pretrained_voicebank_test824_windows.json
python -m pip freeze > benchmark_reports\tfgridnet_dns_evaluation_windows_environment.txt
```

For a three-metric run, retain an explicitly distinct filename such as
`benchmark_reports/tfgridnet_dns_pretrained_voicebank_test824_windows_without_pesq.json`.
Do not replace the GTCRN report or the earlier DNS synthetic check report.
Review `scope`, metric coverage, dataset source/rows hash, checkpoint SHA,
profile and measurement fields before committing a result.

This is an external DNS-pretrained baseline. GTCRN uses a VCTK-DEMAND-trained
checkpoint; training sets, budgets and selection procedures are not matched.
Exact DNS training manifests remain unavailable. Shared test and metrics
permit descriptive pretrained results, not a controlled-training ranking.
The old GTCRN RTF came from another computer. Compare runtime only after
rerunning both models on the same hardware, dependencies and timing protocol;
this offline RTF does not establish streaming latency or edge-device speed.

The installer performs no download, inference, training, evaluation, commit
or push. Quality reports are produced only by the commands above.
