# Windows setup and recovery

Code, configs, audits and committed reports are restored through Git.
Python packages, data, weights and ignored `results/` need separate preparation.
Copy commands only, not terminal prompts or output; stop at the first error.

## Code and Python

Install [Git](https://git-scm.com/install/windows) and
[Python](https://www.python.org/downloads/windows/) if needed. Enable Python's
PATH option and reopen CMD. The project requires Python 3.10+; recorded Windows
integration used Python 3.12 and 3.13 with PyTorch 2.14.1.

If the repository is not present:

```bat
git clone https://github.com/simonzhw32-cyber/edge-speech-denoising-benchmark.git "%USERPROFILE%\edge-speech-denoising-benchmark"
```

Follow Git's browser authentication prompt for the private repository. Then:

```bat
cd /d "%USERPROFILE%\edge-speech-denoising-benchmark"
git status --short
git log -1 --oneline
```

For an existing clean checkout, use `git pull --ff-only`. Preserve pending edits
before updating. Current source is restored by Git; old phase installers are
historical migration tools, not recovery steps. Keep the repository private.

## Dependencies

From the repository root:

```bat
python -m pip install torch==2.14.1 --index-url https://pypi.org/simple
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
python -m pip install -r requirements-benchmark.txt -r requirements-tfgridnet.txt --index-url https://pypi.org/simple
```

For complete metric coverage:

```bat
python -m pip install Cython wheel --index-url https://pypi.org/simple
python -m pip install pesq==0.0.4 --no-build-isolation --index-url https://pypi.org/simple
python -c "from pesq import pesq; print('PASS: PESQ import')"
```

If PESQ reports that Microsoft Visual C++ is missing, install the C++ workload
and recommended components from the official
[Build Tools installer](https://aka.ms/vs/17/release/vs_buildtools.exe).
The installation parameters are documented by
[Microsoft](https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2022).
Reopen CMD and retry the PESQ installation. Without that backend, explicitly
omit PESQ from the requested metrics; keep the report labeled partial coverage.

## Weights, data and checks

Fetch only the assets needed for the task. For GTCRN:

```bat
python -m scripts.fetch_gtcrn_checkpoint
python -m scripts.smoke_gtcrn_pretrained
```

For TF-GridNet DNS:

```bat
python -m scripts.fetch_tfgridnet_checkpoint
python -m scripts.smoke_tfgridnet_pretrained
```

Fetchers verify pinned hashes and reuse matching local weights. The pretrained
checks use synthetic input and do not generate dataset quality scores.
The legacy reference may emit a deprecated-STFT warning.

For dataset evaluation:

```bat
python -m scripts.smoke_benchmark
python -m scripts.smoke_evaluation_cli
python -m scripts.prepare_data --split test
```

Preparation downloads the 132 MB test parquet and validates 824 paired utterances.
Train data is not needed for these evaluations. Evaluation commands are in the
[README](../README.md) and [DNS guide](tfgridnet_dns_evaluation.md).
Do not rerun a complete benchmark solely because the computer changed; rerun
models together only when measuring runtime under matched conditions.

## Keep work between computers

Record `git log -1 --oneline` and ensure intended changes have been committed
and pushed before leaving. If Git asks for an author identity, configure
`git config user.name "YOUR_NAME"` and `git config user.email "YOUR_EMAIL"`
for this repository with your own values.

Git does not upload ignored data/checkpoints/results. Copy a reviewed result
into `benchmark_reports/` with its matching environment snapshot when it needs
to be retained. Back up any other uncommitted local assets separately.
