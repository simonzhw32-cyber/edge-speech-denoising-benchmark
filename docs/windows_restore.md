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

If the repository is private, follow Git's browser authentication prompt. Then:

```bat
cd /d "%USERPROFILE%\edge-speech-denoising-benchmark"
git status --short
git log -1 --oneline
```

For an existing clean checkout, use `git pull --ff-only`. Preserve pending edits
before updating. Current source is restored by Git; old phase installers are
historical migration tools, not recovery steps. Repository visibility must follow
the agreement with the group; opening it is not required for local restoration.

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

## GPU environment and training data

Use a separate environment so CPU evaluation and GPU development do not share
an interpreter accidentally. An environment copied from another computer is
not a portable installation; recreate it instead. The archived Windows probes
used Python 3.13.16 and PyTorch 2.14.1+cu130. To reproduce that recorded build on
compatible Windows hardware, create a fresh environment, then run:

```bat
python -m venv venv\gpu
venv\gpu\Scripts\python.exe -m pip install "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
venv\gpu\Scripts\python.exe -m pip install -r requirements.txt -r requirements-benchmark.txt --index-url https://pypi.org/simple
venv\gpu\Scripts\python.exe -m pip check
venv\gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

The CUDA build above records the known environment; it is not a universal choice
for every driver or GPU. Select an appropriate official build on different
hardware. `requirements.txt` already installs PyYAML through the base package;
there is no separate missing training dependency to add. The legacy TF-GridNet
reference additionally needs `requirements-tfgridnet.txt`. PESQ and its build
tools are needed for complete quality evaluation, not for the SI-SNR loss.

For planned training, restore the parent train source if it is missing:

```bat
venv\gpu\Scripts\python.exe -m scripts.prepare_data --split train
```

Use the current preflight tool if restoring a new local dataset. Choose a new
output filename when an old report exists. Archived preflight reports contain
paths from the old machine and do not replace a newly prepared local manifest.
Do not restart a full 824-item benchmark merely to restore the training setup.

The current training CLI still accepts only descriptions or synthetic fixtures.
Do not expect a real-data run until the next implementation milestone described
in [project requirements](project_requirements.md) has been completed.

Retain real training checkpoints separately once training begins: `checkpoints/`
and `results/` are ignored, so pushing code will not preserve learned weights.
Record the source commit, environment, manifest/split fingerprints and checkpoint
hash alongside any future result that is copied into a tracked reports folder.
