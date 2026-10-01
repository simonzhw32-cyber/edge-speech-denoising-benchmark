# LiSenNet pretrained checkpoint audit

Review date: 2026-10-02 (Asia/Shanghai).

## Decision

LiSenNet architecture/adapter integration is verified. Its main pretrained
baseline remains pending. No public author checkpoint was found in the
inspected official tree, releases or nonempty issue threads. This is a bounded
search result, not a claim that the authors have no weights elsewhere.

A third-party checkpoint is structurally compatible, but is not admitted to
our held-out benchmark comparison because its published training recipe uses
the test split for validation and checkpoint selection, and the checkpoint
contains no epoch/selection logs establishing a different provenance.

## Sources

- Official repository: https://github.com/hyyan2k/LiSenNet
- Official revision: `df4e481aa706fbc344d1013f7a1cbfd46228c8ab`
- Third-party model: https://huggingface.co/claroche1/LiSenNet
- Model revision: `4fe3356f04dc3cebb32335e0c194bb6284c695f8`
- Candidate: `gru/g_best`, 177,557 bytes
- Candidate SHA-256: `f983e79d7018ec7e0df6d5cb457b2178d78c019c759f266cc5bb83f463e0b8d5`
- Recipe repository: https://github.com/LarocheC/eco8-neaixt
- Recipe revision: `b72d38e91874752ced772ab662ee5390776019a5`
- Reviewed training script: [lisennet/train.py](https://github.com/LarocheC/eco8-neaixt/blob/b72d38e91874752ced772ab662ee5390776019a5/lisennet/train.py)
- Training script SHA-256: `158153aa2062be29b170557f93d56128999b411b7d3034dced4949017ecbb43c`

## Compatibility evidence

Downloaded the pinned candidate for local inspection, verified its published
LFS SHA-256, and loaded it with `torch.load(..., weights_only=True)`.
The only container key is `generator`. All 102 state entries map directly into
the extracted official generator: no missing/unexpected keys or shape
mismatches, and `strict=True` loading succeeds. The model has 36,783 parameters.
This establishes structural compatibility, not benchmark eligibility or a
completed end-to-end pretrained inference parity check.

The published configuration retains 16 channels, two blocks, FFT=512,
hop=256 and compression=0.3. It records batch size 16, rather than the official
configuration's 4, and seed 1234. Training is a third-party reproduction, not
author-provided pretrained weights. Full reviewed config is in the JSON audit.

## Selection evidence and its limits

In the pinned training script, lines 181-188 construct the training dataset
from `hf["train"]`, but the validation dataset from `hf["test"]`.
Lines 224-230 evaluate that validation loader, compare its PESQ with the best
validation PESQ and save `g_best` when it improves.

The published default recipe therefore consults the nominal test set to choose
weights. A subsequent score on the same 824 utterances cannot be described as
an independently held-out result under that recipe. The downloaded checkpoint
contains only weights; its actual training/selection logs are unavailable, so
we cannot prove this exact file followed every default or assert a different,
independent selection protocol. This unresolved provenance is why the main
benchmark excludes the candidate for now.

A separately labeled exploratory demo could use these weights after waveform
parity checks, but its scores must not be presented as author results or as
our held-out benchmark. No quality scores are produced in this audit.

## Project follow-up

Keep the tested official LiSenNet network, NumPy normalization and native
phase recovery. Do not change the common dataset or metric protocol to fit
this candidate. Seek auditable author/independent weights, or train later in
Phase 3 using validation drawn only from train. Do not tune or choose checkpoints
on the fixed test set. Proceed next with TF-GridNet architecture integration.

GTCRN's existing pretrained result remains an explicitly pretrained result;
this audit does not retroactively establish a controlled or independently
selected training recipe for GTCRN. Controlled comparisons require all models
to follow the future shared training/selection protocol.
