# Paper-to-repository alignment

Status: **active alignment record**, started 2026-11-09. This page separates two
questions that the repository previously mixed:

1. **Unified benchmark** — one data split, crop, optimizer and loss for all
   models, so cross-model numbers are comparable. This is what
   `configs/training_protocol.yaml` and `speech_denoising/training/real_runner.py`
   implement.
2. **Paper recipe** — the dataset, crop, optimizer, schedule and objective
   stated by each model's original paper. This is a different experiment and
   must not be presented as a controlled comparison with (1).

Historical reports under `training_reports/` and `benchmark_reports/` are
immutable. New paper-track work is written under `results/paper_runs/` with its
own `run.json` and is not merged into the old run directories.

## Sources

| Model | Paper | Pinned official code |
|---|---|---|
| GTCRN | ICASSP 2024, pp. 971–975, DOI `10.1109/ICASSP48485.2024.10448310` | `Xiaobin-Rong/gtcrn@502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73` |
| LiSenNet | arXiv `2409.13285` | `hyyan2k/LiSenNet@df4e481aa706fbc344d1013f7a1cbfd46228c8ab` |
| TF-GridNet | arXiv `2211.12433v2` | upstream ESPnet implementation (not vendored) |

The GTCRN paper is not open access and has no arXiv version. The values below
come from the five-page ICASSP text; where the paper is silent, the text says so
instead of inferring a value from the official README.

## GTCRN

| Item | Paper | Repository unified recipe |
|---|---|---|
| Data | VCTK-DEMAND, 16 kHz, 11,572 train / 824 test; 1,572 train utterances used for validation | same corpus; speaker-disjoint 10,778 train / 794 validation / 824 test |
| STFT | sqrt-Hann, 32 ms window, 16 ms hop, FFT 512 | `n_fft=512`, `hop=256`, sqrt-Hann |
| Batch | 4 on VCTK-DEMAND; 16 on DNS3 | 4 |
| Segment | not stated for VCTK; DNS3 uses 8 s chunks | 4 s (64,000 samples) |
| Optimizer | Adam, lr 1e-3, halved after 5 epochs without validation improvement | AdamW, lr 1e-3, no scheduler |
| Loss | `0.01*L_SISNR + 0.7*L_mag + 0.3*(L_real + L_imag)`; compressed magnitude 0.3, RI normalized by magnitude 0.7 | negative SI-SNR only |
| Epochs | **not stated** | 25-epoch pilot |
| Parameters | 23.7 K (paper abstract), 39.6 MMACs/s | 48,245 all / 23,669 trainable; 33.0 MMACs/s per official README |

Consequences:

- The STFT and batch size already align. The optimizer family, schedule, loss
  and epoch budget do not.
- **A “paper epoch count” for GTCRN does not exist.** The paper gives no VCTK
  epoch count. The DNS3 recipe (8 s chunks, 40,000 pairs per epoch, batch 16)
  and the separate SEtrain DNS3 template's 200 epochs must not be transplanted
  to VCTK.
- The parameter-count discrepancy is explained by the official README: 48.2 K
  includes unlearnable ERB parameters, while the paper's 23.7 K is the learnable
  count. The MAC discrepancy (39.6 vs 33.0 MMACs/s) is not explained there and
  remains recorded, not resolved.
- `configs/paper_recipes.yaml` stores the GTCRN paper values as reference
  metadata. An executable paper track now exists in
  `configs/gtcrn_paper.yaml`, `speech_denoising/training/paper_gtcrn.py`,
  `scripts/train_gtcrn_paper.py` and `scripts/smoke_gtcrn_paper.py`; its 15
  synthetic checks pass and a 3-step CPU fixture completes. No real GTCRN
  paper-track run has been launched because the paper supplies no VCTK epoch
  budget and a real run must take that budget from the operator. The
  implementation plan is in
  [the GTCRN paper-track plan](gtcrn_paper_track_plan_2026-11-09.md).
- The official repository has no trainer; the README redirects to the separate
  SEtrain template, whose epoch budget is a DNS3 per-run value. No source ties
  an epoch count to VCTK-DEMAND, so `epochs_vctk: null` is intentional.
- The hybrid loss normalizes predicted and true RI **each by their own**
  magnitude^0.7 (not both by the clean magnitude), and its SI-SNR is
  `-log10(ratio)`, i.e. negative dB/10. The official `30/70/1` weighting is
  exactly 100× the paper's `0.01/0.7/0.3`; the two scales are not directly
  comparable.

## LiSenNet

| Item | Paper / official config | Repository unified recipe | Repository paper track |
|---|---|---|---|
| Data | VoiceBank+DEMAND, 11,572 train / 872 test, 48→16 kHz | same corpus; 10,778 / 794 / 824 speaker-disjoint | same split as unified (documented deviation) |
| STFT | Hann, 32 ms / 16 ms, FFT 512 | Hann 512/256 | Hann 512/256 |
| Segment | 2 s (`cut_len=32000`) | 4 s (64,000) | **2 s** |
| Batch | 4 | 4 | 4 |
| Optimizer | AdamW, lr 5e-4, betas (0.8, 0.99), wd 0 | AdamW, lr 1e-3, betas (0.9, 0.999), wd 0 | **AdamW 5e-4, (0.8, 0.99), wd 0** |
| Schedule | StepLR, step 1, gamma 0.98 | none | **StepLR 0.98/epoch** |
| Clip | L2 5.0 | 5.0 | 5.0 |
| Loss | `0.9*L_mag + 0.1*L_comp + 0.05*L_adv`; D predicts PESQ rescaled to (0,1) | negative SI-SNR | **same GAN objective** |
| Epochs | **100** | 25-epoch pilot | **100 (launched 2026-11-09)** |
| Parameters | 36,783 generator | 36,783 generator | 36,783 generator + 181,650 discriminator |

LiSenNet is the only one of the three models whose paper defines both a
VoiceBank recipe and an epoch count. The paper track therefore has a concrete
target: 100 epochs, 2 s crops, AdamW 5e-4 with 0.98 decay, and the
PESQ-conditioned GAN loss.

Known deviations, recorded rather than hidden:

- The repository uses its speaker-disjoint split (10,778 / 794) instead of the
  paper's 11,572 / 872 random split, so the test set is 824 rather than 872.
- Cropping happens first, then both signals are divided by the cropped noisy
  standard deviation, matching the official `data_module.py`. An earlier run
  normalized the full utterance before cropping; it was stopped and archived
  under `results/paper_runs/_aborted_lisennet_full_utterance_norm` without being
  reported as a result.
- Validation during the 100-epoch run uses a fixed, SHA-256-ranked 200-utterance
  subset spread over all 794 held-out utterances for checkpoint selection; the
  exact indices and their hash are written to `validation_sample.json`. Full
  validation and the 824-item test are separate post-run evaluations.
- The paper defines no validation split and selects no checkpoint by PESQ;
  the subset/selection protocol is a repository adaptation, not a paper value.
- The current LiSenNet paper runner does not reject a changed `--epochs` on
  resume; run directories are treated as immutable, so extending a finished run
  must be a new run directory rather than an in-place resume.
- Validation feeds the raw utterance through the adapter and scores it against
  the raw clean signal. The official loader normalizes both signals by the noisy
  standard deviation before evaluation. The adapter multiplies its output back
  by that scale, so the comparison is equivalent up to one global factor, but
  PESQ is not strictly scale-invariant; this is recorded as a known deviation
  rather than claimed as an exact reproduction.
- The discriminator restores both spectral-normalized linear head layers and the
  official `[1]`-shaped learnable sigmoid slope. The training loop retains the
  final partial batch (10,778 = 2,694×4 + 2), matching `drop_last=False`. Two
  earlier attempts used plain `nn.Linear` heads and/or dropped the tail batch;
  they were stopped and archived under
  `results/paper_runs/_aborted_lisennet_plain_d_linear` and
  `_aborted_lisennet_tail_batch` without being reported as results.
- PESQ inside the discriminator target is computed per batch with the official
  `(pesq-1)/3.5` rescaling and clipping; batches with an invalid PESQ pair train
  the discriminator on the real term only, matching the official code.

## TF-GridNet

| Item | Paper | Repository local profile |
|---|---|---|
| Task/data | Separation and noisy-reverberant enhancement (WSJ0-2mix, SMS-WSJ, WHAMR!, WSJ0CAM-DEREVERB, L3DAS22); **no VoiceBank-DEMAND recipe** | VoiceBank-DEMAND 16 kHz |
| STFT | sqrt-Hann, 32 ms / **8 ms**, FFT 512, 257 bins | Hann, `n_fft=512`, `stride=256` → **16 ms hop** |
| Segment | 4 s per mixture | 4 s |
| Normalization | sample variance → 1.0, same factor on targets | network-internal variance normalization |
| Optimizer | Adam, lr 1e-3, halved after 3 epochs without validation improvement | AdamW, lr 1e-3, no scheduler |
| Clip | L2 1.0 | 5.0 |
| Loss | Wav+Mag+MC by default (SI-SDR+MC and Wav+Mag also evaluated) | negative SI-SNR |
| Epochs | not stated; paper reports minutes per epoch on 20,000 4 s segments | 25-epoch pilot |

Consequences:

- TF-GridNet's paper is not a VoiceBank-DEMAND paper. A VoiceBank run is an
  adaptation, so there is **no paper epoch budget to match** for this repository's
  data. Calling a 100-epoch VoiceBank run “the TF-GridNet paper recipe” would be
  incorrect.
- The 16 ms hop is a real recipe mismatch: the paper's 8 ms hop doubles the
  frame count (251 → 501 for a 4 s, 16 kHz segment), so activation shapes,
  MACs (≈2.00×) and memory change. Parameter shapes do not: the 6-layer local
  profile has 8,381,504 parameters at either hop. The code has no
  `stride == n_fft/2` assumption, but existing checkpoints would load silently
  under stride 128 and produce wrong results, so a correction is a new config
  profile plus pipeline plumbing and a full retrain, not a drop-in edit.
  See [the hop audit](tfgridnet_hop_audit_2026-11-09.md).
- The window shape is a second, independent deviation: the repository uses
  plain Hann (`spectral.py`), while the paper specifies square-root Hann. This
  was previously undocumented and is not caused by the hop choice.
- The loss mismatch is equally structural: the paper's default Wav+Mag+MC
  combines time-domain and compressed-magnitude terms with a mixture-constraint
  term, while the repository's unified runner optimizes negative SI-SNR.

## Execution record

### LiSenNet, 100 epochs

- Run directory: `results/paper_runs/lisennet_paper_seed42_100ep`
- Entry point: `python -m scripts.train_lisennet_paper`
- Recipe: 2 s crops, batch 4, AdamW 5e-4, betas (0.8, 0.99), StepLR 0.98,
  clip 5.0, `0.9*L_mag + 0.1*L_comp + 0.05*L_adv`, seed 42, 100 epochs.
- Selection: best validation PESQ on a SHA-256-ranked 200-utterance subset
  (`validation_sample.json`); full validation and the 824-item test are run
  after training. At 17:01 the run had completed 6 epochs (PESQ 2.1019,
  2.1877, 2.2577, 2.2532, 2.2913, 2.2824; best epoch 5) at ~8–11 min/epoch.
  A local watcher (`.tmp/finalize_lisennet.sh`, not tracked) waits for the
  trainer, then runs top-5 full-validation reselection and the 824-item test.
  The trainer's own end-of-run test pass is not used: its in-memory checkpoint
  loader predates the RNG-serialization fix, so it will fail after writing
  `run.json: complete`; the watcher loads the same checkpoints with
  `weights_only=False`. The full-validation reselection writes
  `validation_full_topk.json` and `selection_full.json` (the subset-based
  `selection.json` is left untouched), then evaluates the reselected epoch on
  the 824-item test as `test824_full_reselection.json`.
- Status: restarted 2026-11-09 16:03 after restoring spectral normalization on
  both discriminator head layers, matching the official learnable-sigmoid shape,
  retaining the final partial batch, and replacing the truncated stride subset
  with the SHA-256-ranked spread. Earlier attempts are archived under
  `results/paper_runs/_aborted_lisennet_full_utterance_norm` (crop/normalize
  order), `_aborted_lisennet_plain_d_linear` (discriminator heads), and
  `_aborted_lisennet_tail_batch` (tail batch). Results are written
  incrementally to `learning_curve.csv`, `validation/epoch_XXXX.json` and
  `selection.json`. Verify the frozen sources from the repository root with
  `sha256sum -c results/paper_runs/lisennet_paper_seed42_100ep/source.sha256`.
  Two files changed after launch, both recorded in
  `source.sha256.known_mismatch`: `scripts/reselect_lisennet_paper.py` (docstring
  only) and `speech_denoising/training/paper_lisennet.py` (checkpoint load/RNG
  serialization fix; training math, ordering and loss unchanged; the live trainer
  keeps the code it loaded at 16:03). `sha256sum -c` therefore reports those two
  lines as FAILED. Every other training-path file, including the discriminator,
  verifies OK.

Reproduce the launch:

```sh
python -m scripts.train_lisennet_paper \
  --manifest data/voicebank-demand-16k/train_manifest.json \
  --run-dir results/paper_runs/lisennet_paper_seed42_100ep \
  --epochs 100 --seed 42 --device cuda \
  --validation-limit 200 \
  --test-manifest data/voicebank-demand-16k/test_manifest.json
```

### GTCRN and TF-GridNet

GTCRN now has a launched 100-epoch paper-track run; TF-GridNet does not.

- GTCRN: the paper does not state a VCTK epoch count, segment length or weight
  decay. The run below matches STFT/batch/optimizer/loss, and its 100-epoch
  budget is a repository choice, not a paper value.
- TF-GridNet: the paper has no VoiceBank recipe; matching its 8 ms hop and
  Wav+Mag+MC loss requires a new profile and a new run, not a re-use of the
  existing 25-epoch checkpoint. A batch-1 8 ms forward+backward probe with the
  6-layer local profile (8,381,504 parameters) peaked at **12.72 GiB allocated /
  13.03 GiB reserved** and then OOMed while allocating 2.86 GiB, with the GTCRN
  job holding ~2.1 GiB. Even without the concurrent job, batch 1 leaves no room
  for the paper's training batch on a 16 GB RTX 5060 Ti. The 8 ms path therefore
  needs gradient checkpointing or a larger GPU; it is not a launchable 100-epoch
  run on this host.

### GTCRN paper-track run — 2026-11-11

- Run directory: `results/paper_runs/gtcrn_paper_seed42_100ep`
- Entry point: `python -m scripts.train_gtcrn_paper`
- Command: 4 s segments, batch 4, 100 epochs, seed 42, `train_pool=paper_full`
  (11,572 train utterances, 794 in-sample monitor), paper STFT 512/256 with
  sqrt-Hann, Adam 1e-3, `ReduceLROnPlateau(patience=5)`, paper hybrid loss.
- Started 16:48 with PID 6248; paused after epoch 87 at 22:58, resumed at
  01:11 on 2026-11-12, and completed all 100 epochs at 01:54. The paper-full
  loader processes 2,893 steps/epoch; after warm-up epochs took ~3 minutes.
- The 100-epoch number is a budget-parity choice; it is not a paper epoch count.
- Final `epoch_0100.pt` on the fixed 824-item test: PESQ **2.796208432143174**,
  STOI **0.9405479175402401**, SI-SNR **18.74095403659944**,
  improvement **10.295417911217944 dB**, RTF 0.0196 (single repeat, no warmup).
- Interim `epoch_0087.pt` on the same test: PESQ **2.8509247389811914**,
  STOI **0.9400684192256406**, SI-SNR **18.698905751960375**,
  improvement **10.25336962657888 dB**. Epoch 87 has the higher PESQ; epoch 100
  has slightly higher STOI/SI-SNR. Because `train_pool=paper_full` has no held-out
  split, the paper track declares the last epoch (`selection: last`); choosing
  epoch 87 from test PESQ would be test-set selection and is not used as the
  primary number. Both checkpoints are far above the 25-epoch unified pilot
  (PESQ 2.57246) and close to the pretrained VCTK checkpoint (2.86969), but the
  recipe differs (paper hybrid loss, Adam, full 11,572-utterance train), so the
  gain is not attributable to epoch count alone.

## What “aligned” means here

- For every model, the repository now states which values come from the paper,
  which come from the unified benchmark, and which are unspecified.
- LiSenNet has an executable paper track and a 100-epoch run.
- GTCRN has a complete reference recipe, an executable paper track whose
  synthetic checks pass, and an explicit statement that its epoch budget is not
  paper-defined.
- TF-GridNet has an explicit statement that VoiceBank is an adaptation and that
  the hop/loss differences are structural.

This page should be updated with final LiSenNet numbers when the run completes;
the historical pilot reports remain unchanged.

Independent read-only audits are recorded in
[training audit](agent_training_audit_2026-11-09.md),
[TF-GridNet hop audit](tfgridnet_hop_audit_2026-11-09.md),
[GTCRN paper-track plan](gtcrn_paper_track_plan_2026-11-09.md), and
[Codex review](codex_review_2026-11-09.md). The local `.research/gtcrn.pdf` is an
IEEE Xplore error page, not the paper; the GTCRN values above were read from the
five-page implementation text mirrored at
<https://r.jina.ai/https://www.scribd.com/document/1019076201/GTCRN-ICASSP2404>
and cross-checked against the official `loss.py`. A local text copy is saved at
`.research/gtcrn_scribd.txt` (43,560 bytes); the batch/optimizer/loss/parameter
lines were re-grepped there on 2026-11-09. The publisher PDF should still be
substituted if institutional access becomes available.

### GTCRN code review and fixes — 2026-11-09

An independent read-only review (`docs/agent_gtcrn_code_review_2026-11-09.md`)
found a fatal misalignment in the first GTCRN paper-track draft: noisy and clean
were cropped with two independent random starts. The review also found that
`--resume` could not open a non-empty run directory, that resume did not pin the
manifest, that the scored validation count was not recorded, that zero padding
entered the unmasked loss, and that several config keys were ignored.

Those items are fixed in `paper_gtcrn.py`: one shared crop start (short pairs are
wrap-padded instead of zero-padded), resume-aware run directories with manifest
and `rows_sha256` checks, `val_count`/`validation_scored_count` in the curve and
run record, `source.sha256`, per-file audio SHA-256 verification on first read,
and validation/checkpoint/results-root/n_fft/hop/window config checks. The smoke
suite now has 15 checks and passes 15/15; the 3-step fixture still passes. A
100-epoch real run was launched later (see “GTCRN paper-track run — 2026-11-11”
above).

A second read-only pass (`docs/codex_recheck_2026-11-09.md`) confirmed the crop and
wrap-pad fixes but asked for stronger resume provenance. The track now refuses to
resume when `source.sha256` changed, records and compares `split_sha256`, keeps the
epoch history in `run.json`, checks that the checkpoint epoch matches the last
`learning_curve.csv` row, and rejects `--epochs 0` plus non-paper sample rate,
batch, optimizer, learning rate, weight decay, scheduler and loss fields. These
guards were re-checked with the 15/15 CPU smoke suite and a fresh 3-step fixture;
they do not affect the running LiSenNet job.

## LiSenNet paper run — 110 epochs (2026-11-11)

The initial paper run was extended from 100 to 110 epochs without changing the
recipe (2 s crop, batch 4, AdamW 5e-4, betas 0.8/0.99, StepLR 0.98/epoch,
clip 5.0, PESQ-conditioned GAN loss). The 100-epoch artifacts were archived as
`selection_full_100ep.json`, `test824_full_reselection_100ep.json`,
`learning_curve_100ep.csv` and `run.json.100ep` before the extension.

Full 794-utterance validation selected epoch **107**:
PESQ **2.7048619668790015**, STOI **0.8807522036286425**,
SI-SNR **9.625487268051952**.

The fixed 824-utterance test set scored:
PESQ **3.0456501381489836**, STOI **0.9361581799471856**,
SI-SNR **12.960421203972912**, improvement **4.514885078591418 dB**,
RTF **0.005404134757969054** (36,783 parameters).

For comparison, the 100-epoch run selected epoch 99 with full-validation PESQ
2.6978567997814724 and test PESQ 3.0385946486760114. The 10 extra epochs
therefore added +0.0070 full-validation PESQ and +0.0071 test PESQ. The paper
reports PESQ 3.07 on its 872-utterance test split; this run uses the repository's
824-utterance split and speaker-disjoint protocol, so the numbers are aligned but
not identical.

Automatic shutdown was disabled by operator request after this run; the
finalizer now stops after evaluation and leaves the machine on.

## TF-GridNet memory-policy study — 2026-11-12

The 8 ms, six-layer paper configuration does not fit the 16 GB RTX 5060 Ti in
FP32: it peaks at 12.72 GiB allocated and then requests another 2.86 GiB.
FP16 and activation checkpointing are the only practical route on this card.
The measured Pareto, the cheap-first hierarchical policy, the 200-step and
100-step bit-identical loss checks with a real learning trajectory, the 4-layer
boundary check, and the paper draft are in
[`docs/tfgridnet_memory_policy_bench.md`](tfgridnet_memory_policy_bench.md)
and [`docs/tfgridnet_paper_draft.md`](tfgridnet_paper_draft.md). Both
1,000-step arms completed at PESQ 2.123 with identical losses; this is a
separate systems/efficiency study, not a claim that the repository's 16 ms profile
reproduces the paper's 8 ms configuration.
