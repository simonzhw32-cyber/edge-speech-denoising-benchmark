# Documentation audit — 2026-11-09

> **Status:** D1–D5 and the stale/ambiguous items were fixed after this audit; D6–D9 have status banners. This file is the historical audit record, not the current documentation state.

Read-only audit of the human-facing docs against the current tree. Read:
`README.md`; `docs/paper_alignment.md`, `results.md`, `history.md`,
`project_requirements.md`, `real_training.md`, `benchmark_protocol.md`,
`gtcrn_paper_track_plan_2026-11-09.md`, `agent_gtcrn_code_review_2026-11-09.md`,
`agent_training_audit_2026-11-09.md`, `tfgridnet_hop_audit_2026-11-09.md`;
`configs/paper_recipes.yaml`, `configs/gtcrn_paper.yaml`; relevant parts of
`paper_lisennet.py`, `paper_gtcrn.py`; and the LiSenNet run artifacts.
No training/GPU/pip/git action; the only write is this file.

## Confirmed

- Pretrained numbers match the JSON: GTCRN 2.86969 / 0.94029 / 18.79606 /
  10.35052, 48,245 all / 23,669 trainable; TF-GridNet DNS 2.77327 / 0.94310 /
  19.40151 / 10.95598, 2,552,790 / 2,552,790; both 824/824. Matches
  `README.md:48-49`, `docs/results.md:14-16`, `docs/project_requirements.md:50`.
- 25-epoch rows (GTCRN 2.57246/24, LiSenNet 2.57229/21, TF-GridNet 2.79501/23)
  agree across `README.md:62-66`, `docs/results.md:33-37` and pilot reports.
- LiSenNet run matches `docs/paper_alignment.md:164-183`: `run.json` started
  `2026-11-09T16:03:16+0800`, 10,778/794, 100 epochs, batch 4, AdamW 5e-4,
  betas (0.8, 0.99), StepLR 0.98, clip 5.0, loss 0.9/0.1/0.05;
  `learning_curve.csv` has epochs 1–4 with PESQ 2.10186/2.18765/2.25771/
  2.25323; `selection.json` best = epoch 3 (2.25771).
- Current code carries the claimed fixes: wrap-padding, retained tail batch,
  `sha256_ranked_v2` validation sample, `spectral_norm` on both D heads
  (`discriminator.py:43,46`), `git_dirty()` (`paper_lisennet.py:386`), POSIX
  checkpoint path (`:530`).
- GTCRN track matches its description: `epochs: null`, `segment_seconds: null`,
  `train_pool: paper_full` (11,572, 794 in-sample monitor), selection `last`;
  `paper_gtcrn.py` shares one crop start, wrap-pads shorts, enforces
  512/256/sqrt-Hann; smoke suite prints 15 checks. No real run exists.
- TF-GridNet hop facts are correct: 512/256 (16 ms) vs paper 8 ms, params
  8,381,504 either hop; `docs/paper_alignment.md:133,147-157` records the
  Hann/sqrt-Hann deviation.
- No doc claims a GTCRN VCTK epoch budget or paper-mandated 100 epochs for
  GTCRN/TF-GridNet; the opposite is stated at `README.md:112-116,208-209`,
  `docs/paper_alignment.md:47-50,66,143-146,200-207`, `docs/project_requirements.md:13-17`.

## Discrepancies

**D1 — `README.md:109-111`: unqualified "paper-faithful".**
Current: "A paper-faithful run (2 s crops, ... PESQ-conditioned GAN loss) is
running ...". Why wrong: `docs/paper_alignment.md:93-126` records deviations —
824-item test instead of 872, a 200-utterance PESQ-selected subset, no paper
validation split, wrap-padded shorts, and non-scale-invariant validation PESQ.
Suggested: "A paper-derived run ... with the documented deviations (824-item
speaker-disjoint test, 200-utterance selection subset, wrap-padded shorts,
non-scale-invariant validation PESQ); see `docs/paper_alignment.md`".

**D2 — `docs/history.md:109`: "Fourteen synthetic CPU checks pass".**
Why wrong: the suite has 15 (`smoke_gtcrn_paper.py` prints `len(CHECKS)`;
`README.md:114`, `docs/real_training.md:178`, `docs/paper_alignment.md:57-58,254`
and the code-review addendum `:120` say 15/15; plan `:667` notes 14 planned plus
a 15th aligned-crop check). Suggested: "Fifteen synthetic CPU checks pass (15/15)".

**D3 — `docs/project_requirements.md:17`: "paper-track GTCRN/TF-GridNet recipes remain open".**
Why wrong: the GTCRN track is implemented and passes 15 checks
(`README.md:113-116`, `docs/paper_alignment.md:57-63`); only a real GTCRN run is
open, and only TF-GridNet lacks a track. Suggested: "GTCRN's paper track is
implemented with no real run (no paper epoch budget); TF-GridNet has no track."

**D4 — `docs/project_requirements.md:145` (2026-10-08 section): "TF-GridNet real training and a complete three-model comparison remain pending".**
Why wrong: the same file's current update (`:10-12`) says all three 25-epoch
pilots are complete. Suggested: prefix "As of 2026-10-08" and add "superseded by
the 2026-11-09 update", or drop the paragraph.

**D5 — `docs/project_requirements.md:91-101`: next step contradicts line 91.**
Current: "GTCRN already completed steps 2–5 ... for a single seed and a 25-epoch
pilot", then step 3 "Run **one real GTCRN epoch**". Why wrong: the pilot,
validation and 824-item test are done; this is the old integration step.
Suggested: "For the GTCRN paper track, first obtain an operator epoch/segment
budget; the unified 25-epoch pilot is already complete."

**D6 — `docs/agent_training_audit_2026-11-09.md:19-22,406-417`: pre-fix state presented as current.**
Current: snapshot "started 14:30, epochs 1–5, best epoch 5 (2.3645)"; verdict
"one real architectural deviation (unnormalized D head), a biased ... 200-item
selection subset, a provenance chain that does not cover ... the scorer". Why
wrong: the run restarted at 16:03 (`paper_alignment.md:174-183`), D heads are
spectral-normalized (`discriminator.py:43,46`), the subset is `sha256_ranked_v2`,
`source.sha256` covers 15 files including the evaluator, and the current curve
has 4 epochs/best 3 at 458–629 s each (audit says 285–313 s). Suggested: add a
"superseded by the 16:03 restart" banner and separate pre-fix findings from
still-open risks R1/R2/R3/R5/R7 in the Verdict.

**D7 — `docs/tfgridnet_hop_audit_2026-11-09.md:95-96,184-193,279,289-292,313-315`: stale citations, "undocumented sqrt-Hann".**
Current: cites `paper_alignment.md:99` and `:113-116` for text now at `:133` and
`:147-153`; says sqrt-Hann is "**not documented** anywhere"; its "Safe next step"
asks to fix both doc inaccuracies. Why wrong:
`docs/paper_alignment.md:155-157` documents sqrt-Hann, and `:147-150` already
says parameter shapes do not change (only activation shapes/MACs). Suggested:
mark §3/§4 superseded, update cites to `:133,147-157`, drop the doc-fix item.

**D8 — `docs/agent_gtcrn_code_review_2026-11-09.md:22,74,101`: body says the track is unusable.**
Current: "**总判断：当前 track 不能用于真实训练**", B5 zero-padding, "**不能。**".
Why wrong: addendum `:113-121` records B1–B7 fixed (shared crop start,
wrap-padding, resume checks, val count, `source.sha256`, enforced config) and
15/15 checks passing. Suggested: status banner under the title: "Findings below
are pre-fix; see Addendum; track implemented, 15/15, no real run."

**D9 — `docs/gtcrn_paper_track_plan_2026-11-09.md:3-4,645-650`: plan status and next step.**
Current: "read-only plan ... The only write is this file"; "Implement steps 1–2
... only ... Do not create a run directory". Why wrong: config, module, trainer,
smoke and fixture all exist and pass 15 checks. Suggested: "Implemented
2026-11-09; historical plan", and retitle §Safe next step as superseded.

## Stale or ambiguous

- `docs/real_training.md:146` repeats the 2026-10-08 "TF-GridNet ... remain
  pending" sentence in a dated section; the next section corrects it, but there
  is no explicit "superseded" marker (unlike `docs/history.md:3-5`).
- `docs/agent_training_audit_2026-11-09.md:205,213` cite
  `paper_alignment.md:86-88,89-91`; those lines now hold the LiSenNet table and
  100-epoch target, not the subset/deviation text.
- `docs/paper_alignment.md:200-203` "No paper-track run has been launched for
  either model" is scoped by its heading to GTCRN/TF-GridNet but can be misread
  as covering LiSenNet. Suggest "For GTCRN and TF-GridNet, no run ...".
- `docs/agent_gtcrn_code_review_2026-11-09.md:46` "203 tensors / 23,669 params"
  is trainable-only, while `README.md:48` reports 48,245 all; both correct, but
  keep the all/trainable label adjacent.

## Link check

- 77 relative Markdown links across the 11 audited docs: **0 broken** (every
  `](...)` target resolved; HTTP links not fetched).
- 61 unique backticked repo paths: one absent — `README.md:76`
  `results/training_runs/gtcrn_random_seed42/checkpoints/epoch_0024.pt` (no
  `results/training_runs/` in this checkout; generated/historical artifact).
  `configs/tfgridnet_paper_stft.yaml` (`hop_audit:294`) is explicitly proposed.
- All audit/plan files named in `docs/paper_alignment.md:222-233` exist, as do
  the pilot reports and benchmark JSON/environment files.

## Verdict

The numeric spine is sound: pretrained, pilot and LiSenNet run numbers match the
JSON/CSV artifacts, code matches the current alignment page, and no doc claims a
GTCRN/TF-GridNet paper 100-epoch budget. The remaining problems are
documentation state, not measurements: one unqualified "paper-faithful" claim
(D1), one wrong check count (D2), three stale requirements passages (D3–D5), and
four audit/plan documents whose pre-fix findings or obsolete next steps read as
current (D6–D9). Add date/status banners and fix D1–D2 before citing externally;
no result needs recomputing.
