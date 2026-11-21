# LiSenNet paper-track training audit — 2026-11-09

> **Status:** Superseded by the 16:03:16 restart. The findings below describe the pre-fix 14:30 run; see `agent_training_reaudit_2026-11-09.md` for the current verdict. The discriminator heads, validation sample, tail batch, provenance and RNG checkpoint issues were addressed; residual risks are listed there.

Read-only audit of the LiSenNet paper-fidelity track against arXiv:2409.13285
(local text `.research/lisennet.txt`) and the pinned upstream snapshot
(`hyyan2k/LiSenNet@df4e481`, `.research/upstream/lisennet_*.py`).

Method and limits:

- No file was modified except this report; the live run directory
  `results/paper_runs/lisennet_paper_seed42_100ep` was only read.
- No training process was started or stopped, no GPU work was issued, nothing was
  installed, no git commit was made.
- Numerical checks used the bundled CPU interpreter
  (`.conda/python.exe`, torch 2.11.0, `CUDA_VISIBLE_DEVICES=-1`):
  parameter counts, forward shapes, discriminator output range, and the
  repository's own upstream-equivalence smoke test
  (`scripts/smoke_lisennet.py` → `PASS: LiSenNet upstream equivalence, waveform
  alignment, normalization and gradients`, `36783 36783` parameters).
- Artifact snapshot: `run.json` status `running`, started
  `2026-11-09T14:30:02+0800`; `learning_curve.csv` held epochs 1–5;
  `selection.json` best = epoch 5 (`val_pesq` 2.364524359703064). Numbers below
  are as of that snapshot and will drift as training continues.

Paper citations are section + local text line of `.research/lisennet.txt`;
everything else is `path:line` in this repository or in `.research/upstream/`.

---

## Confirmed

**1. Crop/pad, then normalize by the cropped noisy std — order, divisor and eps
all match the official loader.**
`speech_denoising/training/paper_lisennet.py:80-90`: pad with `mode="wrap"` when
short (`:81-84`), otherwise `start = rng.randrange(0, length - cut_len + 1)`
(`:86`) and slice both signals (`:87-88`); only then
`scale = float(np.std(noisy) + 1e-8)` (`:89`) with `noisy / scale, clean / scale`
(`:90`). Upstream `.research/upstream/lisennet_data_module.py:43-60` does exactly
this: pad `:53-54`, cut `:56-59`, then `normalize_src_tgt` `:60`, whose body
(`:32-36`) divides **both** signals by `src.std() + 1e-8`. Popstd (ddof=0) on
float32 in both, since upstream loads `sf.read(..., dtype='float32')`
(`lisennet_data_module.py:24-31`) and the repo loader returns float32
(`speech_denoising/datasets/voicebank.py:82-96`).

**2. 2 s / 32,000 samples.** `paper_lisennet.py:44` `CUT_LEN = 32000` at 16 kHz
(`:43`); upstream default `cut_len: int = 16000 * 2`
(`.research/upstream/lisennet_data_module.py:13`). The paper does not state a
segment length; the value is inherited from official code, and that is the
correct provenance.

**3. batch 4, AdamW 5e-4, betas (0.8, 0.99), StepLR 0.98/epoch, clip 5.0.**
`paper_lisennet.py:45-50` and `:290-293` — `AdamW(..., lr=5e-4,
betas=(0.8,0.99), weight_decay=0.0)`, `StepLR(step_size=1, gamma=0.98)` for both
optimizers; clipping at `:382` (D) and `:390` (G) with `CLIP = 5.0`.
Paper: §III-A, `.research/lisennet.txt:165-169` — AdamW, 100 epochs, L2 clip 5.0,
lr 5e-4 decaying ×0.98 each epoch. Batch size 4 and the betas are **not** in the
paper text (grep for `batch` yields only the symbol `B` at `:102`); they come from
the official config recorded at `configs/paper_recipes.yaml` (lisennet block) and
`speech_denoising/models/lisennet/SOURCE.md:24` (config.yaml sha256), which is a
provenance record, not a vendored file — see Discrepancy D4.
Artifact cross-check: `learning_curve.csv` `g_lr` is `4.9e-4` at epoch 1 and
`4.6132e-4` at epoch 4, i.e. `5e-4·0.98^(n-1)` — the schedule in the code is the
schedule the run used.

**4. Generator objective = 0.9·L_mag + 0.1·L_comp + 0.05·L_adv.**
`paper_lisennet.py:53` weights and `:128-143`:
`complex = MSE(re) + MSE(im)` (`:130-131`) equals `E‖Y^c − Ŷ^c‖²` of Eq. (7)
(`.research/lisennet.txt:124-126`); `magnitude = MSE(est_mag, tgt_mag)` (`:132`)
equals Eq. (6); `adversarial = MSE(D(tgt_mag, est_mag)[-1], 1)` (`:133-134`)
equals Eq. (8); total at `:135-137` with λ = 0.9/0.1/0.05 from Eq. (9)
(`.research/lisennet.txt:131-135`). Identical structure upstream
(`.research/upstream/lisennet_model.py:130-141`).
Independent arithmetic check on the run's own numbers (epoch 4 row of
`learning_curve.csv`): `0.1·0.46762 + 0.9·0.05942 + 0.05·0.46784 = 0.12364`,
equal to the logged `train_generator_loss` 0.12364 — the executed loss actually
carries these weights.

**5. Discriminator objective: real pair → 1, fake pair → scaled PESQ.**
`paper_lisennet.py:147-153`: `fake = D(tgt_mag, est_mag.detach())` (`:147`),
`real = D(tgt_mag, tgt_mag)` (`:148`), `MSE(real[-1], 1)` (`:149`), plus
`MSE(fake[-1], pesq_targets)` only when targets exist (`:150-153`). Same as
`.research/upstream/lisennet_model.py:152-166`, including the `(tgt, tgt)` real
pair (`:158-159`) and the detached estimate (`:157`).

**6. PESQ rescaling and failure handling.** `paper_lisennet.py:106-113`
`min(max((pesq − 1)/3.5, 0), 1)`; `:116-125` runs one PESQ per pair and returns
`None` if **any** score is `None` (`:123-124`), so the D step falls back to the
real term alone (`:150`). Upstream `.research/upstream/lisennet_model.py:170-186`
is the same: exception → `-1`, any `-1` → `(None, None)`, else
`(pesq − 1)/3.5` with `np.clip(..., 0, 1)`. Paper §II-F/Eq. (8) and
`.research/lisennet.txt:119` ("scaled to (0, 1)") are satisfied.

**7. Step order D-then-G, with clipping on both.** `paper_lisennet.py:376-391`;
upstream `.research/upstream/lisennet_model.py:40-52`. G's adversarial term sees
the *updated* D in both, which is what the official code does.

**8. Forward shapes and the stack-vs-cat issue.** CPU forward on `(2, 32000)`
returns `tgt/est (2, 32000)`, `tgt_spec/est_spec (2, 126, 257) complex64`,
`tgt_mag/est_mag (2, 126, 257) float32` — 126 frames = `32000/256 + 1`, 257 bins,
consistent with Hann 512 / hop 256 (`network.py:153-164`) and c = 0.3
(`network.py:181-191`, §II-A). `est_mag` is built as
`(x[:,0] + 1e-8)·src_mag + (x[:,1] + 1e-8)·src_mag` (`network.py:252`), byte-for-byte
the official expression at `.research/upstream/…` generator forward `:334`; phase
comes from 2-iteration GLA (`network.py:254`, upstream `:297, :336`, paper Eq. (5)
`.research/lisennet.txt:148` and k = 2 at `:165`). The discriminator consumes the
pair with `torch.stack([clean_mag, enhanced_mag], dim=1)`
(`speech_denoising/models/lisennet/discriminator.py:51`) giving `(B, 2, T, F)`,
which is exactly what `Conv2d(in_channel=2, ...)` at `:29` expects; concatenating
on the frequency axis would instead produce `(B, T, 2F)` and change both shape and
semantics. Upstream uses the same `stack` (`discriminator.py:54`) and the same
3-plane generator input `stack([src_mag, gd/π, ifd/π], dim=1)` (`network.py:246`,
upstream `:328`). Measured: 19 intermediate outputs, final `(2, 1)` inside
`(0, 1)`.

**9. Validation mechanics.** `paper_lisennet.py:161-192`: `model.eval()` (`:163`),
full utterance through the adapter (`:174-177`, no crop), metrics computed per
utterance in a worker pool (`:181-182`), means and coverage returned
(`:188-191`), training mode restored (`:187`). Macro averaging and mode restore
in `speech_denoising/metrics/audio_metrics.py:120, :154, :157-160`. Adapter
normalization uses only the noisy utterance
(`speech_denoising/models/lisennet/model.py:53-60`), matching the official
validation branch `.research/upstream/lisennet_data_module.py:48-50`.

**10. Checkpoint selection by validation PESQ, strict improvement.**
`paper_lisennet.py:441-447`: writes `selection.json` only when
`val["pesq"] > best["pesq"]` (`:441`), recording epoch, pesq, stoi, si_snr,
relative checkpoint path and sha256 (`:442-444`). Artifacts agree:
`validation/epoch_0004.json` has `"count": 200` with `metric_coverage` 200/200,
and `selection.json` at snapshot time pointed at `checkpoints\\epoch_0005.pt`
with a matching sha256.

**11. Evaluation script loads the selected checkpoint and scores the full sets.**
`scripts/evaluate_lisennet_paper.py:33-35` reads `selection.json` `best` and the
checkpoint it names; `:37-41` rebuilds plan and validation view;
`:49-51` loads and switches to eval; `:55` calls `validate(...)` with `limit=None`
→ all 794 validation utterances (the code path is `paper_lisennet.py:165`
`selected = list(indices)`), written to `validation_full.json` (`:65-66`);
`:69-87` optionally scores the 824-utterance test manifest with
`evaluate(...)` (`audio_metrics.py:111` `count = len(dataset)` when `limit is None`).
The training script itself also does a single end-of-run test pass with the
selected checkpoint (`paper_lisennet.py:459-482`, 824 items, `verify_files=True`
at `:461`).

**12. No train/validation/test leakage.** Measured from the manifests:
train manifest 11,572 rows over 28 speakers; test manifest 824 rows =
p232 (393) + p257 (431); intersection of train and test speaker sets is empty.
Validation = p256 (319) + p259 (475) = 794, held out by the sha256 speaker ranking
in `speech_denoising/training/protocol.py:79-90` and enforced disjoint by
`speech_denoising/training/data.py:38-43` (train ∩ validation = ∅, train ∪
validation = parent). Training uses 10,778 rows / 24 speakers
(`run.json` `split_counts`).

**13. No normalization leakage.** The only normalization is per-utterance
(`paper_lisennet.py:89`, `model.py:53-60`); no corpus-level mean/variance is
estimated from, or applied across, utterances.

**14. Generator equivalence to the pinned upstream is proven in-repo.**
`scripts/smoke_lisennet.py` embeds the official `generator.py` / `dpr_layer.py`
(base64+zlib, sha256-pinned to `SOURCE.md:19-20`), asserts state-dict key/shape
equality and numeric equality of `network(src)["est"]` against it, plus the
std-round-trip and gradient checks; it passes today on CPU. Parameter count
36,783 matches paper Table II (`37k`, `.research/lisennet.txt:211`).

---

## Discrepancies

**D1 (real deviation). The discriminator's two Linear layers lost spectral
normalization.** `speech_denoising/models/lisennet/discriminator.py:43` and `:46`
are plain `nn.Linear(ndf*8, ndf*4)` / `nn.Linear(ndf*4, 1)`, while every conv is
wrapped (`:29, :32, :35, :38`). Upstream wraps **all six** weight layers:
`.research/upstream/lisennet_discriminator.py:44` and `:47` are
`norm_f(nn.Linear(...))`. Consequences: (a) the PESQ-regression head is
unconstrained in Lipschitz constant, so its output scale and gradient magnitude
differ from the official D; (b) an official D `state_dict` would not load
(`weight` vs `weight_orig`); (c) the parameter count is unchanged (181,650 both
ways, measured), so a count comparison cannot detect it. This is not recorded in
`docs/paper_alignment.md`.

**D2 (cosmetic). `LearnableSigmoid.slope` shape** is
`torch.ones(in_features, 1)` (`discriminator.py:17`) versus `torch.ones(in_features)`
upstream (`lisennet_discriminator.py:9`). Broadcasting to `(B, 1)` makes the
numerics identical, but the state-dict shape differs.

**D3 (safe superset). PESQ failure detection is wider than official.**
`paper_lisennet.py:111-112` also rejects non-finite scores; upstream
(`lisennet_model.py:170-176`) only maps exceptions to `-1`. A `NaN` PESQ would
poison upstream's D step and is silently downgraded here. Acceptable, but it is a
behavioural difference and the raw PESQ value is no longer logged for the curve.

**D4 (unverifiable from vendored files). batch size 4 and betas (0.8, 0.99).**
The paper states neither (`.research/lisennet.txt:165-169` covers only optimizer,
epochs, clip, lr, decay). The values rest on `configs/paper_recipes.yaml` and the
sha256 in `SOURCE.md:24`; the official `config.yaml` itself is not vendored in
`.research/upstream/`, so the audit can confirm the *record* but not re-derive the
value from an inspectable upstream file.

**D5 (documentation shaves the mechanism). The 200-utterance validation subset is
a truncated stride, not a uniform spread.** `paper_lisennet.py:168-169`:
`stride = 794 // 200 = 3` → 265 candidates, then `[:200]` discards 65 of them.
Computed from the manifest: selected positions are 0…597, so **196 of 794
utterances (24.7 %, positions 598–793) are never scored**, and the last scored item
is `p259_282`. Composition: 107 of 319 p256 utterances (33.5 %) and 93 of 475 p259
utterances (19.6 %), i.e. the subset is **53.5 % p256 versus 40.2 % in the full
split**, and p259 is sampled only from its first 282 utterances.
`docs/paper_alignment.md:86-88` describes a "fixed 200-utterance subset" without
disclosing the truncation or the skew.

**D6 (paper-fidelity, by design). Validation and model selection do not exist in
the paper.** §III-A (`.research/lisennet.txt:143-146`) defines only a train and a
test partition; the paper reports test numbers for a fixed 100-epoch schedule, not
a PESQ-selected checkpoint. The repository adds a held-out split, a 200-item
selection metric and best-checkpoint selection (`paper_lisennet.py:441-447`).
This is a deliberate, documented deviation (`docs/paper_alignment.md:89-91`), but
any claim of the form "the paper's recipe" must not include the selection step.
It also contradicts the unified protocol the run loads
(`configs/training_protocol.yaml:54-60` declares `selection_metric: loss`,
`selection_direction: minimize`), which `paper_lisennet.py:254` nevertheless
validates at startup.

**D7 (provenance). `run.json` claims a commit that does not contain the code.**
`paper_lisennet.py:324` writes `"git": {"commit": git_commit(), "dirty": None}` —
`dirty` is a hardcoded `None`, never computed, and `git_commit()` (`:64-72`) reads
`.git/HEAD` directly. At audit time `git status` shows
`speech_denoising/training/paper_lisennet.py`,
`speech_denoising/models/lisennet/discriminator.py`,
`scripts/train_lisennet_paper.py`, `scripts/evaluate_lisennet_paper.py`,
`configs/paper_recipes.yaml` and `docs/paper_alignment.md` as **untracked**, so
`f8e4489ad622d38650eaf5d1cf68e7092c864788` is not a reproducible pointer to the
audited training code.

**D8 (provenance). `source.sha256` does not cover the evaluation path.** It lists
8 files (training module, discriminator, network, model, voicebank, data, train
entry, manifest) and `sha256sum -c` passes for all of them, but
`scripts/evaluate_lisennet_paper.py`, `speech_denoising/metrics/audio_metrics.py`,
`speech_denoising/training/protocol.py`, `configs/training_protocol.yaml` and
`configs/voicebank.json` are absent — and the metrics module is what turns
checkpoints into the reported `val_pesq`. A hash set that omits the scorer cannot
support a claim of reproducibility for the scores.

**D9 (provenance). Audio bytes were not verified in this run.**
`paper_lisennet.py:278-281` constructs both datasets with `verify_files=False`, and
`run.json`'s recorded command has no `--verify-audio` (`:494-495`). Manifest
structure is still enforced (`voicebank.py:61-71`: schema, split, 16 kHz,
`rows_sha256`, 11,572 rows), so rows and provenance are checked but sample bytes
are not.

**D10 (small reporting bug). The CSV's `g_lr`/`d_lr` are post-step values**
`scheduler.step()` runs at `paper_lisennet.py:406-407` before the row is built at
`:411-419`, so epoch 1 is reported as `4.9e-4` although it trained at `5e-4`.
Anyone reconstructing the schedule from the curve will be one epoch off.

**D11 (checkpoint cadence).** A checkpoint is written for **every** epoch
(`paper_lisennet.py:428-429`), which matches upstream's per-validation-epoch
saving (`.research/upstream/lisennet_train.py:26-33`) and is what makes
post-hoc re-selection possible. Not a deviation, recorded because the run retains
100 full checkpoints (~278 MB) rather than only the selected one.

---

## Risks

**R1 (selection noise).** The selection metric is PESQ on 200 utterances drawn
from only **two** speakers, with the composition skew of D5; single-utterance PESQ
has a standard error around 0.05–0.1, so differences of 0.03 between adjacent
epochs (2.2741 → 2.3114 → 2.3645 in `learning_curve.csv`) are inside the noise
band unless correlated. Choosing the arg-max over 100 epochs on a 200-item subset
is a winner's-curse setup; the reported "best epoch" is an optimistic estimate of
that epoch's full-set performance.

**R2 (nondeterminism).** The paper track never sets
`torch.use_deterministic_algorithms` or `cudnn.deterministic`/`benchmark` (compare
`speech_denoising/training/fixture.py:62` and `real_runner.py:95-102`, which both
expose deterministic switches). CUDA kernels are therefore free to vary, so
"seed 42" reproduces the recipe, not the bits. The per-epoch RNG is
`random.Random(f"lisennet_paper:{seed}:{epoch}")` (`paper_lisennet.py:361`), which
is stable across processes (str seeds hash via sha512, not `hash()`), so the
utterance order and crop offsets are reproducible even though kernel math is not.

**R3 (resume is not trajectory-identical).** `checkpoint_payload`
(`paper_lisennet.py:197-210`) stores model, discriminator, both optimizers and both
schedulers — but no RNG states. On `--resume` the globals are re-seeded
(`:272-276`) and the loop restarts at `start_epoch = epoch + 1` (`:339`), so the
discriminator's `Dropout(0.3)` masks (`discriminator.py:44`) follow a different
stream than an uninterrupted run would have. Data order and crops *are* restored,
because the order RNG is keyed by epoch (`:361-363`).

**R4 (test peeking).** The training script evaluates the 824-item test set once at
the end with the selected checkpoint (`paper_lisennet.py:459-482`), which respects
"test after selection". But `scripts/evaluate_lisennet_paper.py:69-87` can be run
repeatedly, each time writing `test824_full.json` (`:85`) — including when
`--test-limit` narrows the set (`:76`), so a partial run produces a file whose name
claims the full 824 and whose scope field (`:79`) does not mention the limit.

**R5 (unverified inputs in the evaluation path).** `evaluate_lisennet_paper.py:33-35`
trusts `selection.json`, and `:49-50` loads whatever bytes are at that path; it
never compares the file hash against `best["checkpoint_sha256"]` (`:61` re-emits
the recorded hash, so a swapped checkpoint would be reported under the old
digest). It also does not check the run's `plan_sha256`/manifest sha256 from
`run.json` against the current manifest and
`configs/training_protocol.yaml` (`:37-38`), so an edit to the split config
silently changes which 794 utterances are "validation".

**R6 (Windows path in a portable field).** `selection.json` stores
`"checkpoints\\epoch_0005.pt"` because `:443` uses `str(checkpoint.relative_to(run_dir))`
on Windows. `evaluate_lisennet_paper.py:35` re-joins it, which works here and
breaks on POSIX.

**R7 (objective/metric mismatch).** The discriminator is trained on PESQ of the
2 s, noisy-std-normalized crop (`paper_lisennet.py:377` passes the batch-sized
`clean` and `results["est"]`, both 32,000 samples), while selection scores
full-utterance PESQ (`:174-177`). The two regimes have different score
distributions, so a D that regresses crop PESQ well need not order full-utterance
quality correctly; the adversarial gradient is therefore only loosely aligned with
the selection metric.

**R8 (smoke path can masquerade as a finished run).** With `--max-steps`,
`paper_lisennet.py:449-457` still writes `status: "complete"` plus `finished_at`,
and `:459-482` still evaluates the test set and writes `test824.json`. A smoke run
and a 100-epoch run are distinguishable only by the curve length and `best` epoch.

**R9 (cost/fragility of per-batch PESQ).** `batch_pesq` calls
`joblib.Parallel(n_jobs=min(4, B), prefer="processes")` once per training step
(`:120`), i.e. 2,694 times per epoch. That is why epochs cost 285–313 s
(`learning_curve.csv` `epoch_seconds`) and it means any worker failure aborts
mid-epoch, before the epoch checkpoint exists (`:428-429`).

**R10 (I/O).** `VoiceBankDataset.__getitem__` decodes both WAVs on every access
(`voicebank.py:82-96`) and nothing is cached; the 2 s crop is taken *after* full
decode (`paper_lisennet.py:96-99`). Correct, but it re-reads the corpus ~100 times
plus 200 validation files per epoch.

**R11 (two dataset objects).** Train and validation views are two separate
`VoiceBankDataset` instances over the same manifest
(`paper_lisennet.py:278-281`); harmless, but each holds its own metadata and file
handles, and validation re-decodes rather than reusing the training cache.

**R12 (duplicate curve rows on resume).** The CSV row is written at `:420` before
the checkpoint at `:429`; a crash between them makes `--resume` replay that epoch
and append a second row for it (the file is opened in append mode at `:345`).

**R13 (resume rewrites the run header).** `run.json` is written at the top of
every invocation (`:329`) with a fresh `started_at` (`:327`) and `status:
"running"` (`:317`); a resumed run loses the original start time and the previous
command line, while `best` survives only because it is re-read from
`selection.json` (`:356-357`).

---

## Recommended fixes

1. **Restore spectral normalization on the D head.**
   `discriminator.py:43, :46` → `spectral_norm(nn.Linear(ndf*8, ndf*4))` /
   `spectral_norm(nn.Linear(ndf*4, 1))`, and set the slope shape to
   `torch.ones(in_features)` (`:17`) to match
   `.research/upstream/lisennet_discriminator.py:9, :44, :47`. Do it in a **new**
   run: switching mid-training invalidates the checkpoint series and the
   optimizer/shape history. Until then, record D1 in
   `docs/paper_alignment.md` as a known deviation.
2. **Replace the truncated stride with an order-independent subset** at
   `paper_lisennet.py:168-169`: rank validation ids by
   `sha256("lisennet-val-200-v1:{seed}:{utterance_id}")`, take the first 200, and
   log the resulting id list (plus its own sha256) into `selection.json` and
   `validation/epoch_*.json`. Alternatively score all 794 every *k*-th epoch and
   select on the full number. Either way the subset becomes speaker-balanced and
   auditable.
3. **Log the LR that was actually used** (read `param_groups[0]["lr"]` before
   `scheduler.step()`, or move the step after the row) so `learning_curve.csv`
   matches the epoch it describes.
4. **Record real provenance.** Compute `dirty` from `git status --porcelain`
   instead of `None` (`:324`), and make the paper-track entry point refuse to
   start (or at least warn loudly) when the files it imports are untracked or the
   working tree is dirty.
5. **Widen `source.sha256` to the whole result path** — add
   `scripts/evaluate_lisennet_paper.py`,
   `speech_denoising/metrics/audio_metrics.py`,
   `speech_denoising/training/protocol.py`, `configs/training_protocol.yaml`,
   `configs/voicebank.json` — and have the evaluation script verify
   `checkpoint_sha256`, `plan_sha256` and the manifest sha256 from `run.json`
   before it writes `validation_full.json` / `test824*.json`.
6. **Make resume trajectory-faithful**: add `torch`, `numpy`, `random` and CUDA
   RNG states to `checkpoint_payload` (`:197-210`) and restore them after
   `load_checkpoint`; keep the first `started_at` and append a `resume_log` entry
   instead of overwriting `run.json`.
7. **Order the CSV row after the checkpoint** (or key the curve by epoch and
   rewrite idempotently) to remove the duplicate-row window (R12).
8. **Name test reports by scope**: write `test824_full.json` only when
   `--test-limit` is `None`, otherwise `test{N}.json`, and always store `limit`
   in the report (`evaluate_lisennet_paper.py:76, :85`).
9. **Verify audio for the production launch** — pass `--verify-audio` (or default
   `verify_files=True` once the manifest is final) so the hashes in `run.json`
   actually cover the samples, as they already do for the test set
   (`paper_lisennet.py:461`).
10. **Store checkpoint paths portably** (`checkpoint.relative_to(run_dir).as_posix()`
    at `:443`) so `selection.json` survives a move to another OS.
11. **Separate the smoke path from the result path**: with `--max-steps`, write
    `status: "smoke"` and skip the test block, so a partial run cannot be mistaken
    for a completed 100-epoch result (R8).
12. **Document the selection step as an addition, not the paper.** In
    `docs/paper_alignment.md`, state explicitly that (a) the paper has no
    validation split or checkpoint selection, (b) the 200-item subset is a
    repository choice with its exact construction, and (c) D-head spectral
    normalization differs from the pinned official code until fix 1 lands.

---

## Verdict

The training recipe (crop→normalize order and divisor, 2 s/32,000, batch 4,
AdamW 5e-4 with (0.8, 0.99), StepLR 0.98, clip 5.0, 0.9/0.1/0.05 loss, PESQ
scaling and `None` fallback, D-then-G order, stack-based channel layout) matches
the paper and the pinned official code, and the generator is proven numerically
equivalent to it. The open issues are one real architectural deviation
(unnormalized D head), a biased and under-documented 200-item selection subset, a
provenance chain that does not cover either the code commit or the scorer, and
resume/RNG gaps that prevent bitwise reproduction. No train/validation/test
leakage was found: the split is speaker-disjoint and normalization is strictly
per-utterance.
