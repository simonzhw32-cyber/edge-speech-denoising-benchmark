# LiSenNet post-fix re-audit — 2026-11-09

> **Follow-up (implemented):** `load_checkpoint` now uses `weights_only=False` for local run artifacts; `capture_rng_state` stores the NumPy state as built-ins so future checkpoints remain `weights_only=True`-loadable; the finalize watcher accepts the trainer status `complete`. The two post-launch source changes are recorded in `source.sha256.known_mismatch`.

Read-only re-check of `results/paper_runs/lisennet_paper_seed42_100ep` (launched
16:03:16) after the D1/D5/D7/D8/D10 and tail-batch fixes; only this file written.
Snapshot: epochs 1–4 complete, epoch 5 in flight.

## Confirmed

- **D1** `discriminator.py:43,:46` spectral_norm on both head Linears (`weight_orig`
  at layers 0,3,6,9,14,17); slope `torch.ones(in_features)` → `(1,)` (`:17`); 181,650 params.
- **D5** `paper_lisennet.py:179-192` sha256-ranks the whole split (used at `:312`,
  `:199`); 200 unique ids ⊆ 794, 116 p259 / 84 p256 = 58/42 vs 59.8/40.2 — balanced.
- **Tail batch** `:453` `range(0, len(order), BATCH_SIZE)` → 2695 steps/epoch;
  curve `global_step` 2695/5390/8085/10780 (10,778 = 2694·4+2).
- **D10** LR read pre-step (`:490-493`): 5e-4, 4.9e-4, 4.802e-4, 4.70596e-4 = 5e-4·0.98^(n−1).
- **D7** real `git_dirty()` (`:76-90`) → `"dirty": true` (`:386`); `started_at`
  survives resume (`:371-375`, `:389`); `plan_sha256`/manifest/`split_sha256` re-compute exactly.
- **D8** `source.sha256` now spans 15 files incl. the eval script, metrics,
  protocol and configs; `sha256sum -c` → 14 OK, 1 FAILED = `reselect_lisennet_paper.py`
  70d90f57…, exactly the `source.sha256.known_mismatch` entry.
- **RNG plumbing** `:225-243`, `:262`, `:511-513`, `:422`; `epoch_0004.pt` holds
  cuda/numpy/python/torch states.
- **Guards** `stopped_early` (`:536-542`, test gated at `:547`), `.as_posix()`
  (`:530`), strict `>` (`:528`) keeps epoch 3 (2.25771 > epoch 4's 2.25323).
- **Provenance** `validation_sample.json` == `run.json` (200, `9046899c…`,
  recomputed); `validation/epoch_0004.json` repeats the hash.
- **No leakage** train 10,778 ∩ val 794 = 0 (`data.py:42-43`); test 824 =
  p232+p257, disjoint from both train and val speakers.
- **Eval hardened** `evaluate_lisennet_paper.py:37-42` verifies the checkpoint
  sha, `:46-51` the plan digest, `:99` writes `test_limit_{n}.json`.

## New discrepancies

- **N1 blocking, introduced by the RNG fix — no checkpoint can be loaded.**
  `load_checkpoint` uses `weights_only=True` (`paper_lisennet.py:275`) while
  `rng_state["numpy"]` is `np.random.get_state()` (`:228`), whose element 1 is a
  uint32 `numpy.ndarray` (625,). Reproduced on `checkpoints/epoch_0004.pt`:
  `UnpicklingError` — `WeightsUnpickler error: Unsupported global: GLOBAL
  numpy._core.multiarray._reconstruct` (loads with `weights_only=False`); no
  `add_safe_globals` in repo code. Dead paths: test pass `:552`, resume `:418`,
  `evaluate_lisennet_paper.py:62`, `reselect_lisennet_paper.py:101`. Since
  `run.json` turns `complete` at `:542-545` *before* the test block `:547`, the
  run will look finished while having crashed without `test824.json`.
  → Fix: nothing samples `np.random` (seeded `:330`, crops use `random.Random`
  at `:444`), so drop the `"numpy"` key, or store
  `{"name","keys": state[1].tolist(),"pos","has_gauss","cached_gaussian"}` and
  rebuild with `np.random.set_state`; existing checkpoints recover via
  `weights_only=False` on local files.
- **N2 automation** `.tmp/finalize_lisennet.sh:13` tests `"completed"` but `:542`
  writes `"complete"` → watcher logs "not evaluating" and exits 0.
- **N3** two watcher instances (`finalize.log`: 16:44:56, 16:46:17) would race
  `reselect`; it also polls PID 125168, so a restart orphans it.
- **N4** `known_mismatch` is documentation only (`docs/paper_alignment.md:186`);
  `sha256sum -c` still exits non-zero.

## Residual risks

- `git.commit` `f8e4489` does not contain the paper-track files (all untracked);
  `dirty: true` flags it, but no commit↔code mapping exists.
- Curve row at `:506` precedes the checkpoint at `:511` → a crash duplicates a row.
- A killed run keeps `status: "running"` (only `:542` changes it).
- Epoch 458/536/629/621 s → ≈15–17 h for 100 epochs (per-step joblib PESQ, `:138`).
- `reselect:108` emits a backslash path; `evaluate:73` an absolute Windows path.
- `--verify-audio` stays opt-in (`:339-343`), so sample bytes remain unverified.

## Verdict

All requested fixes and the tail-batch change are present and independently
verified; digests, subset balance and leakage checks hold. But the checkpoint
payload now breaks `weights_only=True`, killing the end-of-run test pass, resume,
and both evaluation scripts — and the watcher looks for `"completed"` while the
trainer writes `"complete"`. Unless N1/N2 are patched, this run reaches epoch 100
with no test number and no fallback automation.
