# TF-GridNet STFT hop audit (2026-11-09)

> **Status:** Audit performed before the 2026-11-09 documentation corrections. Its factual measurements stand; citations to `paper_alignment.md` line numbers and the claim that sqrt-Hann was undocumented are superseded by the current text at `docs/paper_alignment.md:133,147-157`.

Status: **read-only audit**. No training was started, no GPU was used, no dependency was
installed, no existing file was modified. The only write is this file.

Scope: quantify the frame-rate (hop) deviation between the paper's TF-GridNet STFT recipe
and the repository's local 16 kHz profile, and decide whether it can be corrected safely.

Method: source reading with exact line citations; CPU-only `torch` probes from the bundled
interpreter (`.conda/python.exe`, torch 2.11.0, `CUDA_VISIBLE_DEVICES=""`) for parameter
counting and for the STFT frame-count check; arithmetic for MACs and activation sizes.
The GPU numbers quoted below are **taken from existing reports**, not re-measured here.

---

## 1. What the repository actually runs at 16 kHz

| Quantity | Repository (current) | Paper (§G, 16 kHz) |
|---|---|---|
| Window | 32 ms (`n_fft=512`) | 32 ms |
| Hop / stride | **16 ms (`stride=256`)** | **8 ms (`hop=128`)** |
| Overlap | 50 % | 75 % |
| Window shape | plain Hann | square-root Hann |
| Frequency bins | 257 | 257 |
| Attention embedding E | 2 | 2 |

Evidence for the repository side:

- `configs/tfgridnet.yaml:5-6` — `n_fft: 512`, `stride: 256`; `configs/tfgridnet.yaml:20` —
  `window: hann`; `configs/tfgridnet.yaml:22` — `weights: random`.
- `configs/tfgridnet_dns.yaml:6-7` — the DNS pretrained profile uses the same
  `n_fft: 512` / `stride: 256`.
- `speech_denoising/models/tfgridnet/profiles.py:9` — `DNS_MODEL_PARAMS` repeats
  `"n_fft": 512, "stride": 256`.
- `speech_denoising/models/tfgridnet/model.py:20-23` — constructor defaults
  `n_fft=512, stride=256`; `model.py:37-42` stores them in `self.config`.
- `speech_denoising/models/tfgridnet/network.py:93` — `n_freqs = n_fft // 2 + 1` → **257**,
  a function of `n_fft` only, never of `stride`.
- `speech_denoising/models/tfgridnet/spectral.py:31-35` — `torch.stft(..., n_fft=512,
  win_length=512, hop_length=self.hop_length, center=True, onesided=True)`.
- `speech_denoising/models/tfgridnet/spectral.py:27` — `torch.hann_window` (**not** `.sqrt()`).
  Contrast `speech_denoising/models/gtcrn/model.py:18`, which does use
  `torch.hann_window(...).sqrt()`. The sqrt-Hann deviation of TF-GridNet is currently
  **undocumented** anywhere in `docs/`.
- Run identity: `training_reports/tfgridnet_random_seed42_accum4_25ep/run.json:24` and `:64`
  (`"profile": "local_6layer"`), `:48` (`"config": "configs/tfgridnet.yaml"`),
  `:59-61` (`"n_fft": 512`, `"n_layers": 6`, `"stride": 256`).

Paper side, `.research/tfgridnet.txt:707-713` (two-column PDF text; column interleaving
does not affect the sentence):

> "In default, for STFT, the window length is 32 ms and hop length 8 ms, and the
> square-root Hann window is used as the analysis window. In this case, for 16 kHz sampling
> rate, a 512-point discrete Fourier transform (DFT) is applied to extract 257-dimensional
> complex STFT spectra at each frame ... E (see its definition in Table I) is set to 4 for
> 8 kHz and to 2 for 16 kHz."

So the paper fixes three coupled values at 16 kHz: 32 ms window, 8 ms hop, sqrt-Hann,
512-point DFT, 257 bins, E = 2. The repository reproduces the window length, the DFT size,
the bin count and (by derivation) E = 2, and deviates on the hop and on the window shape.

### Frame counts and tensor shapes at 4 s (64,000 samples)

Verified on CPU with the exact STFT call used by `spectral.py`:

| | stride 256 | stride 128 |
|---|---:|---:|
| `torch.stft` output `[B, F, T]` | `[1, 257, 251]` | `[1, 257, 501]` |
| Frames `T` | **251** | **501** |
| Bins `F` | 257 | 257 |
| After `network.py:156-158` | `[1, 2, 251, 257]` | `[1, 2, 501, 257]` |
| After `network.py:161` (`conv` 2→48) | `[1, 48, 251, 257]` | `[1, 48, 501, 257]` |
| Per-block I/O `[B,C,T,F]` | `[1, 48, 251, 257]` | `[1, 48, 501, 257]` |
| Attention matrix `[B·H, T, T]` | `4 × 251 × 251` | `4 × 501 × 501` |
| After `network.py:166-168` | `[1, 1, 2, 251, 257]` | `[1, 1, 2, 501, 257]` |

`T` doubles; **`F` never changes**; the channel dimensions never change.

### Parameters and MACs

Parameter counts, measured by instantiating both variants on CPU:

| | stride 256 | stride 128 |
|---|---:|---:|
| Total parameters | **8,381,504** | **8,381,504** |
| Per block (`network.py:108-122`) | 1,396,605 | 1,396,605 |

Breakdown of one block: `intra_rnn` 592,896 + `inter_rnn` 592,896 + the two
`ConvTranspose1d` 73,776 each + attention convs/projection 63,069. Front-end `conv` 1,008,
`deconv` 866.

**The parameter count is identical for both hops**, because every layer is either
channel-shaped (1×1 / 3×3 convs, LSTM weights `C·emb_ks → H`) or frequency-shaped
(`LayerNormalization4DCF` gamma is `[1, E, 1, n_freqs]`, `network.py:388`). No parameter
depends on `T`. This contradicts a literal reading of
`docs/paper_alignment.md:113-116` ("changes every temporal convolution/attention shape"):
the *activation* shapes change, the *parameter* shapes do not.

MACs (1 MAC = 1 multiply + accumulate; B = 1; C = 48; emb_ks = 4; H = 192; 4 heads; E = 2):

- intra BLSTM: `T·F · 4·(192·192 + 192·192) · 2 = T·F · 589,824`
- inter BLSTM: same, `T·F · 589,824`
- `intra_linear` / `inter_linear` (`ConvTranspose1d(384→48, k=4)`): `T·F · 73,728` each
- attention 1×1 convs: `T·F · 5,376`
- attention matmuls: `4,112 · T²`

Per block: `T·F · 1,332,480 + 4,112·T²`. Total (6 blocks + front/back end):

```
MACs ≈ 7,996,608 · T·F  +  24,672 · T²
```

| | stride 256 (T=251) | stride 128 (T=501) | ratio |
|---|---:|---:|---:|
| MACs per 4 s segment | 5.17 × 10¹¹ | 1.03 × 10¹² | **2.00×** |
| GMAC/s | ≈ 129 | ≈ 259 | 2.00× |
| Attention share of MACs | 0.3 % | 0.6 % | 4× (negligible absolute) |

The paper's own ablation (`.research/tfgridnet.txt:1056-1060`) confirms this scaling
direction: halving the window (halving `F`) "reduc[es] GMAC/s by almost half", i.e. cost is
≈ linear in `T·F`. Doubling `T` therefore ≈ doubles cost, and the quadratic attention term
stays negligible because `E = ceil(512/257) = 2` (`network.py:229-231`) keeps the attention
inner width at `2 × 257`.

---

## 2. Does the code assume `stride == n_fft/2`?

**No.** Every place the hop enters is hop-generic. Full enumeration:

| Location | What it does with the hop | Assumption? |
|---|---|---|
| `model.py:20-23` | accepts `stride` as a constructor argument | none |
| `model.py:25-30` | requires a positive int | none |
| `model.py:31-32` | `if n_fft % 2 or stride > n_fft // 2: raise` | rejects hop **>** `n_fft/2`; 128 passes |
| `model.py:44` | `minimum_samples = max(n_fft // 2 + 1, (emb_ks - 1) * stride)` | stride-scaled, but only a padding floor: 768 at stride 256, 384 at stride 128 |
| `model.py:61-65` | pads to `minimum_samples`, crops output to `original_length` | none |
| `spectral.py:31-35` | `torch.stft(hop_length=stride, center=True, onesided=True)` | none; Hann is NOLA at hop ≤ `n_fft/2` |
| `spectral.py:40` | `frame_lengths = N // hop + 1` | matches `center=True` frame count for any hop (verified: 251 / 501) |
| `spectral.py:41-44` | masks frames past `frame_lengths` | none |
| `spectral.py:60-64` | `torch.istft(..., length=int(lengths.max()))` | none; exact length via NOLA + envelope normalisation |
| `network.py:93` | `n_freqs = n_fft // 2 + 1` | depends on `n_fft` only |
| `network.py:156-159` | reads `n_frames, n_freqs` from the tensor | none |
| `network.py:168-171` | reshapes/decodes with the runtime `n_frames` | none |
| `network.py:280-282` | `T = ceil((old_T-emb_ks)/emb_hs)*emb_hs + emb_ks` (same for `Q`) | generic; with `emb_hs=1` no padding occurs at either hop |
| `network.py:101-102` | conv kernel/padding on `(T, F)` | none |

**Conclusion:** at the model/code level, `stride: 128` produces a valid, correctly
reconstructed signal path with zero code edits. Nothing in overlap-add, iSTFT, length
bookkeeping or normalisation is tied to 50 % overlap.

What is *not* free:

1. **Weights.** Both the 25-epoch local checkpoint and the DNS pretrained checkpoint
   (`configs/tfgridnet_dns.yaml:24`) were learned at 16 ms frame spacing. Parameter shapes
   would load without error — which is precisely the hazard: they would load *silently* and
   behave wrongly. A stride-128 profile is a new training run, never a fine-tune of the
   existing checkpoint.
2. **Training plumbing pins one config.** `scripts/plan_training.py:13` reads the config
   path from `configs/training_protocol.yaml` (`configs/training_protocol.yaml:14` —
   `tfgridnet: configs/tfgridnet.yaml`); `scripts/plan_training.py:25` rejects any
   TF-GridNet config that carries a `profile:` key; `:30-32` pin `n_layers == 6` and the
   literal profile name `local_6layer`; `speech_denoising/training/real_runner.py:43-47`
   fingerprints only `configs/tfgridnet.yaml`. A second profile therefore needs edits to
   those files, not just a new YAML.
3. **Cost.** ≈ 2× MACs and ≈ 2× activation memory (§5).

---

## 3. Is the paper's 8 ms hop tied to its 32 ms window? Is 16 ms a deliberate deviation?

**Yes, it is tied.** 8 ms over a 32 ms window is a 25 % hop, i.e. 75 % overlap; the same
sentence in `.research/tfgridnet.txt:707-712` fixes both numbers, and the derived values
(512-point DFT, 257 bins, E = 2) follow from the 32 ms window at 16 kHz, not from the hop.

**Repository provenance is inferable but not stated.** `network.py:72-73` keeps the upstream
ESPnet defaults `n_fft=128, stride=64`, which is the *8 kHz* pair (16 ms window / 8 ms hop)
at 50 % overlap. Scaling that ratio by 4 for 16 kHz gives exactly `512 / 256`. So the local
profile most likely inherited upstream's 0.5 hop/window ratio instead of the paper's 0.25 —
that is a *derivation* from the code defaults, not a documented decision anywhere.

**What the repository documentation claims:**

- `docs/paper_alignment.md:99` — "STFT | sqrt-Hann, 32 ms / **8 ms**, FFT 512, 257 bins |
  `n_fft=512`, `stride=256` → **16 ms hop**". Correct, and honest.
- `docs/paper_alignment.md:113-116` — "The 16 ms hop is a real architectural mismatch, not a
  cosmetic config value: the paper's 8 ms hop doubles the frame rate and changes every
  temporal convolution/attention shape. Changing it requires a new model profile and a new
  training run; it is not a drop-in edit to the existing checkpoint." — the conclusion is
  right; "changes every ... shape" should say *activation* shape (params are identical, §1).
- `README.md:115-116` — "its 8 ms hop and Wav+Mag+MC loss are structural differences from the
  local 16 ms-hop ...". Correct.
- `docs/paper_alignment.md:156-158` and `:167-168` — restate the mismatch as requiring a new
  profile and run. Correct.
- The sqrt-Hann deviation (`spectral.py:27` plain Hann vs paper's square-root Hann) is **not
  documented** anywhere; `docs/paper_alignment.md:99` lists the paper's sqrt-Hann but the
  repository column only mentions `stride=256`.

So the hop deviation is deliberate *in effect* (documented, accepted, labelled as a
mismatch) but the documentation does not say it was chosen for compute; and it is silent on
the window shape.

---

## 4. Fair correction path

Three candidates, honestly priced:

**A. Documented deviation only (do now, zero risk).** Keep `stride: 256`. It is already
documented in three places; add the two corrections above (parameter-vs-activation wording,
and the sqrt-Hann omission). Cost: one doc edit, no run invalidated. This is the only
correction that is fully compatible with today's read-only audit and with the existing 25-epoch
pilot and DNS checkpoint.

**B. New config profile (the right technical fix, but not config-only).** `stride: 128` needs
no architecture change — parameters are bit-for-bit identical (§1) and no code assumes 50 %
overlap (§2). But it is not a YAML drop-in: `configs/training_protocol.yaml:14`,
`scripts/plan_training.py:25,30,32` and `speech_denoising/training/real_runner.py:43-47` all
pin the single `local_6layer` / `configs/tfgridnet.yaml` identity, and a config carrying a
`profile:` key is rejected outright by `plan_training.py:25`. So B = one new YAML **plus**
four small, reviewable plumbing edits, **plus** a new run directory under
`results/paper_runs/`, **plus** a full retrain (no checkpoint reuse).

**C. New model variant (not justified).** There is no architectural dimension to change: no
layer's shape depends on the hop. Creating a variant class would duplicate `TFGridNetModel`
without differing in a single parameter. If paper fidelity is the goal, the things that do
need work are the *recipe*, not the architecture: sqrt-Hann analysis window and the Wav+Mag+MC
objective (`docs/paper_alignment.md:117-119`).

**Verdict:** the deviation is real, exactly 2× in frame rate, and **safe to correct in
principle but not free in practice**. The cheapest honest state today is **A**, with **B**
queued as a separately reviewed change. **C** is rejected.

Recommended wording for the future doc fix (proposal, not applied):

> The 16 ms hop doubles the frame period relative to the paper's 8 ms. It does not change any
> parameter shape (8,381,504 at either hop) but doubles the frame axis `T` (251 → 501 at 4 s),
> doubles MACs and roughly doubles activation memory. Correcting it is a new training run, not
> a drop-in edit to the existing checkpoint. The analysis window is also plain Hann rather than
> the paper's square-root Hann.

---

## 5. Compute and memory for one 4 s segment, 6-layer local profile

Batch 1, FP32, 64,000 samples, `emb_ks=4`, `emb_hs=1`, `H=192`, `C=48`, `E=2`.

| | stride 256 (T=251) | stride 128 (T=501) |
|---|---:|---:|
| Frames × bins | 251 × 257 | 501 × 257 |
| MACs per segment | 5.17 × 10¹¹ | 1.03 × 10¹² |
| GMAC/s (4 s) | ≈ 129 | ≈ 259 |
| Largest per-block activation `[T,F,C·emb_ks]` | 49.5 MB | 99.0 MB |
| `[T,F,2H]` (BLSTM output, saved for backward) | 99.1 MB | 198.2 MB |
| Attention buffer `[4,T,T]` | 1.0 MB | 4.0 MB |
| **Measured** train peak (fwd+bwd, batch 1) | **10,553,112,576 B ≈ 9.83 GiB** | not measured; ≈ **2× ≈ 19–21 GB** |

Anchors and reasoning:

- The 9.83 GiB anchor is `docs/tfgridnet_real_training.md:10` (RTX 5090, FP32, 4 s crop,
  batch 1, forward + backward). The same report records batch 4 as CUDA OOM, and whole-utterance
  inference at 185,760 samples peaking at 5,183,784,448 B
  (`docs/tfgridnet_real_training.md:11-12`).
- The stride-128 estimate is ≈ 2× because peak memory is dominated by per-block activations
  that are linear in `T`; the only super-linear term (attention, `T²`) is 1–4 MB and does not
  move the total. It is an **estimate**, not a measurement — the boundary probe is a single
  GPU forward/backward, which this audit is not permitted to run.
- On a 32 GB RTX 5090, ≈ 19–21 GB at microbatch 1 leaves headroom, but not enough to relax the
  existing `micro_batch_size=1, gradient_accumulation_steps=4`
  (`docs/tfgridnet_real_training.md:18-19`). Per-epoch wall clock and the reported whole-utterance
  RTF 0.0843 (`docs/tfgridnet_seed42_25ep.md:16`) would both roughly double.

---

## Verdict

The repository's TF-GridNet runs a **16 ms hop against the paper's 8 ms at the same 32 ms
window and 257 bins** — a 2× frame-rate deviation, caused at the model level by nothing but
the `stride` value, and accompanied by an undocumented plain-Hann vs sqrt-Hann difference.
No code path assumes 50 % overlap, so the hop is *mechanically* correctable by configuration;
but the correction invalidates both existing weight sets, roughly doubles compute and
activation memory, and is blocked by a single-config identity pin in the training planner.
Reporting the deviation is correct today; changing it is a new experiment, not a patch.

## Safe next step

1. **Keep `stride: 256` and the existing checkpoints.** Nothing in this audit justifies
   touching them.
2. **Fix the two documentation inaccuracies** in a separate, reviewable doc change:
   `docs/paper_alignment.md:113-116` ("changes every temporal convolution/attention shape" →
   *activation* shapes only; parameters are 8,381,504 at either hop) and the missing sqrt-Hann
   note beside `docs/paper_alignment.md:99`.
3. **If paper fidelity is wanted**, open one scoped change containing: a
   `configs/tfgridnet_paper_stft.yaml` with `stride: 128`, the four plumbing edits
   (`configs/training_protocol.yaml:14`, `scripts/plan_training.py:25,30,32`,
   `speech_denoising/training/real_runner.py:43-47`), a `results/paper_runs/` destination, and
   **one** boundary probe (single 4 s crop, batch 1, forward+backward) to confirm the ≈19–21 GB
   estimate before committing to a 25-epoch budget. Treat sqrt-Hann and Wav+Mag+MC as separate
   decisions — bundling them makes the comparison uninterpretable.
4. Do **not** load the DNS pretrained checkpoint into a stride-128 model: it would load
   silently and be wrong.

---

## Five-line summary

1. Repo runs `n_fft=512`, `stride=256` → 16 ms hop, 257 bins, T=251 for 4 s; the paper uses 8 ms
   hop at the same 32 ms window (75 % overlap) → T=501.
2. Parameters are **identical** at both hops (8,381,504; 1,396,605 per block) — only activation
   shapes, MACs (5.17e11 → 1.03e12 per 4 s, 2.00×) and memory (≈9.83 GiB → ≈19–21 GB) change.
3. No code assumes `stride == n_fft/2`: STFT/iSTFT/frame counts/block padding are all
   hop-generic, so 128 works mechanically — but existing weights become invalid, not loadable.
4. The 8 ms hop is tied to the 32 ms window; 16 ms is a documented repo deviation
   (`docs/paper_alignment.md:99,113-116`) that the code defaults suggest was inherited from
   upstream's 8 kHz `128/64` ratio, and the plain-Hann (vs sqrt-Hann) difference is undocumented.
5. Verdict: document the deviation now, reject a model variant, and gate a `stride: 128` profile
   behind one reviewed change with a single GPU memory probe; do not retrofit the checkpoints.
