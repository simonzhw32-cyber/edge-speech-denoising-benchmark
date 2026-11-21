# GTCRN paper-training track: implementation-ready plan (2026-11-09)

> **Status:** Implemented 2026-11-09 as a historical plan. The config, module, trainer, smoke suite (15 checks) and 3-step fixture now exist; “Safe next step” below is superseded. A real run still requires operator-supplied `--epochs` and `--segment-seconds`.

Status: **read-only plan**. No training was started, no GPU was used, no dependency was
installed, no existing file was modified. The only write is this file.

Scope: define exactly what a faithful GTCRN (ICASSP 2024, doi 10.1109/ICASSP48485.2024.10448310)
paper-recipe training track requires on this repository, and give an engineer everything needed
to implement it without re-reading the paper — including everything the paper does **not** say.

Method: source reading with line citations; CPU-only probes with the bundled interpreter
(`.conda/python.exe`, torch 2.11.0+cu128, no `CUDA_VISIBLE_DEVICES` set, no optimizer steps,
no checkpoints written); the official `Xiaobin-Rong/gtcrn` sources at the pinned commit
`502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73` were fetched over HTTP and quoted verbatim;
arithmetic on the local VCTK-DEMAND manifest for corpus statistics. Numbers below marked
"measured" come from those probes; nothing is estimated unless it says so.

---

## 0. Baseline: three recipes, one model

| Item | Repository unified benchmark | Paper (GTCRN, VCTK-DEMAND) | This plan |
|---|---|---|---|
| Entry point | `scripts/train.py --real` (`scripts/train.py:10-23`) | none published for VCTK | new `scripts/train_gtcrn_paper.py` |
| Config source | `configs/training_protocol.yaml` (closed schema) | paper text | new `configs/gtcrn_paper.yaml` |
| Optimizer | AdamW (`real_runner.py:183-184`) | Adam, lr 1e-3 | Adam, lr 1e-3 |
| LR schedule | none (`protocol.py:55-61`) | halve after 5 stagnant validations | `ReduceLROnPlateau(min, 0.5, 5)` |
| Loss | negative SI-SNR only (`protocol.py:62-67`) | `0.01·L_SISNR + 0.7·L_mag + 0.3·(L_real+L_imag)` | exact port of official `loss.py` |
| STFT | 512/256, sqrt-Hann (`model.py:18,31-35`) | 512/256, sqrt-Hann | identical |
| Train pool | 10,778 (speaker-disjoint) | **11,572** (full VCTK-DEMAND train) | `paper_full` = 11,572 |
| Epochs | 100 (protocol) | **not stated** | operator-supplied, no default |
| Output | `results/training_runs/**`, `training_reports/**` | — | `results/paper_runs/gtcrn_paper_*` |

The unified protocol validator is deliberately closed: it asserts the exact model set
(`protocol.py:33-34`), the exact training block (`protocol.py:55-61`), the exact loss dict
(`protocol.py:62-67`), the validation contract (`protocol.py:68-73`) and the test contract
(`protocol.py:74-77`). Adding an `Adam`/`ReduceLROnPlateau`/hybrid-loss field anywhere in that
schema raises. **Therefore the paper track must be a separate module with its own runner; do not
extend `protocol.py`.**

---

## 1. Model interface: what exists, and what the loss needs

### 1.1 The two APIs

| Object | File | Input | Output |
|---|---|---|---|
| `GTCRNNetwork.forward(spec)` | `speech_denoising/models/gtcrn/network.py:293-319` | `[B, 257, T, 2]` float | `[B, 257, T, 2]` float (enhanced spec) |
| `GTCRNModel.forward(noisy_audio)` | `speech_denoising/models/gtcrn/model.py:20-43` | `[B, L]` float | `[B, L]` float (**waveform only**) |
| `GTCRN` alias | `speech_denoising/models/gtcrn/model.py:47` | — | — |

**Answer to "does forward return waveform only?": yes.** `GTCRNModel.forward` builds the
spectrum at `model.py:31-35`, hands `torch.view_as_real(spec)` to the network at `model.py:36`,
inverts at `model.py:37-42` with `length=padded.shape[-1]`, and returns `enhanced[..., :length]`
at `model.py:43`. The enhanced spectrum never leaves the call, so the paper loss — which is
defined entirely on `(Ŝ_real, Ŝ_imag, S_real, S_imag)` plus two ISTFTs — cannot be computed
from this return value. The waveform output is a dead end for training; do not try to STFT it back.

### 1.2 Verified shapes (CPU probe, `torch.no_grad`, seed 0)

| Stage | Expression | Shape |
|---|---|---|
| audio | `x` | `[B, L]` float32 |
| STFT | `torch.stft(x, 512, 256, 512, sqrt_hann, center=True, return_complex=True)` | `[B, 257, T]` complex64 |
| network in/out | `torch.view_as_real(spec)` → `network(...)` | `[B, 257, T, 2]` float32 |
| mask path | `Mask.forward(m, spec)` at `network.py:316` | mask `[B, 2, T, 257]` × spec `[B, 2, T, 257]` |
| ISTFT | `torch.istft(spec, 512, 256, 512, sqrt_hann)` | `[B, (T-1)·256]` |
| `GTCRNModel` end-to-end | `model(torch.randn(2, 16000))` | `[2, 16000]` — measured |

Frame count is `T = 1 + L // hop_length` (measured: `L=16,000 → T=63`, `L=64,000 → T=251`,
`L=36,480 → T=143`). `n_freqs = 257` is constant (`model.py:12-13`, `network.py:301-302`).

The window is **sqrt-Hann and must stay sqrt-Hann**: `model.py:18` registers
`torch.hann_window(512).sqrt()`; the probe confirmed `model.window == torch.hann_window(512).sqrt()`
exactly, dtype float32, shape `[512]`. `network.py:301` uses the same `+1e-12` inside its own
magnitude, independently of the loss.

### 1.3 What must change

Nothing in `model.py` needs to change, and nothing in it may change. The paper track needs a
**training-time frontend wrapper that owns a `GTCRNNetwork` and exposes the spectrum**, so the
loss can see both `Ŝ` and `S`. Requirements:

1. Use `GTCRNNetwork` directly (`from speech_denoising.models.gtcrn.network import GTCRN as GTCRNNetwork`),
   not `GTCRNModel`, so the enhanced spectrum is available.
2. Keep the submodule attribute name `network` and a non-persistent `window` buffer, so the
   `state_dict` keys match `GTCRNModel`'s (`network.*` only; `model.py:18` is
   `persistent=False`) and checkpoints are interchangeable in both directions.
3. Reproduce the frontend exactly as `model.py:31-35`: `center=True`, `pad_mode="reflect"`,
   `normalized=False`, `onesided=True`, `win_length = n_fft`, and `return_complex=True` +
   `torch.view_as_real`. The official `infer.py` uses the deprecated
   `return_complex=False` (torch 2.11 warns); the values are identical, so use the
   non-deprecated form.
4. Do **not** inherit `BaseEnhancementModel`: its contract is waveform-in/waveform-out
   (`models/base.py`) and it would force a loss-hostile signature.

### 1.4 Concrete signatures

```python
# speech_denoising/training/paper_gtcrn.py

class GTCRNPaper(nn.Module):
    """sqrt-Hann STFT frontend + official GTCRN network, spectrum in and out."""

    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 256
    n_freqs: int = 257

    def __init__(self) -> None: ...

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """[B, L] float32 -> [B, 257, T, 2] float32, T = 1 + L // hop_length."""

    def istft(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """[B, 257, T, 2] float32 -> [B, L]; length=None reproduces the official
        (T-1)*hop truncation used by loss.py."""

    def enhance_spec(self, spec: torch.Tensor) -> torch.Tensor:
        """[B, 257, T, 2] -> [B, 257, T, 2]; thin wrapper over self.network."""

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """[B, L] -> [B, L] waveform, parity with GTCRNModel for smoke comparisons."""
```

```python
# speech_denoising/training/paper_gtcrn.py

class HybridLoss(nn.Module):
    """Exact port of Xiaobin-Rong/gtcrn loss.py @502ebfa (see §2.1)."""

    def __init__(
        self,
        *,
        n_fft: int = 512,
        hop_length: int = 256,
        alpha: float = 0.01,      # weight of L_SISNR
        beta: float = 0.3,        # weight of (L_real + L_imag)
        mag_power: float = 0.3,   # exponent on |S| in L_mag
        ri_power: float = 0.7,    # exponent on each signal's own magnitude
        mag_eps: float = 1e-12,   # inside sqrt, official constant
        sisnr_eps: float = 1e-8,  # projection + ratio denominator
        scale: str = "paper",     # "paper" | "official_x100"
    ) -> None: ...

    def forward(
        self, est: torch.Tensor, ref: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """est, ref: [B, 257, T, 2]. Returns (total, {'ri','mag','sisnr','total'})."""
```

`HybridLoss` must hold its own `window` buffer (`register_buffer(..., persistent=False)`) so it
follows `.to(device)` and stays out of checkpoints.

---

## 2. The loss, exactly

### 2.1 Authoritative source

The paper text is **not readable locally**: `.research/gtcrn.pdf` is a 16,794-byte IEEE Xplore
HTML stub (it begins with `MEMBER_PROFILE_CHANGE_USERNAMEPASS_LINK` and contains zero
`/Type /Page` objects). `.research/gtcrn.txt` and `.research/gtcrn_loss.py` do not exist;
`.research/upstream/` holds only LiSenNet files. The official code was therefore fetched from
`Xiaobin-Rong/gtcrn` at the commit pinned by `configs/paper_recipes.yaml:16`, file `loss.py`:

```python
pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
pred_real_c = pred_stft_real / (pred_mag**0.7)      # each signal, its OWN magnitude
pred_imag_c = pred_stft_imag / (pred_mag**0.7)
true_real_c = true_stft_real / (true_mag**0.7)
true_imag_c = true_stft_imag / (true_mag**0.7)
real_loss = nn.MSELoss()(pred_real_c, true_real_c)
imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
mag_loss  = nn.MSELoss()(pred_mag**0.3, true_mag**0.3)
y_pred = torch.istft(..., 512, 256, 512, window=torch.hann_window(512).pow(0.5))
y_true = torch.istft(..., 512, 256, 512, window=torch.hann_window(512).pow(0.5))
y_true = torch.sum(y_true * y_pred, -1, keepdim=True) * y_true / (torch.sum(torch.square(y_true), -1, keepdim=True) + 1e-8)
sisnr = -torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2
                     / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2 + 1e-8) + 1e-8).mean()
return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr
```

### 2.2 Equations

Let `Ŝr, Ŝi` be `est[..., 0], est[..., 1]` and `Sr, Si` the reference components, all `[B, F, T]`.

```
|Ŝ| = sqrt(Ŝr² + Ŝi² + 1e-12)                     |S| = sqrt(Sr² + Si² + 1e-12)

L_ri  = mean[(Ŝr/|Ŝ|^0.7 − Sr/|S|^0.7)²] + mean[(Ŝi/|Ŝ|^0.7 − Si/|S|^0.7)²]

L_mag = mean[(|Ŝ|^0.3 − |S|^0.3)²]

y_hat = ISTFT(Ŝr + j·Ŝi),  y = ISTFT(Sr + j·Si)          # no length=, i.e. (T−1)·256 samples
y_t   = (<y, y_hat> / (‖y‖² + 1e-8)) · y                 # projection of y onto y_hat
L_sisnr = mean_b[ −log10( ‖y_t‖² / (‖y_hat − y_t‖² + 1e-8) + 1e-8 ) ]

L_total = alpha·L_sisnr + (1 − beta)·L_mag + beta·L_ri
        = 0.01·L_sisnr + 0.7·L_mag + 0.3·L_ri
```

with `alpha = 0.01`, `beta = 0.3` (`configs/paper_recipes.yaml:31-32`). Both `mean` are means
over all elements: `nn.MSELoss()`'s default reduction is `mean` over `B×F×T`. Only `L_sisnr` is
per-utterance before its mean over the batch.

### 2.3 Two traps in the prose form of the recipe

1. **"RI normalized by |S|^0.7" is only half true.** The official code normalizes *each* signal
   by *its own* magnitude: `Ŝr/|Ŝ|^0.7` against `Sr/|S|^0.7`. Normalizing both by the clean
   magnitude instead is a different objective: since `Sr = |S|·cosθ`, the official target is
   `|S|^0.3·cosθ`, while the "both by |S|^0.7" variant reduces to `cosθ` and therefore silently
   up-weights quiet bins by `|S|^−0.3` (for `|S| = 1e-4` that is a factor of ~16). Implement the
   official per-signal form; §4.7 test 8 exists to catch this exact mistake.
2. **`L_sisnr` is not decibels.** It is `−log10(ratio)`, i.e. `−(SI-SNR in dB)/10`. The mean over
   the batch is taken after the `log10`. A `−10·log10` "dB" implementation changes the relative
   weight of the SI-SNR term by 10× and makes the 0.01 coefficient meaningless.

### 2.4 The ×100 scale convention

`30·L_ri + 70·L_mag + 1·L_sisnr` = `100 × (0.3·L_ri + 0.7·L_mag + 0.01·L_sisnr)`. Measured on
the same tensors: `official = −10.2014`, `paper = −0.1020`, ratio `100.000006`. Both train
essentially identically (Adam is nearly scale-invariant), but:

- logged `train_loss` differs by 100× between conventions, so a `learning_curve.csv` is
  uninterpretable unless the convention is recorded;
- **a gradient-clip threshold is scale-dependent.** Measured on one real utterance at random
  init: `‖g‖ = 0.1996` in paper scale, `≈ 20.0` in official scale. A clip value of 5.0 would
  never bind in paper scale and would bind on the first step in official scale. Never copy a
  clip value across scales.

Decision: implement paper scale (`scale="paper"`), record `loss_scale: "paper"` in `run.json`,
and keep `scale="official_x100"` available only as a cross-check in the smoke test.

### 2.5 Stability analysis (measured, not assumed)

| Case | Result | Why |
|---|---|---|
| `est == ref` | `L_ri = 0`, `L_mag = 0`, `L_sisnr = −(log10‖y‖² + 8)` = **−10.2014**, total **−0.1020** | `‖y_hat − y_t‖² = 0` → ratio `= ‖y‖²/1e-8` |
| all-zero `est` and `ref` | total **+0.0800** exactly, gradients finite (and zero) | `|Ŝ| = |S| = 1e-6`, ratio `0/1e-8 = 0`, `−log10(1e-8) = 8` |
| random `est`, real `ref` | gradients finite, `max|g| ≈ 1.6e-3` on the synthetic case | `istft` is differentiable; no NaN path |
| one real utterance, random init | `L_ri = 0.3383`, `L_mag = 0.0212`, `L_sisnr = +0.4938` (−4.94 dB), total `0.1213` | RI term dominates at init because `est ≈ 0` |

Consequences for the implementation:

- `1e-12` **must be inside the sqrt** (`|Ŝ| ≥ 1e-6` always), which floors the RI denominator at
  `(1e-6)^0.7 ≈ 6.3e-5`. No clamp, no `torch.where`, no `nan_to_num` is needed or wanted; the
  official code has none.
- `1e-8` appears twice in `L_sisnr` (`+1e-8` on the projection denominator and `+1e-8` inside the
  `log10`) and nowhere else. Do not "normalize" them to a single constant.
- The loss is **not non-negative**, so `loss < 0` in a log is normal, not a bug; the scheduler
  only needs monotone semantics (`mode="min"`), not a positive range.
- `0.3 + 0.7 = 1` is load-bearing: it makes the RI target bounded by `|S|^0.3 ≤ 1` for
  `|S| ≤ 1`. Changing one exponent without the other changes the objective's dynamic range.
- `L_mag`'s derivative `0.3·|S|^−0.7` diverges as `|S| → 0`; the `1e-12` inside the sqrt caps it
  at `0.3·(1e-6)^−0.7 ≈ 4.7e3` for the magnitude branch. That is finite but large — expect a
  handful of large-magnitude gradients on near-silent frames and do not add epsilon elsewhere to
  "fix" it.

### 2.6 An implementation already exists in the tree, and what it changes

`speech_denoising/losses/gtcrn.py` is present **untracked** (150 lines, written 2026-11-09 15:01,
not by this session). Its `GTCRNHybridLoss.forward(enhanced_audio, clean_audio, lengths=None)`
(`:86`) is **formula-exact**: per-signal `mag^0.7` (`:116-119`), `1e-12` inside the sqrt
(`:113-114`), `mse_loss` defaults (`:121-123`), no `length=` in `_istft` (`:73-84`), `−log10`
SI-SNR with `1e-8` in both places (`:132-141`), and the official `30/70/1` scale (`:143-147`).
Cross-checked on one real utterance against a hand-written official form: **bit-identical
(measured diff `0.000e+00`, both `79.526039`)**.

Two deliberate divergences make it unsuitable as *the* paper-track training loss:

| Divergence | Measured effect |
|---|---|
| Waveform in, internal STFT → it scores `STFT(ISTFT(Ŝ))`, not the network's `Ŝ` | paper-scale loss `0.108695` (on `Ŝ`) vs `0.102358` (re-analysed) → **5.83 %**, RI `−11 %`, mag `+28 %`, SI-SNR unchanged; `max|Ŝ − STFT(ISTFT(Ŝ))| = 44.3` at the boundary frames |
| Official ×100 scale | `train_loss` values 100× the paper scale; no clip value is transferable (§2.4) |

It also cannot serve as the padding solution of §4.4 option B: `lengths` truncates the whole
batch to `int(lengths.min())` (`:127-130`) and leaves the padded frames inside the RI/mag means.

Decision: do **not** modify that file; use it as the independent oracle in the smoke test (§4.7
test 14) and implement the training loss on spectra (§2.2). The 5.83 % delta is the price of the
waveform interface, and it is not the paper's objective.

---

## 3. Optimizer, scheduler, and the epoch budget that does not exist

### 3.1 Optimizer

| Property | Value | Source |
|---|---|---|
| Type | `torch.optim.Adam` | `configs/paper_recipes.yaml:26` |
| lr | `1e-3` | `configs/paper_recipes.yaml:27` |
| betas | `(0.9, 0.999)` (default; paper silent) | torch default — flagged as unknown |
| eps | `1e-8` (default) | torch default — flagged as unknown |
| weight decay | `0.0` (paper silent; `paper_recipes.yaml:37`) | Adam default |
| Parameters passed | trainable only: 203 tensors / **23,669** params | measured; mirror the filter style of `real_runner.py:183` |

`GTCRNNetwork` has 48,245 parameters in 205 tensors, of which the two ERB matrices
(`network.py:20-21`, `nn.Parameter(..., requires_grad=False)`, 64×192 and 192×64 = 24,576) are
frozen. Passing `net.parameters()` unfiltered is harmless (Adam skips `grad is None`) but hides
the frozen set from any optimizer-signature record, so filter on `requires_grad`.

### 3.2 Scheduler

The paper's "halve when validation loss does not decrease for 5 epochs" is implemented, in the
author's training template (`Xiaobin-Rong/SEtrain`, `trainer.py`), as:

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
...
self.scheduler.step(valid_loss)     # once per epoch, after the validation pass
```

Three concrete facts for the port:

1. **`verbose=True` is gone.** On torch 2.11.0 the constructor signature ends at
   `..., cooldown=0, min_lr=0, eps=1e-08)` and passing `verbose` raises
   `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`
   (measured). Copy the call without it.
2. **Patience semantics are one epoch later than the prose.** Measured with a constant metric:
   lr stays `1e-3` over step calls 1–6 and drops to `5e-4` on call 7; `2.5e-4` on call 13. Because
   torch reduces when `num_bad_epochs > patience`, "5 stagnant validations" (paper) becomes a
   first drop after the 6th consecutive non-improving epoch. Keep `patience=5` (author's value,
   i.e. faithful to the code) and record the off-by-one in the notes rather than "fixing" it.
3. **"Not decreased" is a 0.01 % relative test, not equality.** Defaults are
   `threshold=1e-4, threshold_mode="rel"`: an epoch counts as improvement only if
   `loss < best·(1 − 1e-4)`. Set them explicitly in the config so the behaviour is visible.

Undocumented in the paper and therefore **not** invented here: `min_lr` (leave 0), `cooldown`
(leave 0), and whether the scheduler monitors the full validation loss or a validation SI-SNR.
Monitor the validation **loss** (that is what the author's code passes).

### 3.3 Gradient clipping

`SEtrain`'s trainer calls `torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)` after
every backward, with the value in its DNS3 `config.toml`. The GTCRN paper does not state clipping.
Plan: `grad_clip_norm: null` by default (no clipping), explicitly settable, and never copied from
`SEtrain` — see §2.4 for why a clip value is meaningless without the loss scale.

### 3.4 The epoch budget is missing — do not invent one

Three independent sources agree, and they are the only sources available:

1. `configs/paper_recipes.yaml:35` — `epochs_vctk: null`, with `:36-39`: "The paper states no
   VCTK epoch count, segment length, or weight decay. The DNS3 recipe chunks utterances to 8 s and
   samples 40,000 pairs per epoch; those values do not apply to VCTK-DEMAND."
2. `docs/paper_alignment.md:47` — "A 'paper epoch count' for GTCRN does not exist."
3. The official repository at the pinned commit contains `gtcrn.py`, `infer.py`, `loss.py`,
   checkpoints, streaming/LADSPA code and test wavs — **no training script at all**
   (`train.py` → HTTP 404). Its README redirects training to the separate `SEtrain` template,
   whose `epochs` lives in a DNS3 `config.toml`; nothing in either repo ties an epoch count to
   VCTK-DEMAND.

Required behaviour: `epochs: null` in the config, `--epochs` **required** on the CLI with
`type=int` and no default, and `run.json` must record
`"epochs_source": "operator (paper does not define a VCTK epoch budget)"`. The same treatment
applies to `segment_seconds` (§4.4). A default of 100 here would be the unified protocol's number
(`protocol.py:55-61` / `docs/project_requirements.md`), not the paper's, and would silently
counterfeit fidelity.

---

## 4. Implementation plan

### 4.0 Boundaries (what must not be touched)

| Protected | Why |
|---|---|
| `speech_denoising/training/{protocol,runner,real_runner,validation,selection,real_store,run_store}.py` | unified benchmark semantics (`protocol.py:33-77`) |
| `scripts/{train,validate,evaluate,evaluate_trained,plan_training,preflight_training}.py` | unified entry points |
| `configs/{training_protocol,gtcrn,paper_recipes}.yaml` | closed schema; `paper_recipes.yaml` is a reference document |
| `speech_denoising/models/gtcrn/{model,network}.py`, `models/__init__.py` | do **not** register `gtcrn_paper` in `MODEL_REGISTRY`; that would make it selectable by the unified runner |
| `training_reports/**`, `benchmark_reports/**`, `results/training_runs/**`, `data/**` | immutable historical artifacts |

All new writes go to `results/paper_runs/<run name>/`. No `configs/*.yaml` glob exists in
`scripts/` (checked), so adding a config file cannot perturb an existing smoke test.

### 4.1 New files only

| File | Lines (est.) | Purpose |
|---|---|---|
| `configs/gtcrn_paper.yaml` | ~45 | paper-track settings; `extends: gtcrn.yaml` |
| `speech_denoising/training/paper_gtcrn.py` | ~380 | frontend, loss, scheduler, data selection, loop, run-spec |
| `scripts/train_gtcrn_paper.py` | 6 | CLI wrapper (`from speech_denoising.training.paper_gtcrn import parse_args, run`) |
| `scripts/smoke_gtcrn_paper.py` | ~200 | synthetic-tensor verification, CPU only |

Pre-existing and **not to be modified**: `speech_denoising/losses/gtcrn.py` (untracked,
`GTCRNHybridLoss`, 150 lines, written by a concurrent session at 15:01). It is a formula-exact
waveform-level port used here only as the oracle in smoke test 14 (§2.6); it is not imported by the
training module.

Mirror `scripts/train_lisennet_paper.py` (6 lines) and `speech_denoising/training/paper_lisennet.py`
for structure: module docstring stating the separation, `sha256_file`/`git_commit`/`environment_record`
helpers (duplicate the ~30 lines rather than importing `paper_lisennet`, which drags in `pesq`,
`joblib` and the LiSenNet discriminator), `run(args)` guard at `paper_lisennet.py:261-264`, and
the run-dir guard `run_dir.is_relative_to(REPO_ROOT / "results/paper_runs")`.

### 4.2 `configs/gtcrn_paper.yaml`

```yaml
# GTCRN paper-recipe training track. Reference-only sibling of the unified benchmark:
# it is NOT read by scripts/train.py, protocol.py or real_runner.py.
extends: gtcrn.yaml

paper_track:
  reference:
    paper: "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources"
    venue: "ICASSP 2024, pp. 971-975"
    doi: "10.1109/ICASSP48485.2024.10448310"
    official_code: "Xiaobin-Rong/gtcrn@502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
    dataset: "VCTK-DEMAND, 16 kHz, 11,572 train / 824 test"

  # --- defined by the paper ---
  sample_rate: 16000
  n_fft: 512
  hop_length: 256
  window: sqrt_hann
  batch_size: 4
  optimizer: Adam
  learning_rate: 0.001
  scheduler: {type: ReduceLROnPlateau, mode: min, factor: 0.5, patience: 5,
              threshold: 0.0001, threshold_mode: rel, cooldown: 0, min_lr: 0.0}
  loss: {alpha: 0.01, beta: 0.3, magnitude_power: 0.3, ri_power: 0.7,
         mag_eps: 1.0e-12, sisnr_eps: 1.0e-8, scale: paper}

  # --- NOT defined by the paper: null means "must come from the CLI" ---
  epochs: null
  segment_seconds: null
  grad_clip_norm: null
  weight_decay: 0.0
  betas: [0.9, 0.999]
  adam_eps: 1.0e-8

  # --- repository decisions, recorded as such ---
  train_pool: paper_full          # paper_full = 11,572 | held_out = 10,778
  validation: {split: repository_held_out_794, batch_size: 1, segment: full,
               purpose: scheduler_monitor_only}
  checkpoint: {every_epoch: true, selection: last}
  results_root: results/paper_runs
  seed: 42
```

`load_config` (`speech_denoising/utils/utils.py:16-29`) is `extends`-aware and deep-merges, so
the `model:` block inherited from `gtcrn.yaml:2-4` stays intact and `config_sha256` stays
reproducible.

### 4.3 Data selection: 11,572 is available locally

Measured from `data/voicebank-demand-16k/train_manifest.json`: `count = 11572`,
`utterances` has 11,572 rows of `{utterance_id, noisy_path, noisy_sha256, clean_path,
clean_sha256, samples}`; durations 1.09 s–15.11 s (median 2.68 s, p90 4.07 s), 9.39 h total,
`T` from 69 to 945. The 10,778/794 speaker-disjoint split is **derived at plan time**, not stored:
`subset_indices` partitions the same parent rows (`data.py:42-43`).

So the paper's full training pool needs no new data:

```python
path, manifest_bytes, manifest, plan = load_training_inputs(manifest_path, "gtcrn")
all_rows = tuple(range(len(manifest["utterances"])))          # 11,572
assert set(plan["split"]["train_ids"]) | set(plan["split"]["validation_ids"]) == set(
    row["utterance_id"] for row in manifest["utterances"]
)                                                            # guaranteed by data.py:42-43
```

`train_pool: paper_full` → train on `all_rows`, monitor `plan["split"]["validation_ids"]`;
`train_pool: held_out` → `subset_indices(manifest, plan, "train")`, monitor
`subset_indices(manifest, plan, "validation")`. The default `paper_full` is faithful in pool size
but its monitor is **in-sample**; therefore (a) `validation.purpose` is
`scheduler_monitor_only`, (b) no checkpoint is selected by it (`checkpoint.selection: last`),
and (c) `run.json` records the contamination flag. `held_out` is the clean-monitor alternative at
the cost of 794 training utterances.

### 4.4 Batching: the paper's batch 4 vs its unknown segment length

The official loss has no padding mask: `nn.MSELoss()` averages every element (`§2.2`), so padded
frames would be scored as signal. With 1.09–15.11 s utterances a batch of 4 either shares one
length or contaminates the loss. Options, in order of fidelity:

| Option | Batch 4 | Segment | Loss clean | Verdict |
|---|---|---|---|---|
| A: fixed-length random crop, `--segment-seconds S`, batch 4 | yes | invented (`S`) | yes | **default**, with `S` recorded as operator-supplied |
| B: bucket by length, pad to bucket max, mask the loss | yes | near-full | needs a mask the official loss lacks | acceptable only if the mask is recorded as a deviation |
| C: batch 1, full utterance | no | faithful | yes | use when batch fidelity matters less than segment fidelity |

Recommended: A. `segment_seconds` is `null` in the config and **required** on the CLI; nothing in
the plan supplies a number. Note that `S = 4.0` is the *unified protocol's* crop
(`docs/paper_alignment.md:32-41`), not a paper value — if the operator picks 4.0, `run.json` must
still say `segment_source: "operator"`.

Per-step shape with option A and `S` seconds: `L = 16000·S`, `T = 1 + L//256`; for `S = 4`,
`L = 64,000`, `T = 251`, and `ISTFT` without `length=` returns exactly `(251−1)·256 = 64,000`
samples, i.e. the truncation issue disappears. For `S = 1`, `T = 63` and the official ISTFT
returns 15,872 of 16,000 samples — that 128-sample truncation is the faithful behaviour of
`loss.py`, and the plan keeps it (`istft(..., length=None)` in the loss; `length=L` only in the
waveform-facing `GTCRNPaper.forward`).

### 4.5 Training step

```python
model.train(); est = model.enhance_spec(model.stft(noisy))     # [B,257,T,2]
total, parts = loss_fn(est, model.stft(clean))
optimizer.zero_grad(set_to_none=True)
total.backward()
if clip is not None:
    torch.nn.utils.clip_grad_norm_(trainable_params, clip)
assert torch.isfinite(total)
optimizer.step()
```

Validation: `model.eval()`, `torch.no_grad()`, batch 1, full utterance, same loss, mean over
utterances; `scheduler.step(val_total)` once per epoch; `optimizer.param_groups[0]["lr"]` logged
before the step so the curve shows the drop on the epoch it happens.
Caveat worth writing into the module docstring: at step 0 the BatchNorm running statistics are
the initialization values, so the first validation losses are not comparable to later ones; do
not act on an lr drop in the first few epochs.

### 4.6 Run artifacts

`results/paper_runs/<name>/`:

| File | Content |
|---|---|
| `run.json` | `schema_version: 1`, `scope: "gtcrn_paper_training"`, `status`, `recipe` (from the config), `dataset: {manifest path, sha256, count: 11572}`, `split: {train_pool, train_count, val_count, val_sha256, in_sample_monitor: true|false}`, `epochs` + `epochs_source`, `segment_seconds` + `segment_source`, `loss_scale`, `optimizer`/`scheduler`/`betas`/`weight_decay`/`adam_eps`/`grad_clip_norm`, `git: {commit, dirty}`, `environment`, `command`, `started_at`, `best`/`last` |
| `learning_curve.csv` | `epoch,lr,train_total,train_ri,train_mag,train_sisnr,train_sisnr_db,val_total,val_ri,val_mag,val_sisnr,val_sisnr_db,steps,seconds` |
| `checkpoints/epoch_XXXX.pt` | `{epoch, model, optimizer, scheduler, loss, rng, settings_sha256}` (mirror `state.py:1-80` for rng/`_signature`/`_finite`) |
| `checksums.json` | sha256 of each checkpoint |

Resume rules: same run dir + `--resume`, same `settings_sha256`, restore rng
(`state.capture_rng/restore_rng`), reject a changed `segment_seconds`, `epochs`, `train_pool`,
loss weights or git dirty state.

### 4.7 Tests with synthetic tensors — `scripts/smoke_gtcrn_paper.py`

Plain `assert` blocks, no pytest, no data, no GPU, `torch.manual_seed(0)`, print `[ok] name` per
check, exit 1 on the first failure. Every expected value below was measured on this machine.

1. `frontend_shapes` — `stft(randn(2, 64000))` → `(2, 257, 251, 2)`; `stft(randn(2, 16000))` →
   `(2, 257, 63, 2)`; dtype float32; `T == 1 + L // 256`.
2. `istft_lengths` — for `T = 63`: `istft(spec)` → `15,872` samples, `istft(spec, length=16000)`
   → `16,000`.
3. `roundtrip` — `istft(stft(x), length=L)` vs `x`: `max|Δ| < 1e-5`.
4. `parity_with_adapter` — `GTCRNPaper.stft(x)` equals `GTCRNModel`'s internal spectrum: compare
   `GTCRNPaper.forward(x)` against `GTCRNModel().forward(x)` after loading the same
   `state_dict` (keys match: `network.*`), `max|Δ| < 1e-6` on `x = randn(1, 16000)`.
5. `loss_identity` — `est = ref`: `ri == 0`, `mag == 0`,
   `total == 0.01·mean_b[−(log10‖y_b‖² + 8)]` within `1e-3` (measured `−10.2014` / `−0.1020`).
6. `loss_scale_equivalence` — `paper·100 == official` within `1e-4` (measured ratio
   `100.000006`) on `randn(2, 257, 63, 2)`.
7. `loss_degenerate` — all-zero `est` and `ref`: `total == 0.08` within `1e-6`, all gradients
   finite.
8. `ri_is_per_signal` — **closed form, one bin**: `ref = [[[[1.0, 0.0]]]]` (`|S| = 1`, phase 0),
   `est = 0.01 · ref`. The official per-signal form gives `L_ri = (1 − 0.01^0.3)² = 0.56072`; the
   prose "both normalized by the clean magnitude" variant gives `(1 − 0.01)² = 0.98010`. Assert
   `abs(ri − 0.560718) < 1e-4`. Both numbers were measured, and `est = 0` is useless here: with a
   zero estimate both variants agree exactly.
9. `sisnr_is_log10` — compute `ratio` by hand from the same formula on `randn(2, 257, 63, 2)`
   tensors, then assert `L_sisnr == −log10(ratio + 1e-8)` within `1e-6` **and**
   `abs(L_sisnr − (−10·log10(ratio + 1e-8))) > 1` (measured `4.56905` vs `45.69052`, factor
   exactly 10). Rescaling both signals does *not* discriminate: SI-SNR is scale-invariant, so
   `L_sisnr` barely moves (measured `Δ = −0.0045` for a ×10 rescale) — do not write that test.
10. `gradients` — `est.requires_grad_(True)` on `randn(2, 257, 63, 2)`: `total.backward()`,
    `torch.isfinite(est.grad).all()`, `est.grad.abs().max() > 0`.
11. `scheduler_trace` — `ReduceLROnPlateau(Adam([p], lr=1e-3), mode="min", factor=0.5,
    patience=5)` stepped 13 times on a constant `1.0`: lr sequence
    `[1e-3]×6, [5e-4]×6, [2.5e-4]`; and `pytest`-style `TypeError` check that `verbose=` is
    rejected by this torch build.
12. `frozen_erb` — one Adam step on a synthetic batch leaves `erb.erb_fc.weight` bit-identical;
    trainable tensor count `203`, trainable params `23,669`, total `48,245`.
13. `guards` — 3-D input, complex input, `NaN` in `ref`, empty batch: each raises `ValueError`.
14. `oracle_equivalence` — independent check against the pre-existing
    `speech_denoising/losses/gtcrn.py:GTCRNHybridLoss` (which takes waveforms and applies the
    official `30/70/1` scale, `:143-147`). Build `x = randn(2, 16000)` and `y = x + 0.1·randn`,
    analyse both with the **same** sqrt-Hann STFT, then assert
    `abs(HybridLoss(stft(y), stft(x)) − 30·(ri+imag)... − GTCRNHybridLoss(y, x)) < 1e-3` — i.e.
    compare `paper·100` against the oracle. Measured on one real utterance: `79.526039` vs
    `79.526039`, diff `0.000e+00`, so a `1e-3` tolerance only catches an actual porting error.
    This test does **not** bless the waveform interface (§2.6): it deliberately feeds the oracle
    the same re-analysed spectrum the oracle would compute itself.

### 4.8 Smoke commands (CPU only, no GPU, no real training)

```bat
:: from D:\Project\denoising, PowerShell
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.conda\python.exe scripts\smoke_gtcrn_paper.py
```

```bat
:: 3-step fixture: synthetic audio, CPU, one epoch, writes only under results/paper_runs
.\.conda\python.exe scripts\train_gtcrn_paper.py --fixture --steps 3 --device cpu ^
  --epochs 1 --segment-seconds 1.0 --run-dir results\paper_runs\_fixture_gtcrn_paper
```

`--fixture` must use `torch.randn` audio (not `data/`), force `device = cpu`, and assert the loss
is finite at every step; it exists so the loop, the scheduler call, `run.json` and the checkpoint
round-trip can be verified without touching the corpus or the GPU. No real-data command belongs in
this document's executable set — the real run needs operator-supplied `--epochs` and
`--segment-seconds` (`§3.4`, `§4.4`).

### 4.9 Order of work and gates

| Step | Work | Gate |
|---|---|---|
| 1 | `paper_gtcrn.py`: frontend + `GTCRNPaper` | smoke 1–4 pass; `state_dict` keys equal `GTCRNModel`'s |
| 2 | `HybridLoss` | smoke 5–10 and 14 pass; official-code cross-check (6) within `1e-4`; oracle agreement (14) within `1e-3` |
| 3 | optimizer/scheduler wiring | smoke 11–13 pass; frozen-ERB check (12) |
| 4 | data selection + loop + run artifacts | fixture run of 3 steps; `run.json` contains `epochs_source` and `segment_source` |
| 5 | resume path | fixture resume reproduces the rng and the next-epoch lr |
| 6 | hand-off to operator | real run only after `--epochs`/`--segment-seconds` are chosen and recorded |

---

## 5. Unresolved unknowns and their evidence

| # | Unknown | Evidence it is unknown | How the plan handles it |
|---|---|---|---|
| 1 | VCTK epoch budget | `configs/paper_recipes.yaml:35` (`epochs_vctk: null`), `docs/paper_alignment.md:47`, no trainer in the official repo | `epochs: null`; CLI required; `epochs_source: operator` |
| 2 | Segment length / crop | `configs/paper_recipes.yaml:36-39` ("no … segment length") | `segment_seconds: null`; CLI required; option table §4.4 |
| 3 | Weight decay | same lines | `0.0` (Adam default), recorded |
| 4 | Adam `betas`, `eps` | paper silent; §3.1 | torch defaults, recorded |
| 5 | Gradient clipping value | `SEtrain/trainer.py` clips, value lives in a DNS3 `config.toml`; paper silent | `grad_clip_norm: null`; never copied across loss scales (§2.4) |
| 6 | Validation set identity | paper silent; repo has a 794-utterance speaker-disjoint split | `train_pool` switch; monitor purpose and in-sample flag in `run.json` |
| 7 | Train pool 11,572 vs 10,778 | paper says 11,572 train; repo unified uses 10,778 | `paper_full` default, `held_out` alternative |
| 8 | Batch-4 padding policy | official loss has no mask; paper silent | option table §4.4, crop default |
| 9 | Loss scale (paper vs ×100) | official code ×100 of the paper weights | `loss_scale` recorded; paper default |
| 10 | Paper's own loss values / epoch count in text | `.research/gtcrn.pdf` is an HTML paywall stub; no `.txt`, no `loss.py` in `.research/` | cannot be settled locally; do not fabricate; re-fetch the PDF with institutional access if a number is ever needed |
| 11 | Whether the paper's SI-SNR used `length=` in ISTFT | official `loss.py` passes none | keep the truncation; document ≤255 lost samples |
| 12 | Paper's reported scores as targets | README Table 1 (SISNR 18.83 / PESQ 2.87 / STOI 0.940) is the **upstream README**, not a local measurement | use as a target only after an actual run; do not pre-register them as expected |

---

## Verdict

The paper track is implementable today with **zero new data and zero changes to existing files**.
The model needs a spectrum-level wrapper (the repo's `forward` returns waveform only), the loss
must be the official per-signal-normalized hybrid with `1e-12` inside the sqrt and an
`−log10` SI-SNR term, the optimizer/scheduler port is exact except that torch 2.11 no longer
accepts `verbose=` and patience fires one epoch later than the prose suggests, and the epoch
budget genuinely does not exist in any available source — it must be an operator input, recorded
as such.

## Safe next step

Implement steps 1–2 of §4.9 only (frontend + loss) behind `scripts/smoke_gtcrn_paper.py`; both are
CPU-only, data-free, and reversible. Do not create a run directory, do not touch `results/`, and
do not pick an epoch count or a segment length to "get started" — those two numbers are the
entire fidelity question.

## Five-line summary

1. Repo `GTCRNModel.forward` returns waveform only (`model.py:20-43`); the paper loss needs
   `GTCRNNetwork`'s `[B,257,T,2]` spectrum, so a thin `GTCRNPaper` wrapper (spectrum in/out, same
   `network.*` state-dict keys) is the only model-side change.
2. The loss is `0.01·L_sisnr + 0.7·L_mag + 0.3·L_ri`, with `|Ŝ|=sqrt(Ŝr²+Ŝi²+1e-12)`,
   **per-signal** `mag^0.7` normalization (not clean-only), `MSE(|·|^0.3)`, and
   `−log10` SI-SNR on `(T−1)·256`-sample ISTFTs; the official code is exactly 100× this.
3. Optimizer is `Adam(1e-3)` on the 203 trainable tensors / 23,669 params, and the schedule is
   `ReduceLROnPlateau(min, factor=0.5, patience=5)` stepped on validation loss — `verbose=` must be
   dropped on torch 2.11, and the first drop lands on the 7th flat epoch.
4. No VCTK epoch budget exists in the paper, in `paper_recipes.yaml:35`, or in the official repo
   (no trainer at all); `--epochs` and `--segment-seconds` are required operator inputs, recorded
   in `run.json` as non-paper values.
5. Implement as new files only (`configs/gtcrn_paper.yaml`,
   `speech_denoising/training/paper_gtcrn.py`, two `scripts/*_gtcrn_paper.py`), verified by 14 planned
   synthetic-tensor checks (plus a 15th aligned-crop check added after review) and a CPU 3-step `--fixture` run, leaving the unified runner and all
   historical reports untouched.
