# TF-GridNet pretrained checkpoint audit

Audit date: 2026-10-03 (Asia/Shanghai). Baseline: `32c1b59`.
The companion JSON records pinned revisions, source hashes, candidate decisions,
binary verification and a shared tensor inventory. This is an audit result;
no TF-GridNet pretrained model has been installed in the benchmark yet.

## Result

**Prefer `Zhaoheng/tfgridnet_dns_ins20_epoch33` for the next pretrained integration.**
The ESPnet DNS Interspeech 2020 recipe explicitly links this model. Its published
config and actual separator tensor names/shapes match an original TF-GridNet
single-microphone, single-output enhancement profile. It is not compatible with
our current random-weight 6-layer default, so a separately named pretrained
profile is required. No forced partial loading, resized tensors or silent
architecture replacement is appropriate.

| Item | Current local profile | Published DNS / MS-SNSD profiles |
|---|---|---|
| Variant | Original tfgridnet | Original tfgridnet |
| Input / output | 1 / 1 | 1 / 1 |
| FFT / hop | 512 / 256 | 512 / 256 |
| Blocks | 6 | 4 |
| LSTM hidden | 192 | 128 |
| Embedding | 48 | 32 |
| Embedding kernel / hop | 4 / 1 | 4 / 4 |
| Parameters | 8,381,504 | 2,552,790 |
| State entries | 540 | 362 |

This distinction matters for both checkpoint loading and parameter/RTF reports.
Published recipe performance is not copied into our VoiceBank score table.

## Preferred candidate: DNS Interspeech 2020

- Model: https://huggingface.co/Zhaoheng/tfgridnet_dns_ins20_epoch33
- Pinned revision: `667c73df547006f47513f2b31c1709bdac63f79c`.
- Weight: `exp/enh_train_enh_tfgrid_raw/33epoch.pth`.
- Verified size: 10,332,558 bytes.
- Verified SHA-256: `613db9fb4dafa7860d9e1390ec8f6bb61bba62cc0a67482bebcfbd0221d865c3`.
- Config SHA-256: `ff6979552f066e52ca09f5eb1f77d33ff788afe09e6a4d5d38e0b284d015a1e1`.
- Model card declares BSD-2-Clause; no separate full weight-license file was
  present in the inspected model tree. ESPnet implementation is separately
  Apache-2.0. Retain the card, exact source and applicable notices when integrating.
- Training artifact metadata records ESPnet 202308 / PyTorch 2.1.0+cu118;
  the official recipe result lists source revision
  `60ce18efa06ca5a5922534682f47e2107ef88b13`.

We downloaded the two approximately 10 MB original-profile files only into the
audit environment for SHA-256 and metadata checks. Safe deserialization used
`torch.load(weights_only=True, map_location="meta")`. The DNS container is an
OrderedDict with 362 tensor entries, all beginning with `separator.`. Removing
only that prefix gives zero missing names, unexpected names or shape mismatches
against the **published** profile. No tensors were assigned to a running model.
Against the current default, 178 names are missing and 213 shared shapes differ.
This proves structural compatibility with the alternative profile, not strict
pretrained loading, finite weight values, inference parity or denoising quality.

The pinned ESPnet DNS recipe uses `tr_synthetic` for training and `cv_synthetic`
for validation. Its data-preparation script divides the generated training list
90/10; test data is prepared separately. The uploaded config lists validation
loss and SI-SNR selection criteria, but the checkpoint-specific reason for
choosing epoch33 and the exact original audio manifests are not published.
These observations do not establish independent no-overlap proof against our
VoiceBank test audio. They also provide no evidence of VoiceBank test selection.

## Sample-rate interpretation

Both original-profile configs contain `sample_rate: 8000` and 16k stats paths.
Neither field alone proves the native audio rate. Both also set
`preprocessor: null`; the inspected ESPnet task returns no preprocessing and
does not pass this `sample_rate` argument to `build_model`. Their official
`run.sh` recipes default to `sample_rate=16k` and pass that to data preparation.
Thus the recipes support a 16 kHz integration, while exact checkpoint training
audio manifests remain unverified. Do not automatically label the weights as
8 kHz or silently resample based on this dormant configuration field.

## Secondary candidate: MS-SNSD

- Model: https://huggingface.co/espnet/ms_snsd_tfgridnet
- Pinned revision: `49144db068f74542487eb23c0b6fbe63b67bfdd0`.
- Weight: `valid.loss.best.pth`.
- Verified size: 10,332,955 bytes.
- Verified SHA-256: `a680263b1339d86ddc5d168a5f985fbd0b3e658d7711971f3c03fe2d23af7bac`.
- Config SHA-256: `2f2763974aa83cd1831be67aafed5f8ebd03795e12462d1807d637daf7c6c8ea`.

It has the same 362-entry, 2,552,790-parameter published architecture and matching
tensor metadata. Its config points to `tr_ms_snsd` / `cv_ms_snsd`, and its card
states MS-SNSD training. It is a secondary candidate. The inspected card does
not declare a weight license; do not infer a checkpoint license solely from the
Apache-2.0 code license. Exact training manifests and pretrained output parity
also remain unverified.

## Other inspected public candidates

| Candidate | Finding | Current decision |
|---|---|---|
| kohei0209/tfgridnet_urgent25 | Challenge-official enhancement; V3 with adaptive external STFT, multi-rate pipeline | Requires a separate V3 variant |
| wyz/tfgridnet_for_urgent24 | V3; actual epoch file distinct from an 11-byte best.pth entry | Requires separate variant and native pipeline |
| wyz/vctk_dns2020_whamr_tfgridnet_small / tiny | V3, mixed VCTK-DEMAND/DNS/WHAMR data | Separate variant and split/selection audit |
| espnet/enh_train_enh_tfgridnet_tf_lr-patience3_patience5_raw | Pinned repository contains only .gitattributes; related WSJ recipe is two-output separation | No usable weights in this inspected repository |
| niobures/TF-GridNet and FushiXie/TF-GridNet | Same WHAMR-named file hash; no inspected config/card/license | Insufficient metadata for integration |

V3 is not merely another checkpoint for the integrated original architecture.
Multi-rate STFT, normalization and inference output scaling need their own
protocol. A declared VCTK training corpus does not itself prove test leakage;
the source partitions and selection logs would need examination. Conversely,
published model-card scores do not substitute for our fixed 824-test evaluation.

## Integration gate and next step

1. Add a distinct DNS pretrained profile with the published 4-layer dimensions.
   Keep the original 6-layer random-weight profile explicit.
2. Download the pinned file with the verified hash; use weights-only safe
   loading and strict state matching. Validate tensor finiteness and every key.
3. Compare preprocessing and whole-waveform output with its historical native
   ESPnet implementation, including standard-deviation normalization, STFT,
   iSTFT length handling and any output amplitude normalization.
4. Record all unresolved training-manifest/selection provenance. Any later
   VoiceBank evaluation belongs to an **external pretrained baseline** track.
   It is not a matched-training comparison with GTCRN or LiSenNet.
5. Run the same shared metrics on the same test manifest, without tuning or
   choosing checkpoints on those 824 test utterances.

Current status: audit complete; pretrained integration and quality evaluation
pending. No optimizer step, training, TF-GridNet inference or score generation.
The user installer transfers only this audit, README and recovery documentation;
it does not transfer the checkpoint inspection files.

## Evidence boundaries

The search was bounded: a public HF query returned 20 model names, with nine
repositories examined in detail, plus pinned ESPnet recipes. This is not a claim
that no other eligible weights exist. Binary hashes were personally verified
only for the two original-profile candidates. Other weight hashes in the JSON
are published LFS metadata, not downloaded-content verification.

Source recipe snapshot:
https://github.com/espnet/espnet/tree/158e4959d17d41e544d86ac6bd85db45f838de7a/egs2/dns_ins20/enh1
Related sample-rate/task code:
https://github.com/espnet/espnet/blob/158e4959d17d41e544d86ac6bd85db45f838de7a/espnet2/tasks/enh.py
URGENT baseline source:
https://urgent-challenge.github.io/urgent2025/baseline/
