# TF-GridNet DNS external pretrained integration

## Identity and scope

Candidate: `Zhaoheng/tfgridnet_dns_ins20_epoch33`
HF revision: `667c73df547006f47513f2b31c1709bdac63f79c`
File: `exp/enh_train_enh_tfgrid_raw/33epoch.pth`
Bytes: 10,332,558
SHA-256: `613db9fb4dafa7860d9e1390ec8f6bb61bba62cc0a67482bebcfbd0221d865c3`

This is original TF-GridNet, not TF-GridNetV2/V3. The published DNS profile
uses one input channel, one output, FFT=512/hop=256, Hann window, four blocks,
hidden=128, four attention heads, embedding=32, embedding kernel/hop=4/4,
approximate QK dimension=512, PReLU and eps=1e-5. It has **2,552,790**
parameters and **362** state entries. The existing six-block default remains
8,381,504 parameters and is never silently reconfigured by this loader.

`configs/tfgridnet_dns.yaml` records the separate profile; `profiles.py` is
the programmatic constructor. The YAML is a profile record, not a general
config parser. `build_model("tfgridnet")` still returns the six-layer default.

The Phase 2.8 audit in `docs/tfgridnet_checkpoint_audit.md` is retained as a
historical record. Its "next gate" describes the state before this integration.
No training or VoiceBank quality evaluation has been performed in Phase 2.9.

## Provenance and sample rate

The official ESPnet DNS 2020 recipe links this candidate. Published training
and validation paths are `tr_synthetic` and `cv_synthetic`. This is an external
pretrained baseline; it is not a matched-training comparison with GTCRN or
LiSenNet. Exact training manifests and checkpoint-selection logs are absent.
Independent verification of training overlap with VoiceBank is therefore absent.

The uploaded config has `sample_rate:8000`, `preprocessor:null`, and outer
`encoder:same` / `decoder:same`. The preprocessing sample-rate argument is not
passed into the separator when preprocessing is disabled. The official recipe
defaults to `--fs 16k`; the integration uses mono 16 kHz without resampling.
This explains the chosen integration rate but does not independently establish
the exact original training audio rate. Keep this caveat in later reports.

Pinned config SHA-256:
`ff6979552f066e52ca09f5eb1f77d33ff788afe09e6a4d5d38e0b284d015a1e1`
Pinned meta.yaml SHA-256:
`a9a6524ee608f781e4a8537698396e1cce478116f824d8a893230f1c663ff8a9`

The HF card declares `bsd-2-clause`; its exact 30-byte content is retained as
`CHECKPOINT_CARD.upstream.md`. A full weight-license notice was not present
in the inspected repository. This declaration is distinct from the ESPnet
Apache-2.0 code license retained in `LICENSE.upstream`.

## Loading and inference

```python
from speech_denoising.models.tfgridnet.profiles import build_tfgridnet_dns_model
from speech_denoising.models.tfgridnet.checkpoint import load_tfgridnet_dns_checkpoint

model = build_tfgridnet_dns_model().eval()
metadata = load_tfgridnet_dns_checkpoint(model, "checkpoints/tfgridnet_dns_epoch33.pth")
enhanced = model(noisy_audio)  # mono CPU FP32 [batch, samples], 16 kHz
```

The loader verifies byte size and SHA-256 before `torch.load(weights_only=True,
map_location="cpu")`. It accepts only the raw 362-entry `separator.*` state,
removes only that exact prefix, and checks all keys, shapes, FP32 dtypes and
finite values before assigning any tensors. `load_state_dict(strict=True)` is
mandatory. Missing, extra, nonfinite and mismatched states are errors.
Load into CPU FP32 before moving/casting. No partial loading or random fallback.

The adapter uses the existing internal sample-standard-deviation normalization
and reverses it at output. No extra peak/volume normalization is added.
Whole-utterance bidirectional LSTMs and global attention make this offline.
Exact silence maps to exact silence; inputs shorter than 768 samples are
right-padded then cropped. These finite/alignment guards extend the upstream
domain and are documented rather than claimed as unmodified native behavior.

## Independent reference and limits

The model's metadata declares ESPnet `202308`. The recipe's reported training
commit `60ce18efa06ca5a5922534682f47e2107ef88b13` could not be retrieved from
the public ESPnet repository during this integration. Instead, the smoke check
pins official **v.202308**, commit
`01d7df74f711758edab1d90b24e2f0859e0b31da`. This is version-matched release
reference equivalence, not proof of equivalence to every local modification
in the original training checkout or reproduction of published quality scores.

`scripts/_tfgridnet_dns_reference.py` embeds only necessary Apache-2.0 sources:
the original separator, its own STFT encoder/decoder and Stft implementation,
activation lookup, and necessary complex utility functions. Each complete
source payload is checksum-verified before execution. Package interfaces and
the padding-mask utility are provided locally; training/loss/task modules and
the complete original repository are not copied.

The reference preserves default `use_builtin_complex=False` and uses
`torch-complex==0.4.4`. The old release sends stacked real/imaginary values to
iSTFT, which modern PyTorch rejects. A test-only compatibility bridge converts
that representation to builtin complex solely at the iSTFT API boundary;
reference class/function computation bodies remain unchanged. Imports for the
unused pre-PyTorch-1.10 librosa fallback and runtime typeguard checks are removed.
This compatibility bridge is not used by the production adapter.

The non-segmented native inference path is identity encoder -> separator ->
identity decoder, with `normalize_output_wav=False`. It performs no extra
preprocessing, segmentation or peak scaling for this configuration.

The CPU FP32 smoke check loads the actual pinned tensors independently into
the adapter and release reference. It checks single/batched 768, 1025, 16000
and 16001-sample inputs with rtol=1e-5 / atol=1e-6, plus batch independence,
gain, padded short-audio parity, silence and load rejection. It writes a
synthetic integration report, with no quality metrics. A deprecation warning
for `stft(return_complex=False)` comes from the historical reference path.

## Windows commands

After the Phase 2.9 source installer, from the repo root:

```bat
python -m pip install torch==2.14.1 --index-url https://pypi.org/simple
python -m pip install -r requirements-tfgridnet.txt --index-url https://pypi.org/simple
python -m scripts.fetch_tfgridnet_checkpoint
python -m scripts.smoke_tfgridnet_pretrained
python -m scripts.smoke_tfgridnet
```

The first installation command selects the standard Windows wheel rather than
depending on the CPU index that previously failed certificate verification.
Run this integration on CPU regardless of GPU availability. Windows inference
must be verified locally; Linux validation is not a Windows-pass claim.

For an actual input WAV, optionally run:

```bat
python -m scripts.inference_tfgridnet --input "C:\path\to\noisy.wav" --output results/tfgridnet_demo/enhanced.wav
```

Use an actual mono 16k PCM16 file path. The script reuses the existing shared
WAV I/O, records sample alignment and clipping count, and produces no quality
scores. It does not download audio. WAV quantization/clipping occurs only during
export, not inside the float waveform model used by the shared evaluator.

`scripts.evaluate` stays unchanged and restricted to GTCRN in this phase.
Any later TF-GridNet quality evaluation must use the common
`evaluate(model, dataset)` metrics, the fixed 824-test manifest and explicit
checkpoint/profile metadata. No synthetic check is a dataset benchmark.

## Primary sources

- HF: https://huggingface.co/Zhaoheng/tfgridnet_dns_ins20_epoch33/tree/667c73df547006f47513f2b31c1709bdac63f79c
- Release reference: https://github.com/espnet/espnet/tree/01d7df74f711758edab1d90b24e2f0859e0b31da
- Recipe: https://github.com/espnet/espnet/blob/158e4959d17d41e544d86ac6bd85db45f838de7a/egs2/dns_ins20/enh1/README.md
- Source hashes and workflow: `docs/tfgridnet_dns_pretrained_integration.json`
