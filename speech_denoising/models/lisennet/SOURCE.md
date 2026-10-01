# LiSenNet integration provenance

Upstream: https://github.com/hyyan2k/LiSenNet

Pinned revision: `df4e481aa706fbc344d1013f7a1cbfd46228c8ab`.

License: MIT, copyright 2024 hyyan2k. The original notice is preserved in
`LICENSE.upstream`. The official source snapshots in the smoke test carry the
same notice. This does not assign that license to unrelated benchmark code.

## Extracted files and reviewed inference sources

| Upstream path | SHA-256 of original bytes | Use |
| --- | --- | --- |
| `model/generator/generator.py` | `151e91c1ea7af699239f93530a5121b679fe22a88825106b873556f96625553e` | Enhancement generator and native STFT/phase reconstruction |
| `model/generator/dpr_layer.py` | `2d497284b7d2652cc3cfef5d5b967626739a28d2a6d99c5d754340c26d606c9c` | Dual-path GRU, normalization and convolutional GLU |
| `data_module.py` | `0ffc1a74c9dde2cf66b524f03e3491671db0fc37eed6765e2a70e203af456810` | Review only: per-utterance numpy.std + 1e-8 normalization |
| `config.yaml` | `432b6f92c0c5104439ef099f7de1a0d7ff570eac6a284b7b0db12ba29a56dc04` | Review only: official architecture defaults |
| `LICENSE` | `60f81c335a77fe6023bb4b2b9c03e1e0b224a58a138c5a3c99ef1003c66eef92` | Retained verbatim |

Only necessary model definitions are extracted. No upstream repository is
cloned, forked or copied as a whole. No training module, discriminator, original
dataset loader or upstream evaluation code is installed.

## Local changes

- `network.py` retains the generator, encoder/decoder, compression, wrapped
  phase features and two-iteration Griffin-Lim path. Parameter names and tensor
  shapes match the original generator exactly.
- The unused standalone NoiseDetector class and unused mel_scale method/import
  are omitted. The official generator forward does not invoke either. This
  integration makes no claim about adaptive noise gating.
- `dpr_layer.py` retains the official implementation, with a trailing newline.
- STFT windows and phase-difference zero tensors follow the input dtype/device;
  the optional target test uses `is None`. Float32 inference matches the pinned
  upstream reference.
- `LiSenNetModel` implements the common 16 kHz mono `[batch, samples]` interface,
  using the exact numpy.std preprocessing, then restoring the original amplitude.
  No clean/reference audio is used to enhance a noisy waveform.
- The normalization scale is a detached preprocessing constant, as in the
  original NumPy data loader. Model parameters and the waveform division still
  support backpropagation. No optimizer step is performed by the smoke test.
- For inputs of 256 samples or fewer, normalize first, zero-pad to 257 samples,
  then trim the output. This explicitly defined extension avoids the reflect-
  STFT restriction; upstream does not define such short-input behavior.

## Native inference protocol

Defaults: 16 channels, two DPR blocks, FFT=512, hop=256, periodic Hann window,
centered reflect STFT, power compression=0.3, two Griffin-Lim iterations,
momentum=0.99. The generator has 36,783 total/trainable parameters.

NumPy normalization is deliberate: tiny rounding differences from torch.std
can cross wrapped-phase boundaries and change the network features. CPU is the
validated baseline. A GPU call copies its waveform to CPU to compute the same
normalization constant; this internal adapter transfer is included in the
adapter's elapsed time. Future GPU efficiency protocols must disclose this.

This is whole-utterance inference. Centered STFT and Griffin-Lim require future
context; causal layers do not make this adapter a stateful streaming system.

## Verification and current limits

Run `python -m scripts.smoke_lisennet`:

- Strictly map all state keys/shapes into the original generator reference.
- Compare generator outputs and end-to-end NumPy-normalized waveform outputs.
- Verify odd/short lengths, silence, batches, FP32/FP64 and invalid inputs.
- Verify finite waveform/parameter gradients without updating weights.
- Check the existing shared `evaluate(model, dataset)` using one synthetic
  fixture; discard its scores and do not create a dataset benchmark report.

The test embeds compressed, hash-checked original generator/DPR snapshots;
only the unused torchaudio import is suppressed when loading that reference.
It needs no downloads, torchaudio or PyTorch Lightning installation.

No checkpoint is bundled or downloaded. The pinned official repository tree
contains no checkpoint file; that does not establish whether one is available
elsewhere. Loading pretrained weights needs a separate source/provenance audit,
strict generator-key mapping and reference-output check before quality scoring.
The dataset evaluation CLI remains gated to the existing pretrained GTCRN path.
Random-weight LiSenNet outputs must not be reported as a quality baseline.

Next: audit an available checkpoint, verify its training/data recipe and exact
architecture, then enable LiSenNet in the same evaluation CLI. Unified training,
losses and validation stay in Phase 3, with test excluded from selection.
