# TF-GridNet source and local enhancement profile

Upstream: ESPnet, https://github.com/espnet/espnet
Pinned revision: `158e4959d17d41e544d86ac6bd85db45f838de7a`.
Variant: **original `tfgridnet`**, not `tfgridnetv2` or `tfgridnetv3`.
Main source: https://github.com/espnet/espnet/blob/158e4959d17d41e544d86ac6bd85db45f838de7a/espnet2/enh/separator/tfgridnet_separator.py
Reference: Z.-Q. Wang et al., TF-GridNet: Integrating Full- and Sub-Band Modeling
for Speech Separation, https://arxiv.org/abs/2211.12433 .

License: Apache-2.0. The full upstream license is in `LICENSE.upstream`.
Its appendix attributes Copyright 2017 Johns Hopkins University (Shinji Watanabe).
ESPnet contributors and the TF-GridNet authors retain attribution. This is a
derived extraction, not an official author distribution or whole-repository copy.

## Pinned file SHA-256

- `espnet2/layers/stft.py`: `827763e26be6ede1556414d6918763385774bfa39afbc6229267fbe67980d7d6`
- `espnet2/enh/encoder/stft_encoder.py`: `bb76ea25b4cb02b26bc668887f540d5952c714c746ebf6cfc9299723b807ff45`
- `espnet2/enh/decoder/stft_decoder.py`: `4187de232932da18804392b08bf8de3dabdcbf7bca39c6c9e1a46bed10ea8e79`
- `espnet2/torch_utils/get_layer_from_string.py`: `f8821d9566f326f75e69339eca18fa3b84d81b84915502dddbfe64b30dc0884e`
- `espnet2/enh/separator/tfgridnet_separator.py`: `e08616b3e1964c7b0019320fff47110a4708fff44752173faff95aefed50e1f6`
- `LICENSE`: `4696c3c9551da6fef1368be1e4ed2c80cf13e55448c6dcf2aba9462f5ff29ef5`

## Local profile

16 kHz; one input microphone, one output source. Hann STFT n_fft=512, hop=256,
center=True, reflect padding, normalized=False, onesided=True. Six blocks,
bidirectional LSTM hidden=192, four attention heads, approximate Q/K dimension
512, embedding=48, embedding kernel=4/hop=1, PReLU, normalization epsilon=1e-5.
This is a **local speech enhancement profile with random weights**, not a paper
checkpoint configuration or a claim of reproduced quality. The original source
defaults to two sources and FFT/hop 128/64; these defaults are intentionally
changed for the local mono 16 kHz enhancement task. Parameter count is reported
by `python -m scripts.smoke_tfgridnet` for this exact local profile:
8,381,504 total/trainable parameters; 540 network state entries.

## Extraction and modifications

- Keep the original conv, deconv, GridNetBlock, LSTM, attention, normalization
  operations and learned state names/shapes at matching configuration.
- Replace the abstract ESPnet base with nn.Module, optional imports with local
  equivalents and legacy complex with PyTorch builtin complex tensors.
- Extract only the fixed-rate offline STFT path into `spectral.py`. No ESPnet,
  torch_complex or typeguard runtime dependency. Drop streaming, spectral
  transforms and sampling-rate reconfiguration. The reference test retains
  the original frontend, with small dependency shims for its builtin complex path.
- Preserve ESPnet sample-standard-deviation normalization (torch.std correction=1),
  amplitude restoration and its explicit audio.float() before STFT. FP64 inputs
  keep their interface dtype but the FFT input is quantized to FP32 as upstream.
- Floor standard deviation at 1e-8 for constant inputs. Exact zero input gives
  exact zero output. These are defined extensions of upstream's undefined
  zero-variance case, not changes to normal-input reference comparisons.
- Default inputs shorter than 768 samples are right-padded to 768, ensuring
  reflect padding and at least four STFT frames for the embedding kernel.
  The output is cropped to the original sample count. Normal-input parity is
  tested at lengths >=768; short-input behavior is tested separately.
- The adapter rejects empty, non-mono, non-finite and unsupported dtype inputs,
  and requires audio/model device and dtype agreement. No silent resampling.

## Verification and current limits

The smoke test embeds compressed, SHA-256-checked upstream sources; no download
is needed. It compares independent original/local STFT and iSTFT, strict state
loading, full waveform outputs, batching, odd/short lengths, silence, FP32/FP64,
gradients without an optimizer step, and the shared evaluator using a synthetic
fixture. CUDA reference equivalence runs only if available.

This is an **offline** model: bidirectional LSTMs and frame-global attention
depend on future audio. Centered STFT also uses future samples. It is not a
causal or stateful edge streaming implementation. Long utterances may use
substantial attention memory; chunking would change the protocol and is not
introduced here. No pretrained checkpoint, dataset quality score or training.
The evaluation CLI stays restricted to audited GTCRN pretrained loading.
