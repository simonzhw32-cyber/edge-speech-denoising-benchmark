# GTCRN integration — Phase 2.2

Source: https://github.com/Xiaobin-Rong/gtcrn
Pinned revision: 502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73
Model source: gtcrn.py; waveform frontend reference: infer.py.
License: MIT, Copyright (c) 2024 Rong Xiaobin; retained in LICENSE.upstream.

Only network definitions were extracted. The upstream executable profiling and
causality example was removed. Channel rearrangement uses equivalent PyTorch
flatten instead of einops. GRU initial hidden states inherit the input dtype.
Architecture and native network state-dict names are preserved.

The benchmark adapter uses 16 kHz mono [batch, samples], sqrt-Hann window,
n_fft=512, hop=256, centered reflect-padding STFT, complex masking and iSTFT.
Modern complex tensors replace deprecated upstream real/imag STFT APIs.
Explicit reconstruction length preserves non-hop-aligned utterance lengths.
Inputs shorter than 257 samples receive right-zero-padding before STFT and
are trimmed after reconstruction. The wrapper is whole-utterance inference;
it does not claim stateful streaming or an end-to-end zero-lookahead frontend.

The GTCRN alias preserves the existing build_model import. State-dict keys
inside the adapter have a network. prefix. Checkpoint loading/mapping is a
future step. This patch uses random weights and cannot demonstrate denoising
quality. No checkpoints, data, training code or upstream evaluation code
are included. LiSenNet and TF-GridNet remain placeholders.

Test: python -m scripts.smoke_gtcrn
Checks waveform shape/finite output, short/odd lengths, batch independence,
upstream frontend equivalence, gradient flow, and invalid input handling.
No dataset, checkpoint, optimizer, or training is used.
