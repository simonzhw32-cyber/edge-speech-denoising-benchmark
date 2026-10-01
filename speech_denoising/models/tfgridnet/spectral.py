"""Reduced offline ESPnet-compatible STFT path using only PyTorch.

Apache-2.0 derived implementation; see LICENSE.upstream and SOURCE.md.
Supports the fixed-rate, centered, one-sided, unnormalized frontend used here.
Streaming, resampling, spectral transforms and legacy complex are omitted.
"""

import torch
from torch import nn


class STFTEncoder(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, window="hann",
                 use_builtin_complex=True):
        super().__init__()
        if not use_builtin_complex:
            raise ValueError("Only PyTorch builtin complex is supported")
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.window = window

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, audio, lengths):
        batch_size = audio.shape[0]
        multi_channel = audio.ndim == 3
        if multi_channel:
            audio = audio.transpose(1, 2).reshape(-1, audio.shape[1])
        window = getattr(torch, self.window + "_window")(
            self.win_length, dtype=audio.dtype, device=audio.device)
        # ESPnet explicitly casts the FFT input to FP32. Keep that operation,
        # including for FP64, then restore its stored spectrum dtype.
        spectrum = torch.stft(audio.float(), n_fft=self.n_fft,
                              win_length=self.win_length,
                              hop_length=self.hop_length, window=window,
                              center=True, normalized=False, onesided=True,
                              return_complex=True)
        parts = torch.view_as_real(spectrum).to(audio.dtype).transpose(1, 2)
        if multi_channel:
            parts = parts.view(batch_size, -1, parts.shape[1], parts.shape[2], 2)
            parts = parts.transpose(1, 2)
        frame_lengths = torch.div(lengths, self.hop_length, rounding_mode="trunc") + 1
        shape = [parts.shape[0], parts.shape[1]] + [1] * (parts.ndim - 2)
        indices = torch.arange(parts.shape[1], device=parts.device)[None, :]
        mask = (indices >= frame_lengths.to(parts.device)[:, None]).view(shape)
        parts = parts.masked_fill(mask, 0.0)
        return torch.complex(parts[..., 0], parts[..., 1]), frame_lengths


class STFTDecoder(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, window="hann"):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.window = window

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, spectrum, lengths):
        if spectrum.ndim != 3 or not torch.is_complex(spectrum):
            raise ValueError("Expected complex [batch, frames, frequencies]")
        window = getattr(torch, self.window + "_window")(
            self.win_length, dtype=spectrum.real.dtype, device=spectrum.device)
        audio = torch.istft(spectrum.transpose(1, 2), n_fft=self.n_fft,
                            win_length=self.win_length, hop_length=self.hop_length,
                            window=window, center=True, normalized=False,
                            onesided=True, length=int(lengths.max().item()),
                            return_complex=False)
        return audio, lengths
