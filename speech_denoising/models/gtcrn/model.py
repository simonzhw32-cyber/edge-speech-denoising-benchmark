"""16 kHz waveform adapter for the official GTCRN network (random weights)."""

import torch
from torch.nn import functional as F

from ..base import BaseEnhancementModel
from .network import GTCRN as GTCRNNetwork


class GTCRNModel(BaseEnhancementModel):
    sample_rate = 16000
    n_fft = 512
    hop_length = 256

    def __init__(self):
        super().__init__()
        self.network = GTCRNNetwork()
        self.register_buffer("window", torch.hann_window(self.n_fft).sqrt(), persistent=False)

    def forward(self, noisy_audio):
        if noisy_audio.ndim != 2 or min(noisy_audio.shape) == 0:
            raise ValueError("Expected nonempty mono audio [batch, samples].")
        if noisy_audio.dtype not in (torch.float32, torch.float64):
            raise TypeError("Use float32 or float64 audio with matching model dtype.")
        if noisy_audio.device != self.window.device or noisy_audio.dtype != self.window.dtype:
            raise ValueError("Move model and input to the same device and dtype first.")
        length = noisy_audio.shape[-1]
        # Default reflect padding requires length > n_fft/2. Only tiny inputs
        # receive right-zero-padding; ordinary utterances follow upstream infer.py.
        padded = F.pad(noisy_audio, (0, max(0, self.n_fft // 2 + 1 - length)))
        spec = torch.stft(
            padded, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.window, center=True,
            pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
        )
        enhanced_spec = self.network(torch.view_as_real(spec))
        enhanced = torch.istft(
            torch.view_as_complex(enhanced_spec.contiguous()),
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, center=True, normalized=False, onesided=True,
            length=padded.shape[-1],
        )
        return enhanced[..., :length]


# Preserve the existing models/__init__.py import and build_model("gtcrn").
GTCRN = GTCRNModel
