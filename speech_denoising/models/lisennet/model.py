"""16 kHz waveform adapter for the official LiSenNet generator."""

import numpy as np
import torch
from torch.nn import functional as F

from ..base import BaseEnhancementModel
from .network import LiSenNet as LiSenNetNetwork


class LiSenNetModel(BaseEnhancementModel):
    """Whole-utterance enhancement with upstream normalization and phase recovery.

    Weights are random until explicitly loaded. No checkpoint is bundled.
    The centered STFT and two Griffin-Lim iterations are an offline protocol,
    not a claim of stateful streaming inference.
    """

    sample_rate = 16000

    def __init__(self, num_channels=16, n_blocks=2, n_fft=512,
                 hop_length=256, compress_factor=0.3):
        super().__init__()
        if n_fft != 512:
            raise ValueError("Official LiSenNet frequency layers require n_fft=512")
        if not isinstance(num_channels, int) or num_channels < 4 or num_channels % 4:
            raise ValueError("num_channels must be a positive multiple of four")
        if not isinstance(n_blocks, int) or n_blocks < 1:
            raise ValueError("n_blocks must be a positive integer")
        if hop_length != 256 or compress_factor != 0.3:
            raise ValueError("This adapter pins the upstream hop=256 and compression=0.3 protocol")
        self.network = LiSenNetNetwork(num_channels=num_channels, n_blocks=n_blocks,
                                      n_fft=n_fft, hop_length=hop_length,
                                      compress_factor=compress_factor)

    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        if not isinstance(noisy_audio, torch.Tensor):
            raise TypeError("noisy_audio must be a torch.Tensor")
        if noisy_audio.ndim != 2 or min(noisy_audio.shape) < 1:
            raise ValueError("Expected nonempty mono [batch, samples] audio")
        if noisy_audio.dtype not in (torch.float32, torch.float64):
            raise TypeError("Audio must use float32 or float64")
        parameter = next(self.network.parameters())
        if noisy_audio.device != parameter.device or noisy_audio.dtype != parameter.dtype:
            raise ValueError("Audio and model must use the same device and dtype")
        if not torch.isfinite(noisy_audio).all():
            raise ValueError("Audio contains NaN or infinity")

        length = noisy_audio.shape[-1]
        # Match the official numpy preprocessing, including its reduction order.
        # Tiny torch.std rounding differences can cross wrapped-phase boundaries.
        # The loader's scale is a preprocessing constant, outside autograd.
        scale_array = np.std(noisy_audio.detach().cpu().numpy(), axis=-1,
                             keepdims=True) + 1e-8
        scale = torch.from_numpy(scale_array).to(noisy_audio.device)
        normalized = noisy_audio / scale
        # PyTorch's centered reflect-STFT requires more than n_fft // 2 samples.
        # This explicitly defined extension applies only to very short inputs.
        if length <= self.network.n_fft // 2:
            normalized = F.pad(normalized, (0, self.network.n_fft // 2 + 1 - length))
        enhanced = self.network(normalized)["est"]
        return enhanced[..., :length] * scale


# Preserve the Phase 1 registry import: from .lisennet.model import LiSenNet.
LiSenNet = LiSenNetModel

__all__ = ["LiSenNetModel", "LiSenNet"]
