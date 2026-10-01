"""Waveform interface shared by every model adapter."""

import torch
from torch import nn


class BaseEnhancementModel(nn.Module):
    sample_rate = 16000

    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """Map mono [batch, samples] audio to an aligned tensor of the same shape."""
        raise NotImplementedError
