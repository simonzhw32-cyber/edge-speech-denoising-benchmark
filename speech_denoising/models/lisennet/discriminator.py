"""PESQ-conditioned discriminator from the official LiSenNet implementation.

Ported from ``hyyan2k/LiSenNet`` at commit ``df4e481`` (MIT, see
``LICENSE.upstream``). The discriminator scores a clean/enhanced compressed
magnitude pair and is trained against PESQ rescaled to ``(0, 1)``.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class Discriminator(nn.Module):
    """Four spectral-normalized conv blocks followed by a scalar PESQ head."""

    def __init__(self, ndf=16, in_channel=2):
        super().__init__()
        self.layers = nn.ModuleList([
            spectral_norm(nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(ndf * 2),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(ndf * 4),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(ndf * 8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(ndf * 4),
            spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        ])

    def forward(self, clean_mag, enhanced_mag):
        pair = torch.stack([clean_mag, enhanced_mag], dim=1)
        outputs = []
        for layer in self.layers:
            pair = layer(pair)
            outputs.append(pair)
        return outputs
