"""Unified training loss contract."""

from torch import nn


class EnhancementLoss(nn.Module):
    def forward(self, enhanced_audio, clean_audio):
        raise NotImplementedError("Phase 3: select and implement the shared loss.")
