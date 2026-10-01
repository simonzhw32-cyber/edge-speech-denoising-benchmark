"""LiSenNet adapter placeholder; no upstream model code has been copied."""

from ..base import BaseEnhancementModel


class LiSenNet(BaseEnhancementModel):
    def forward(self, noisy_audio):
        raise NotImplementedError("Phase 2: integrate LiSenNet waveform inference.")
