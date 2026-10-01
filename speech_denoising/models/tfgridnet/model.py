"""TFGridNet adapter placeholder; no upstream model code has been copied."""

from ..base import BaseEnhancementModel


class TFGridNet(BaseEnhancementModel):
    def forward(self, noisy_audio):
        raise NotImplementedError("Phase 2: integrate TFGridNet waveform inference.")
