"""Negative SI-SNR over valid samples, averaged per utterance."""

import math

import torch
from torch import nn


class EnhancementLoss(nn.Module):
    def __init__(self, eps=1e-12, reduction="mean"):
        super().__init__()
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if reduction not in {"mean", "none"}:
            raise ValueError("Use mean or none reduction")
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, enhanced_audio, clean_audio, lengths=None):
        """Accept aligned [batch, samples] FP32/FP64 waveforms and integer lengths.

        Omitted lengths mean every sample is valid. Padding is removed before
        centering/projection, including non-finite values in masked tails.
        """
        if not isinstance(enhanced_audio, torch.Tensor) or not isinstance(clean_audio, torch.Tensor):
            raise TypeError("Audio inputs must be tensors")
        if enhanced_audio.ndim != 2 or enhanced_audio.shape != clean_audio.shape:
            raise ValueError("Expected aligned [batch, samples] audio")
        batch, samples = enhanced_audio.shape
        if batch < 1 or samples < 2:
            raise ValueError("Audio must contain nonempty batches and at least two samples")
        if enhanced_audio.dtype not in (torch.float32, torch.float64):
            raise TypeError("Use float32 or float64 audio")
        if clean_audio.dtype != enhanced_audio.dtype or clean_audio.device != enhanced_audio.device:
            raise ValueError("Audio inputs must share dtype and device")
        if lengths is None:
            lengths = torch.full((batch,), samples, dtype=torch.long, device=enhanced_audio.device)
        if not isinstance(lengths, torch.Tensor):
            raise TypeError("lengths must be a tensor")
        if lengths.shape != (batch,) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must be an integer [batch] tensor")
        if lengths.device != enhanced_audio.device:
            raise ValueError("Move lengths and audio to the same device")
        if ((lengths < 2) | (lengths > samples)).any():
            raise ValueError("Each valid length must be between two and the waveform width")
        mask = torch.arange(samples, device=enhanced_audio.device)[None, :] < lengths[:, None]
        # Multiplying by zero would retain NaNs in padded tails.
        estimate = torch.where(mask, enhanced_audio, 0.0)
        reference = torch.where(mask, clean_audio, 0.0)
        if not torch.isfinite(estimate).all() or not torch.isfinite(reference).all():
            raise ValueError("Valid audio samples contain NaN or infinity")
        denominator = lengths.to(dtype=estimate.dtype)[:, None]
        estimate = torch.where(mask, estimate - estimate.sum(dim=-1, keepdim=True) / denominator, 0.0)
        reference = torch.where(mask, reference - reference.sum(dim=-1, keepdim=True) / denominator, 0.0)
        reference_energy = reference.square().sum(dim=-1, keepdim=True)
        if not torch.isfinite(reference_energy).all() or (reference_energy <= self.eps).any():
            raise ValueError("SI-SNR is undefined for silent/constant or overflowing references")
        projection = reference * ((estimate * reference).sum(dim=-1, keepdim=True) / reference_energy)
        residual = estimate - projection
        projected_energy = projection.square().sum(dim=-1)
        residual_energy = residual.square().sum(dim=-1)
        if not torch.isfinite(projected_energy).all() or not torch.isfinite(residual_energy).all():
            raise ValueError("SI-SNR energy overflow")
        values = -10.0 * torch.log10((projected_energy + self.eps) / (residual_energy + self.eps))
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite SI-SNR loss")
        return values.mean() if self.reduction == "mean" else values
