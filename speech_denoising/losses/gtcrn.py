"""GTCRN hybrid loss (ICASSP 2024, Eq. 1).

The official implementation uses ``30*(real + imag) + 70*magnitude + sisnr``
on STFT coefficients. The paper writes the same objective with normalized
weights ``0.01*sisnr + 0.7*mag + 0.3*(real + imag)``; the factor of 100 does
not change the Adam trajectory except through gradient clipping, so the
official scaling is kept here for reproducibility.
"""

import math

import torch
from torch import nn


class GTCRNHybridLoss(nn.Module):
    """Sqrt-Hann STFT hybrid loss with compressed RI and magnitude terms."""

    def __init__(
        self,
        n_fft=512,
        hop_length=256,
        win_length=512,
        eps=1e-12,
        real_imag_weight=30.0,
        magnitude_weight=70.0,
        sisnr_weight=1.0,
    ):
        super().__init__()
        for name, value in (
            ("n_fft", n_fft),
            ("hop_length", hop_length),
            ("win_length", win_length),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hop_length > n_fft:
            raise ValueError("hop_length must not exceed n_fft")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft")
        for name, value in (
            ("eps", eps),
            ("real_imag_weight", real_imag_weight),
            ("magnitude_weight", magnitude_weight),
            ("sisnr_weight", sisnr_weight),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = float(eps)
        self.real_imag_weight = float(real_imag_weight)
        self.magnitude_weight = float(magnitude_weight)
        self.sisnr_weight = float(sisnr_weight)

    def _stft(self, audio):
        window = torch.hann_window(
            self.win_length, dtype=audio.dtype, device=audio.device
        ).pow(0.5)
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )

    def _istft(self, spectrum, dtype):
        window = torch.hann_window(
            self.win_length, dtype=dtype, device=spectrum.device
        ).pow(0.5)
        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
        )

    def forward(self, enhanced_audio, clean_audio, lengths=None):
        if not isinstance(enhanced_audio, torch.Tensor) or not isinstance(clean_audio, torch.Tensor):
            raise TypeError("Audio inputs must be tensors")
        if enhanced_audio.ndim != 2 or enhanced_audio.shape != clean_audio.shape:
            raise ValueError("Expected aligned [batch, samples] audio")
        if enhanced_audio.dtype not in (torch.float32, torch.float64):
            raise TypeError("Use float32 or float64 audio")
        if clean_audio.dtype != enhanced_audio.dtype or clean_audio.device != enhanced_audio.device:
            raise ValueError("Audio inputs must share dtype and device")
        if not torch.isfinite(enhanced_audio).all() or not torch.isfinite(clean_audio).all():
            raise ValueError("Audio contains NaN or infinity")
        batch, samples = enhanced_audio.shape
        if batch < 1 or samples < self.win_length:
            raise ValueError("Audio must be at least one window long")
        if lengths is not None:
            if not isinstance(lengths, torch.Tensor) or lengths.shape != (batch,):
                raise ValueError("lengths must be an integer [batch] tensor")
            if lengths.device != enhanced_audio.device or lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("lengths must be integer and on the audio device")
            if ((lengths < self.win_length) | (lengths > samples)).any():
                raise ValueError("Each valid length must cover a full window")

        pred_stft = self._stft(enhanced_audio)
        true_stft = self._stft(clean_audio)

        pred_real, pred_imag = pred_stft.real, pred_stft.imag
        true_real, true_imag = true_stft.real, true_stft.imag
        pred_mag = torch.sqrt(pred_real.square() + pred_imag.square() + self.eps)
        true_mag = torch.sqrt(true_real.square() + true_imag.square() + self.eps)

        pred_real_c = pred_real / pred_mag.pow(0.7)
        pred_imag_c = pred_imag / pred_mag.pow(0.7)
        true_real_c = true_real / true_mag.pow(0.7)
        true_imag_c = true_imag / true_mag.pow(0.7)

        real_loss = nn.functional.mse_loss(pred_real_c, true_real_c)
        imag_loss = nn.functional.mse_loss(pred_imag_c, true_imag_c)
        mag_loss = nn.functional.mse_loss(pred_mag.pow(0.3), true_mag.pow(0.3))

        y_pred = self._istft(pred_stft, enhanced_audio.dtype)
        y_true = self._istft(true_stft, enhanced_audio.dtype)
        if lengths is not None:
            limit = int(lengths.min())
            y_pred = y_pred[..., :limit]
            y_true = y_true[..., :limit]

        projection = (
            torch.sum(y_true * y_pred, dim=-1, keepdim=True)
            * y_true
            / (torch.sum(y_true.square(), dim=-1, keepdim=True) + 1e-8)
        )
        sisnr = -torch.log10(
            torch.norm(projection, dim=-1, keepdim=True).square()
            / (torch.norm(y_pred - projection, dim=-1, keepdim=True).square() + 1e-8)
            + 1e-8
        ).mean()

        total = (
            self.real_imag_weight * (real_loss + imag_loss)
            + self.magnitude_weight * mag_loss
            + self.sisnr_weight * sisnr
        )
        if not torch.isfinite(total):
            raise ValueError("Non-finite GTCRN hybrid loss")
        return total
