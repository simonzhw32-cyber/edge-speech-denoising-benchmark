"""16 kHz, mono, single-output waveform adapter for original TF-GridNet."""

import torch
from torch.nn import functional as F

from ..base import BaseEnhancementModel
from .network import TFGridNetNetwork


class TFGridNetModel(BaseEnhancementModel):
    """Offline whole-utterance model, initialized with random weights.

    This local enhancement profile is not a published pretrained configuration.
    Bidirectional LSTMs and global attention require future frames. No streaming
    interface or checkpoint loading is provided.
    """

    sample_rate = 16000

    def __init__(self, n_fft=512, stride=256, n_layers=6,
                 lstm_hidden_units=192, attn_n_head=4,
                 attn_approx_qk_dim=512, emb_dim=48, emb_ks=4,
                 emb_hs=1, activation="prelu", eps=1e-5):
        super().__init__()
        integers = dict(n_fft=n_fft, stride=stride, n_layers=n_layers,
                        lstm_hidden_units=lstm_hidden_units, attn_n_head=attn_n_head,
                        attn_approx_qk_dim=attn_approx_qk_dim,
                        emb_dim=emb_dim, emb_ks=emb_ks, emb_hs=emb_hs)
        if any(type(value) is not int or value < 1 for value in integers.values()):
            raise ValueError("All integer configuration values must be positive integers")
        if n_fft % 2 or stride > n_fft // 2:
            raise ValueError("n_fft must be even; stride must be at most n_fft // 2")
        if emb_dim % attn_n_head or emb_hs > emb_ks or emb_ks > n_fft // 2 + 1:
            raise ValueError("Invalid embedding/head dimensions or embedding stride")
        if not isinstance(eps, (int, float)) or not 0 < eps < float("inf"):
            raise ValueError("eps must be finite and positive")
        self.config = dict(n_fft=n_fft, stride=stride, n_layers=n_layers,
                           lstm_hidden_units=lstm_hidden_units,
                           attn_n_head=attn_n_head,
                           attn_approx_qk_dim=attn_approx_qk_dim,
                           emb_dim=emb_dim, emb_ks=emb_ks, emb_hs=emb_hs,
                           activation=activation, eps=eps)
        # Enough samples for centered reflect padding and >= emb_ks frames.
        self.minimum_samples = max(n_fft // 2 + 1, (emb_ks - 1) * stride)
        self.network = TFGridNetNetwork(input_dim=0, n_srcs=1, n_imics=1,
                                        window="hann", use_builtin_complex=True,
                                        **self.config)

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
        original_length = noisy_audio.shape[-1]
        audio = F.pad(noisy_audio, (0, max(0, self.minimum_samples - original_length)))
        lengths = torch.full((audio.shape[0],), audio.shape[-1], dtype=torch.long,
                             device=audio.device)
        enhanced = self.network(audio, lengths)[0][0][..., :original_length]
        # Define exact silent input as silent output, instead of tiny bias noise.
        silent = (noisy_audio == 0).all(dim=-1, keepdim=True)
        return torch.where(silent, torch.zeros_like(enhanced), enhanced)


# Preserve the Phase 1 registry import.
TFGridNet = TFGridNetModel

__all__ = ["TFGridNetModel", "TFGridNet"]
