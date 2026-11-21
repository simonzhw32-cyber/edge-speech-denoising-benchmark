"""GTCRN paper-recipe training track (ICASSP 2024, arXiv 2401.13260).

The unified benchmark runner (``training/real_runner.py`` + ``training/protocol.py``)
deliberately trains every model with one optimizer and one negative-SI-SNR loss.
This module is the separate, explicit GTCRN paper track described in
``docs/gtcrn_paper_track_plan_2026-11-09.md``:

* sqrt-Hann STFT, ``n_fft=512`` / ``hop_length=256`` at 16 kHz;
* the official hybrid objective
  ``0.01*L_sisnr + 0.7*L_mag + 0.3*(L_real + L_imag)``
  (``scale="official_x100"`` reproduces the released code, which is 100x this);
* Adam at 1e-3 and ``ReduceLROnPlateau(mode="min", factor=0.5, patience=5)``
  stepped once per epoch on the validation loss;
* batch 4 of fixed-length random crops; a pair shorter than the segment is
  wrap-padded, because the official loss has no padding mask and would
  otherwise score silent target frames as signal.

Two values are *not* defined by the paper and are therefore required on the
command line and recorded in ``run.json`` as operator inputs: ``--epochs`` and
``--segment-seconds``.  The paper reports no VCTK epoch count, the official
repository ships no training script, and ``configs/paper_recipes.yaml`` stores
``epochs_vctk: null``.

Writes only under ``results/paper_runs/<name>``; never touches ``training_reports``,
``benchmark_reports`` or the unified ``results/training_runs`` artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from speech_denoising.models.gtcrn.network import GTCRN as GTCRNNetwork
from speech_denoising.training.data import load_training_inputs, subset_indices
from speech_denoising.utils.utils import REPO_ROOT, load_config

SAMPLE_RATE = 16000
BATCH_SIZE = 4
LEARNING_RATE = 1.0e-3
N_FFT = 512
HOP_LENGTH = 256
N_FREQS = N_FFT // 2 + 1
PAPER_SCALE = 1.0
OFFICIAL_SCALE = 100.0
SCOPE = "gtcrn_paper_training"
DEFAULT_CONFIG = "configs/gtcrn_paper.yaml"
RESULTS_ROOT = REPO_ROOT / "results" / "paper_runs"
SOURCE_FILES = (
    "speech_denoising/training/paper_gtcrn.py",
    "speech_denoising/models/gtcrn/model.py",
    "speech_denoising/models/gtcrn/network.py",
    "speech_denoising/training/data.py",
    "speech_denoising/datasets/voicebank.py",
    "speech_denoising/utils/utils.py",
    "scripts/train_gtcrn_paper.py",
    "scripts/smoke_gtcrn_paper.py",
    "configs/gtcrn_paper.yaml",
    "configs/gtcrn.yaml",
)
_VERIFIED_AUDIO: set[Path] = set()


# --------------------------------------------------------------------------- #
# small duplicated helpers (importing paper_lisennet would drag in pesq/joblib)
# --------------------------------------------------------------------------- #
def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    head = REPO_ROOT / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = REPO_ROOT / ".git" / value[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
    return value


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def environment_record(device: torch.device) -> dict:
    record = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
    }
    record["cuda_available"] = device.type == "cuda" and torch.cuda.device_count() > 0
    if record["cuda_available"]:
        record["gpu"] = torch.cuda.get_device_properties(device.index or 0).name
    else:
        record["gpu"] = None
    return record


def capture_rng_state(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.device_count() > 0:
        torch.cuda.set_rng_state_all(state["cuda"])


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class GTCRNPaper(nn.Module):
    """sqrt-Hann STFT frontend + the released GTCRN network, spectrum in and out.

    The submodule is deliberately named ``network`` and the window buffer is
    non-persistent, so ``state_dict()`` keys are identical to
    ``GTCRNModel``'s (``network.*`` only) and checkpoints are interchangeable.
    """

    sample_rate = SAMPLE_RATE
    n_fft = N_FFT
    hop_length = HOP_LENGTH
    n_freqs = N_FREQS

    def __init__(self) -> None:
        super().__init__()
        self.network = GTCRNNetwork()
        self.register_buffer(
            "window", torch.hann_window(self.n_fft).sqrt(), persistent=False
        )

    # -- frontend ---------------------------------------------------------- #
    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` -> ``[B, 257, T, 2]`` with ``T = 1 + L // hop_length``."""
        if audio.ndim != 2:
            raise ValueError("audio must be [batch, samples]")
        if audio.dtype not in (torch.float32, torch.float64):
            raise TypeError("audio must be float32 or float64")
        window = self.window.to(dtype=audio.dtype, device=audio.device)
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.view_as_real(spectrum)

    def istft(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """``[B, 257, T, 2]`` -> ``[B, L]``; ``length=None`` is the official truncation."""
        if spec.ndim != 4 or spec.shape[1] != self.n_freqs or spec.shape[-1] != 2:
            raise ValueError("spec must be [batch, 257, frames, 2]")
        if spec.dtype not in (torch.float32, torch.float64):
            raise TypeError("spec must be float32 or float64")
        window = self.window.to(dtype=spec.dtype, device=spec.device)
        complex_spec = torch.complex(spec[..., 0], spec[..., 1])
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            length=length,
        )

    # -- network ----------------------------------------------------------- #
    def enhance_spec(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.ndim != 4 or spec.shape[1] != self.n_freqs or spec.shape[-1] != 2:
            raise ValueError("spec must be [batch, 257, frames, 2]")
        return self.network(spec)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` -> ``[B, L]`` waveform, parity with ``GTCRNModel.forward``."""
        spectrum = self.stft(audio)
        enhanced = self.enhance_spec(spectrum)
        return self.istft(enhanced, length=audio.shape[-1])


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
class HybridLoss(nn.Module):
    """Exact port of ``Xiaobin-Rong/gtcrn`` ``loss.py`` @ 502ebfab.

    ``L_ri``   = MSE(re/|.|^0.7) + MSE(im/|.|^0.7), each signal normalized by its
                 *own* magnitude (not by the clean magnitude);
    ``L_mag``  = MSE(|.|^0.3, |.|^0.3);
    ``L_sisnr``= mean_b[ -log10( ||y_t||^2 / (||y_hat - y_t||^2 + 1e-8) + 1e-8 ) ],
                 i.e. -(SI-SNR in dB)/10, on ISTFTs without ``length=``.

    ``scale="paper"`` returns ``0.01*sisnr + 0.7*mag + 0.3*ri``;
    ``scale="official_x100"`` returns exactly 100x that, matching the released code.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        beta: float = 0.3,
        mag_power: float = 0.3,
        ri_power: float = 0.7,
        mag_eps: float = 1e-12,
        sisnr_eps: float = 1e-8,
        scale: str = "paper",
    ) -> None:
        super().__init__()
        for name, value in (
            ("alpha", alpha),
            ("beta", beta),
            ("mag_power", mag_power),
            ("ri_power", ri_power),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        for name, value in (("mag_eps", mag_eps), ("sisnr_eps", sisnr_eps)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if scale not in ("paper", "official_x100"):
            raise ValueError('scale must be "paper" or "official_x100"')
        if abs((1.0 - beta) - 0.7) > 1e-9 or abs(beta - 0.3) > 1e-9:
            # Keep the paper weights inspectable, but never silently accept a
            # combination that is not the released 0.3/0.7 split.
            raise ValueError("the released objective requires beta == 0.3")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.mag_power = float(mag_power)
        self.ri_power = float(ri_power)
        self.mag_eps = float(mag_eps)
        self.sisnr_eps = float(sisnr_eps)
        self.scale = scale
        self.register_buffer(
            "window", torch.hann_window(N_FFT).sqrt(), persistent=False
        )

    def _istft(self, spec: torch.Tensor) -> torch.Tensor:
        window = self.window.to(dtype=spec.dtype, device=spec.device)
        complex_spec = torch.complex(spec[..., 0], spec[..., 1])
        return torch.istft(
            complex_spec,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window=window,
            center=True,
        )

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if est.shape != ref.shape:
            raise ValueError("est and ref must share a shape")
        if est.ndim != 4 or est.shape[1] != N_FREQS or est.shape[-1] != 2:
            raise ValueError("est and ref must be [batch, 257, frames, 2]")
        if est.dtype not in (torch.float32, torch.float64) or ref.dtype != est.dtype:
            raise TypeError("est and ref must share float32/float64 dtype")
        if est.shape[0] < 1:
            raise ValueError("est and ref must contain at least one utterance")

        pred_real, pred_imag = est[..., 0], est[..., 1]
        true_real, true_imag = ref[..., 0], ref[..., 1]
        pred_mag = torch.sqrt(pred_real.square() + pred_imag.square() + self.mag_eps)
        true_mag = torch.sqrt(true_real.square() + true_imag.square() + self.mag_eps)

        ri_loss = torch.nn.functional.mse_loss(
            pred_real / pred_mag.pow(self.ri_power),
            true_real / true_mag.pow(self.ri_power),
        ) + torch.nn.functional.mse_loss(
            pred_imag / pred_mag.pow(self.ri_power),
            true_imag / true_mag.pow(self.ri_power),
        )
        mag_loss = torch.nn.functional.mse_loss(
            pred_mag.pow(self.mag_power), true_mag.pow(self.mag_power)
        )

        y_pred = self._istft(est)
        y_true = self._istft(ref)
        projection = (
            torch.sum(y_true * y_pred, dim=-1, keepdim=True)
            * y_true
            / (torch.sum(y_true.square(), dim=-1, keepdim=True) + self.sisnr_eps)
        )
        ratio = torch.norm(projection, dim=-1, keepdim=True).square() / (
            torch.norm(y_pred - projection, dim=-1, keepdim=True).square()
            + self.sisnr_eps
        )
        sisnr = -torch.log10(ratio + self.sisnr_eps).mean()

        paper_total = (
            self.alpha * sisnr + (1.0 - self.beta) * mag_loss + self.beta * ri_loss
        )
        factor = OFFICIAL_SCALE if self.scale == "official_x100" else PAPER_SCALE
        total = paper_total * factor
        if not torch.isfinite(total):
            raise ValueError("non-finite GTCRN hybrid loss")
        parts = {
            "ri": float(ri_loss.detach()),
            "mag": float(mag_loss.detach()),
            "sisnr": float(sisnr.detach()),
            "sisnr_db": float(-10.0 * sisnr.detach()),
            "total": float(total.detach()),
        }
        return total, parts


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _resolve_row(root: Path, row: dict, key: str) -> Path:
    relative = Path(row[key])
    return relative if relative.is_absolute() else (root / relative)


def _read_pair(root: Path, row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_path = _resolve_row(root, row, "noisy_path")
    clean_path = _resolve_row(root, row, "clean_path")
    for path, key in ((noisy_path, "noisy_sha256"), (clean_path, "clean_sha256")):
        expected = row.get(key)
        if expected and path not in _VERIFIED_AUDIO:
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"{key} mismatch for {row['utterance_id']}: {actual}")
            _VERIFIED_AUDIO.add(path)
    noisy, noisy_rate = sf.read(noisy_path, dtype="float32")
    clean, clean_rate = sf.read(clean_path, dtype="float32")
    if noisy_rate != SAMPLE_RATE or clean_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio in {row['utterance_id']}")
    if noisy.ndim != 1 or clean.ndim != 1:
        raise ValueError(f"expected mono audio in {row['utterance_id']}")
    if noisy.shape != clean.shape:
        raise ValueError(f"unaligned pair {row['utterance_id']}")
    return torch.from_numpy(noisy), torch.from_numpy(clean)


def select_rows(manifest_path: str | os.PathLike[str], train_pool: str) -> dict:
    """Return the manifest rows plus the train/validation index views.

    ``paper_full`` trains on all 11,572 rows and monitors the repository's 794
    speaker-disjoint utterances **in-sample**; ``held_out`` trains on 10,778 and
    monitors the same 794 out-of-sample.
    """
    path, raw, manifest, plan = load_training_inputs(manifest_path, "gtcrn")
    rows = manifest["utterances"]
    if train_pool == "paper_full":
        train_indices = tuple(range(len(rows)))
        in_sample = True
    elif train_pool == "held_out":
        train_indices = subset_indices(manifest, plan, "train")
        in_sample = False
    else:
        raise ValueError('train_pool must be "paper_full" or "held_out"')
    validation_indices = subset_indices(manifest, plan, "validation")
    return {
        "path": path,
        "root": Path(path).parent,
        "rows": rows,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows_sha256": manifest.get("rows_sha256"),
        "count": len(rows),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "in_sample_monitor": in_sample,
        "plan": plan,
    }


def _crop_pair(
    noisy: torch.Tensor, clean: torch.Tensor, length: int
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Crop one aligned segment from a noisy/clean pair.

    The same random start is used for both signals.  A pair shorter than the
    segment is wrap-padded rather than zero-padded: zero padding would add
    silent target frames that the official loss has no mask for and would make
    short utterances dominate the objective.
    """
    if noisy.numel() != clean.numel():
        raise ValueError("noisy and clean must have the same length")
    if noisy.numel() == 0:
        raise ValueError("cannot crop an empty pair")
    if noisy.numel() >= length:
        start = int(torch.randint(0, noisy.numel() - length + 1, (1,)).item())
        return noisy[start : start + length], clean[start : start + length], False
    repeats = (length + noisy.numel() - 1) // noisy.numel()
    return noisy.repeat(repeats)[:length], clean.repeat(repeats)[:length], True


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_model(device: torch.device) -> GTCRNPaper:
    return GTCRNPaper().to(device)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(model: nn.Module, settings: dict) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        trainable_parameters(model),
        lr=settings["learning_rate"],
        betas=tuple(settings["betas"]),
        eps=settings["adam_eps"],
        weight_decay=settings["weight_decay"],
    )


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler: dict):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler["mode"],
        factor=scheduler["factor"],
        patience=scheduler["patience"],
        threshold=scheduler["threshold"],
        threshold_mode=scheduler["threshold_mode"],
        cooldown=scheduler["cooldown"],
        min_lr=scheduler["min_lr"],
    )


def training_settings(config: dict, args: argparse.Namespace) -> dict:
    paper = config["paper_track"]
    if args.epochs is None or args.segment_seconds is None:
        raise ValueError("--epochs and --segment-seconds are required (not defined by the paper)")
    if args.device == "cuda" and torch.cuda.device_count() == 0:
        raise RuntimeError("--device cuda requested but no CUDA device is visible")
    if int(args.epochs) < 1 or float(args.segment_seconds) <= 0:
        raise ValueError("--epochs must be >= 1 and --segment-seconds must be > 0")
    if int(paper["sample_rate"]) != SAMPLE_RATE:
        raise ValueError(f"this track implements {SAMPLE_RATE} Hz audio")
    if int(paper["batch_size"]) != BATCH_SIZE:
        raise ValueError(f"the paper states batch size {BATCH_SIZE} for VCTK-DEMAND")
    if str(paper["optimizer"]) != "Adam":
        raise ValueError("the paper states Adam")
    if float(paper["learning_rate"]) != LEARNING_RATE or float(paper["weight_decay"]) != 0.0:
        raise ValueError("the paper states Adam lr=1e-3 and no weight decay")
    scheduler = paper["scheduler"]
    if (str(scheduler["type"]) != "ReduceLROnPlateau"
            or str(scheduler["mode"]) != "min"
            or float(scheduler["factor"]) != 0.5
            or int(scheduler["patience"]) != 5):
        raise ValueError("the paper states ReduceLROnPlateau(min, factor=0.5, patience=5)")
    loss = paper["loss"]
    expected_loss = {
        "alpha": 0.01, "beta": 0.3, "magnitude_power": 0.3,
        "ri_power": 0.7, "mag_eps": 1e-12, "sisnr_eps": 1e-8,
    }
    for key, value in expected_loss.items():
        if float(loss[key]) != value:
            raise ValueError(f"paper loss field {key} must be {value!r}")
    if str(loss["scale"]) not in ("paper", "official_x100"):
        raise ValueError('loss scale must be "paper" or "official_x100"')
    if int(paper["n_fft"]) != N_FFT or int(paper["hop_length"]) != HOP_LENGTH:
        raise ValueError(
            f"this track implements n_fft={N_FFT}, hop_length={HOP_LENGTH}; "
            "the config must match those constants"
        )
    if str(paper.get("window", "")) != "sqrt_hann":
        raise ValueError('this track implements window: "sqrt_hann"')
    validation = paper["validation"]
    if int(validation["batch_size"]) != 1 or str(validation["segment"]) != "full":
        raise ValueError("this track validates one full utterance at a time")
    checkpoint = paper["checkpoint"]
    if not bool(checkpoint["every_epoch"]) or str(checkpoint["selection"]) != "last":
        raise ValueError("this track writes every epoch and selects the last checkpoint")
    results_root = Path(paper["results_root"])
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root
    loss = paper["loss"]
    settings = {
        "sample_rate": int(paper["sample_rate"]),
        "n_fft": int(paper["n_fft"]),
        "hop_length": int(paper["hop_length"]),
        "batch_size": int(paper["batch_size"]),
        "learning_rate": float(paper["learning_rate"]),
        "betas": [float(value) for value in paper["betas"]],
        "adam_eps": float(paper["adam_eps"]),
        "weight_decay": float(paper["weight_decay"]),
        "grad_clip_norm": paper["grad_clip_norm"],
        "epochs": int(args.epochs),
        "segment_seconds": float(args.segment_seconds),
        "segment_samples": int(round(float(args.segment_seconds) * SAMPLE_RATE)),
        "train_pool": paper["train_pool"],
        "seed": int(args.seed if args.seed is not None else paper["seed"]),
        "results_root": str(results_root),
        "validation": {
            "batch_size": int(validation["batch_size"]),
            "segment": str(validation["segment"]),
            "purpose": str(validation["purpose"]),
        },
        "checkpoint": {
            "every_epoch": bool(checkpoint["every_epoch"]),
            "selection": str(checkpoint["selection"]),
        },
        "loss": {
            "alpha": float(loss["alpha"]),
            "beta": float(loss["beta"]),
            "magnitude_power": float(loss["magnitude_power"]),
            "ri_power": float(loss["ri_power"]),
            "mag_eps": float(loss["mag_eps"]),
            "sisnr_eps": float(loss["sisnr_eps"]),
            "scale": str(loss["scale"]),
        },
        "scheduler": {key: paper["scheduler"][key] for key in (
            "mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lr"
        )},
    }
    if settings["segment_samples"] < N_FFT:
        raise ValueError("--segment-seconds must cover at least one STFT window")
    return settings


def settings_fingerprint(settings: dict) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# loop
# --------------------------------------------------------------------------- #
def _batch_loss(model, loss_fn, noisy, clean):
    est = model.enhance_spec(model.stft(noisy))
    return loss_fn(est, model.stft(clean))


def train_epoch(model, loss_fn, optimizer, rows, root, indices, settings, device, log_every):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    order = torch.randperm(len(indices)).tolist()
    batch_size = settings["batch_size"]
    segment = settings["segment_samples"]
    totals, ris, mags, sisnrs, padded = [], [], [], [], 0
    steps = 0
    for position in range(0, len(order), batch_size):
        chunk = [indices[index] for index in order[position : position + batch_size]]
        noisy_batch, clean_batch = [], []
        for row_index in chunk:
            noisy, clean = _read_pair(root, rows[row_index])
            noisy, clean, wrapped = _crop_pair(noisy, clean, segment)
            padded += int(wrapped)
            noisy_batch.append(noisy)
            clean_batch.append(clean)
        noisy_batch = torch.stack(noisy_batch).to(device)
        clean_batch = torch.stack(clean_batch).to(device)

        total, parts = _batch_loss(model, loss_fn, noisy_batch, clean_batch)
        total.backward()
        if settings["grad_clip_norm"] is not None:
            clip_grad_norm_(trainable_parameters(model), float(settings["grad_clip_norm"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        totals.append(parts["total"])
        ris.append(parts["ri"])
        mags.append(parts["mag"])
        sisnrs.append(parts["sisnr_db"])
        steps += 1
        if log_every and steps % log_every == 0:
            print(
                f"    step {steps}: total={parts['total']:.4f} "
                f"ri={parts['ri']:.4f} mag={parts['mag']:.4f} sisnr_db={parts['sisnr_db']:.2f}",
                flush=True,
            )
    if not totals:
        raise ValueError("no training batches were produced")
    return {
        "total": sum(totals) / len(totals),
        "ri": sum(ris) / len(ris),
        "mag": sum(mags) / len(mags),
        "sisnr_db": sum(sisnrs) / len(sisnrs),
        "steps": steps,
        "wrapped_utterances": padded,
    }


@torch.no_grad()
def validate(model, loss_fn, rows, root, indices, settings, device, limit=None):
    model.eval()
    selected = list(indices)[: limit if limit else len(indices)]
    totals, ris, mags, sisnrs = [], [], [], []
    for row_index in selected:
        noisy, clean = _read_pair(root, rows[row_index])
        noisy = noisy.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)
        _, parts = _batch_loss(model, loss_fn, noisy, clean)
        totals.append(parts["total"])
        ris.append(parts["ri"])
        mags.append(parts["mag"])
        sisnrs.append(parts["sisnr_db"])
    if not totals:
        raise ValueError("validation produced no utterances")
    return {
        "total": sum(totals) / len(totals),
        "ri": sum(ris) / len(ris),
        "mag": sum(mags) / len(mags),
        "sisnr_db": sum(sisnrs) / len(sisnrs),
        "count": len(selected),
    }


def checkpoint_payload(model, optimizer, scheduler, epoch, settings, rng_state, history):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "settings": settings,
        "settings_sha256": settings_fingerprint(settings),
        "rng": rng_state,
        "history": history,
    }


def _prepare_run_dir(run_dir: Path, results_root: Path, allow_existing: bool) -> Path:
    resolved = run_dir.resolve()
    root = results_root.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"run directory must live under {root}")
    if resolved.exists():
        if any(resolved.iterdir()) and not allow_existing:
            raise ValueError(f"run directory already exists and is not empty: {resolved}")
    else:
        resolved.mkdir(parents=True)
    (resolved / "checkpoints").mkdir(exist_ok=True)
    return resolved


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_source_checksums(run_dir: Path, allow_existing: bool = False) -> None:
    lines = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"source file is missing: {path}")
        lines.append(f"{sha256_file(path)}  {relative}")
    target = run_dir / "source.sha256"
    if target.exists():
        if not allow_existing:
            raise ValueError(f"source.sha256 already exists in {run_dir}")
        if target.read_text(encoding="utf-8").splitlines() != lines:
            raise ValueError(
                "source files changed since this run directory was created; "
                "refusing to resume with a different training implementation"
            )
        return
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    paper = config["paper_track"]
    settings = training_settings(config, args)
    if args.fixture and args.device != "cpu":
        raise ValueError("--fixture is CPU-only")
    device = torch.device(args.device)
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])

    run_dir = _prepare_run_dir(
        Path(args.run_dir), Path(settings["results_root"]), allow_existing=bool(args.resume)
    )
    command = " ".join(["python", "scripts/train_gtcrn_paper.py", *[str(a) for a in getattr(args, "argv", [])]])

    data = None
    if not args.fixture:
        data = select_rows(args.manifest, settings["train_pool"])
    if args.fixture:
        # Synthetic audio only: no manifest, no data/, no corpus statistics.
        fixture_rows = [{"utterance_id": f"fixture_{index}"} for index in range(8)]
        train_indices = tuple(range(8))
        validation_indices = (0, 1)
        root = None
        rows = fixture_rows
    else:
        rows = data["rows"]
        root = data["root"]
        train_indices = data["train_indices"]
        validation_indices = data["validation_indices"]

    resume_record = None
    if args.resume:
        record_path = run_dir / "run.json"
        if not record_path.is_file():
            raise ValueError("--resume requires an existing run.json in the run directory")
        resume_record = json.loads(record_path.read_text(encoding="utf-8"))
        if data is not None:
            if resume_record.get("dataset", {}).get("sha256") != data["sha256"]:
                raise ValueError("resume manifest differs from the run that wrote this directory")
            if resume_record.get("dataset", {}).get("rows_sha256") != data["rows_sha256"]:
                raise ValueError("resume manifest rows_sha256 differs from run.json")
            if resume_record.get("split_sha256") != data["plan"]["split_sha256"]:
                raise ValueError("resume split differs from run.json; refusing to reuse this checkpoint")
        if resume_record.get("status") == "completed":
            raise ValueError("run.json says this run already completed")

    model = build_model(device)
    loss_fn = HybridLoss(**{
        "alpha": settings["loss"]["alpha"],
        "beta": settings["loss"]["beta"],
        "mag_power": settings["loss"]["magnitude_power"],
        "ri_power": settings["loss"]["ri_power"],
        "mag_eps": settings["loss"]["mag_eps"],
        "sisnr_eps": settings["loss"]["sisnr_eps"],
        "scale": settings["loss"]["scale"],
    }).to(device)
    optimizer = build_optimizer(model, settings)
    scheduler = build_scheduler(optimizer, settings["scheduler"])

    resume_state = None
    start_epoch = 1
    history: list[dict] = []
    if args.resume:
        checkpoint_path = Path(args.resume).resolve(strict=True)
        if not checkpoint_path.is_relative_to(run_dir):
            raise ValueError("--resume checkpoint must live under the run directory")
        # Local, operator-supplied checkpoint: it contains Python/NumPy RNG state.
        resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if resume_state["settings_sha256"] != settings_fingerprint(settings):
            raise ValueError("resume settings differ from the checkpoint")
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        restore_rng_state(resume_state.get("rng"))
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state["epoch"]) + 1
        curve_path = run_dir / "learning_curve.csv"
        if curve_path.is_file():
            with curve_path.open(newline="", encoding="utf-8") as handle:
                curve_rows = list(csv.DictReader(handle))
            if curve_rows and int(curve_rows[-1]["epoch"]) != int(resume_state["epoch"]):
                raise ValueError(
                    "checkpoint epoch and learning_curve.csv disagree; refusing to append"
                )

    run_record = {
        "schema_version": 1,
        "scope": SCOPE,
        "status": "running",
        "recipe": paper["reference"],
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "dataset": (
            {"fixture": True, "count": len(rows)}
            if args.fixture
            else {
                "path": str(data["path"]),
                "sha256": data["sha256"],
                "rows_sha256": data["rows_sha256"],
                "count": data["count"],
            }
        ),
        "split": {
            "train_pool": "fixture" if args.fixture else settings["train_pool"],
            "train_count": len(train_indices),
            "validation_count": len(validation_indices),
            "validation_scored_count": None,
            "in_sample_monitor": False if args.fixture else data["in_sample_monitor"],
            "purpose": "scheduler_monitor_only",
            "checkpoint_selection": "last",
        },
        "split_sha256": None if args.fixture else data["plan"]["split_sha256"],
        "epochs": settings["epochs"],
        "epochs_source": "operator (paper does not define a VCTK epoch budget)",
        "segment_seconds": settings["segment_seconds"],
        "segment_source": "operator (paper does not state a segment length)",
        "batch_size": settings["batch_size"],
        "loss": settings["loss"],
        "loss_scale": settings["loss"]["scale"],
        "optimizer": {"type": "Adam", "learning_rate": settings["learning_rate"],
                      "betas": settings["betas"], "eps": settings["adam_eps"],
                      "weight_decay": settings["weight_decay"],
                      "parameters": "trainable only"},
        "scheduler": {"type": "ReduceLROnPlateau", **settings["scheduler"]},
        "grad_clip_norm": settings["grad_clip_norm"],
        "seed": settings["seed"],
        "git": {"commit": git_commit(), "dirty": git_dirty()},
        "environment": environment_record(device),
        "command": command,
        "started_at": (
            resume_record.get("started_at") if resume_record else _now()
        ),
        "best": None,
        "last": None,
        "history": history,
    }
    _write_json(run_dir / "run.json", run_record)
    _write_source_checksums(run_dir, allow_existing=bool(args.resume))

    curve_path = run_dir / "learning_curve.csv"
    fields = ["epoch", "lr", "train_total", "train_ri", "train_mag", "train_sisnr_db",
              "val_total", "val_ri", "val_mag", "val_sisnr_db", "val_count", "steps",
              "wrapped_utterances", "seconds"]
    write_header = not curve_path.exists()
    best = min((row["val_total"] for row in history), default=None)

    for epoch in range(start_epoch, settings["epochs"] + 1):
        started = time.time()
        lr_before = float(optimizer.param_groups[0]["lr"])
        print(f"epoch {epoch}/{settings['epochs']} lr={lr_before:.3e}", flush=True)
        if args.fixture:
            # Synthetic pairs only: no manifest, no data/, no corpus statistics.
            model.train()
            totals, ris, mags, sisnrs = [], [], [], []
            for step in range(max(1, args.steps)):
                noisy = torch.randn(settings["batch_size"], settings["segment_samples"])
                clean = noisy + 0.1 * torch.randn_like(noisy)
                optimizer.zero_grad(set_to_none=True)
                total, parts = _batch_loss(model, loss_fn, noisy.to(device), clean.to(device))
                total.backward()
                if settings["grad_clip_norm"] is not None:
                    clip_grad_norm_(trainable_parameters(model), float(settings["grad_clip_norm"]))
                optimizer.step()
                if not math.isfinite(parts["total"]):
                    raise ValueError(f"non-finite fixture loss at step {step}")
                totals.append(parts["total"])
                ris.append(parts["ri"])
                mags.append(parts["mag"])
                sisnrs.append(parts["sisnr_db"])
                print(f"  fixture step {step + 1}/{args.steps}: total={parts['total']:.4f}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            mean = lambda values: sum(values) / len(values)
            train_stats = {"total": mean(totals), "ri": mean(ris), "mag": mean(mags),
                           "sisnr_db": mean(sisnrs), "steps": len(totals),
                           "wrapped_utterances": 0}
            val_stats = {"total": mean(totals), "ri": mean(ris), "mag": mean(mags),
                         "sisnr_db": mean(sisnrs), "count": len(validation_indices)}
        else:
            train_stats = train_epoch(model, loss_fn, optimizer, rows, root, train_indices,
                                      settings, device, args.log_every)
            val_stats = validate(model, loss_fn, rows, root, validation_indices, settings,
                                  device, args.validation_limit)

        scheduler.step(val_stats["total"])
        lr_after = float(optimizer.param_groups[0]["lr"])
        if best is None or val_stats["total"] < best:
            best = val_stats["total"]

        row = {
            "epoch": epoch,
            "lr": lr_before,
            "train_total": train_stats["total"], "train_ri": train_stats["ri"],
            "train_mag": train_stats["mag"], "train_sisnr_db": train_stats["sisnr_db"],
            "val_total": val_stats["total"], "val_ri": val_stats["ri"],
            "val_mag": val_stats["mag"], "val_sisnr_db": val_stats["sisnr_db"],
            "val_count": val_stats["count"],
            "steps": train_stats["steps"],
            "wrapped_utterances": train_stats["wrapped_utterances"],
            "seconds": round(time.time() - started, 3),
        }
        history.append(row)
        with curve_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)
        print(
            f"  train total={row['train_total']:.4f} sisnr_db={row['train_sisnr_db']:.2f} | "
            f"val total={row['val_total']:.4f} | lr {lr_before:.3e} -> {lr_after:.3e}",
            flush=True,
        )

        payload = checkpoint_payload(model, optimizer, scheduler, epoch, settings,
                                     capture_rng_state(device), history)
        torch.save(payload, run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")

        run_record.update(last={"epoch": epoch, "val_total": val_stats["total"],
                                "checkpoint": f"checkpoints/epoch_{epoch:04d}.pt"},
                          best={"val_total": best}, history=history)
        run_record["split"]["validation_scored_count"] = val_stats["count"]
        _write_json(run_dir / "run.json", run_record)

    checksums = {
        path.name: sha256_file(path)
        for path in sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    }
    _write_json(run_dir / "checksums.json", checksums)
    run_record.update(status="completed", finished_at=_now(), history=history)
    _write_json(run_dir / "run.json", run_record)
    print(f"done: {run_dir}", flush=True)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTCRN paper-recipe training track (CPU/GPU)")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", default=None,
                        help="complete train manifest; required unless --fixture")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True,
                        help="required: the paper does not define a VCTK epoch budget")
    parser.add_argument("--segment-seconds", type=float, required=True,
                        help="required: the paper does not state a segment length")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fixture", action="store_true", help="synthetic CPU-only 3-step check")
    parser.add_argument("--steps", type=int, default=3, help="fixture steps only")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--validation-limit", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.fixture and not args.manifest:
        parser.error("--manifest is required unless --fixture is given")
    if args.fixture:
        args.epochs = 1
    args.argv = list(sys.argv[1:]) if argv is None else list(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
