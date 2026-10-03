"""Deterministic paired waveform crops and batches with valid sample lengths."""

import hashlib
import json

import torch
from torch.nn import functional as F


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def crop_offset(utterance_id, samples, segment_samples, seed, epoch):
    _integer(samples, "samples", 2)
    _integer(segment_samples, "segment_samples", 2)
    _integer(seed, "seed")
    _integer(epoch, "epoch")
    if not isinstance(utterance_id, str) or not utterance_id:
        raise ValueError("utterance_id must be a nonempty string")
    if samples <= segment_samples:
        return 0
    # An utterance's crop does not depend on worker count, visit order or RNG state.
    key = json.dumps(["paired_crop_v1", seed, epoch, utterance_id, samples, segment_samples],
                     separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    number = int.from_bytes(hashlib.sha256(key).digest(), "big")
    return number % (samples - segment_samples + 1)


def _check_pair(sample):
    if not isinstance(sample, dict):
        raise TypeError("Expected an audio-pair dictionary")
    if not isinstance(sample["utterance_id"], str) or not sample["utterance_id"]:
        raise ValueError("Missing utterance ID")
    if sample["sample_rate"] != 16000:
        raise ValueError("Expected 16 kHz audio")
    noisy, clean = sample["noisy_audio"], sample["clean_audio"]
    if not isinstance(noisy, torch.Tensor) or not isinstance(clean, torch.Tensor):
        raise TypeError("Audio must be tensors")
    if noisy.ndim != 1 or noisy.shape != clean.shape or noisy.numel() < 2:
        raise ValueError("Expected aligned mono audio with at least two samples")
    if noisy.dtype not in (torch.float32, torch.float64):
        raise TypeError("Use float32 or float64 audio")
    if noisy.dtype != clean.dtype or noisy.device != clean.device:
        raise ValueError("Paired audio must share dtype and device")
    if not torch.isfinite(noisy).all() or not torch.isfinite(clean).all():
        raise ValueError("Audio contains NaN or infinity")
    return noisy, clean


def paired_crop(sample, *, segment_samples, seed, epoch):
    """Crop both signals at one offset; right-pad short signals without resampling."""
    noisy, clean = _check_pair(sample)
    if "valid_samples" in sample or "crop_start" in sample:
        raise ValueError("paired_crop expects an uncropped audio pair")
    start = crop_offset(sample["utterance_id"], noisy.numel(), segment_samples, seed, epoch)
    length = min(noisy.numel(), segment_samples)
    return {
        "utterance_id": sample["utterance_id"], "sample_rate": 16000,
        "noisy_audio": F.pad(noisy[start:start + length], (0, segment_samples - length)).clone(),
        "clean_audio": F.pad(clean[start:start + length], (0, segment_samples - length)).clone(),
        "valid_samples": length, "crop_start": start,
    }


def collate_audio_pairs(samples):
    """Collate raw pairs or paired crops; preserve fixed crop width and valid lengths."""
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("Provide a nonempty list of audio pairs")
    pairs = [_check_pair(sample) for sample in samples]
    dtype, device = pairs[0][0].dtype, pairs[0][0].device
    if any(noisy.dtype != dtype or noisy.device != device for noisy, _ in pairs):
        raise ValueError("Every batch item must share dtype and device")
    width = max(noisy.numel() for noisy, _ in pairs)
    lengths, starts, noisy_batch, clean_batch = [], [], [], []
    for sample, (noisy, clean) in zip(samples, pairs):
        length = sample.get("valid_samples", noisy.numel())
        start = sample.get("crop_start", 0)
        _integer(length, "valid_samples", 2)
        _integer(start, "crop_start")
        if length > noisy.numel():
            raise ValueError("Valid length exceeds the audio width")
        lengths.append(length)
        starts.append(start)
        noisy_batch.append(F.pad(noisy[:length], (0, width - length)))
        clean_batch.append(F.pad(clean[:length], (0, width - length)))
    return {
        "utterance_ids": [sample["utterance_id"] for sample in samples],
        "sample_rate": 16000, "crop_starts": starts,
        "noisy_audio": torch.stack(noisy_batch), "clean_audio": torch.stack(clean_batch),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
    }
