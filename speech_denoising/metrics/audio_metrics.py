"""One shared quality/efficiency implementation for every waveform adapter."""

import importlib
import importlib.metadata
import math
import platform
from pathlib import Path
import statistics
import time
import warnings

import numpy as np
import torch

QUALITY_METRICS = ("pesq", "stoi", "si_snr", "si_snr_improvement")
METRIC_NAMES = QUALITY_METRICS + ("rtf", "parameter_count")


def _array(audio):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Metrics require finite nonempty mono [samples] audio")
    return audio


def si_snr(estimate, reference, eps=1e-12):
    estimate, reference = _array(estimate), _array(reference)
    if estimate.shape != reference.shape:
        raise ValueError("SI-SNR requires aligned equal-length audio")
    estimate, reference = estimate - estimate.mean(), reference - reference.mean()
    energy = np.dot(reference, reference)
    if energy <= eps:
        raise ValueError("SI-SNR is undefined for a silent/constant reference")
    target = reference * (np.dot(estimate, reference) / energy)
    residual = estimate - target
    return float(10 * np.log10((np.dot(target, target) + eps) / (np.dot(residual, residual) + eps)))


def check_backends(metrics):
    for name, module in [("pesq", "pesq"), ("stoi", "pystoi")]:
        if name in metrics:
            try:
                importlib.import_module(module)
            except ImportError as exc:
                raise ImportError(f"{name} backend missing. Install {module} or select metrics explicitly.") from exc


def compute_audio_metrics(clean_audio, enhanced_audio, noisy_audio, sample_rate=16000, metrics=QUALITY_METRICS):
    clean, enhanced, noisy = map(_array, (clean_audio, enhanced_audio, noisy_audio))
    if sample_rate != 16000 or clean.shape != enhanced.shape or clean.shape != noisy.shape:
        raise ValueError("Metrics require aligned 16 kHz signals of the same length")
    if any(name not in QUALITY_METRICS for name in metrics):
        raise ValueError("Unknown quality metric")
    check_backends(metrics)
    values, errors = {}, {}
    for name in metrics:
        try:
            if name == "pesq":
                from pesq import pesq
                value = pesq(16000, clean, enhanced, "wb")
            elif name == "stoi":
                from pystoi import stoi
                # A warning can signal inadequate speech frames; never include its fallback score silently.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    value = stoi(clean, enhanced, 16000, extended=False)
            elif name == "si_snr":
                value = si_snr(enhanced, clean)
            else:
                value = si_snr(enhanced, clean) - si_snr(noisy, clean)
            if not math.isfinite(float(value)):
                raise ValueError("Non-finite metric score")
            values[name] = float(value)
        except Exception as exc:
            values[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"
    return {"values": values, "errors": errors}


def processor_name():
    identifier = platform.processor()
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return identifier or "unknown"


def evaluate(model, dataset, *, metrics=QUALITY_METRICS, limit=None, warmup=1, repeats=3):
    """Per-utterance macro quality means and duration-weighted RTF; restore mode.

    Batch size 1. Timing includes the model's waveform frontend/back end,
    excludes decoding, device transfer and quality metrics. Median synchronized
    elapsed time of repeated inference after warmup is used per utterance.
    """
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or any(m not in QUALITY_METRICS for m in metrics):
        raise ValueError("Choose unique quality metrics from QUALITY_METRICS")
    if repeats < 1 or warmup < 0 or limit is not None and limit < 1:
        raise ValueError("Invalid limit/warmup/repeats")
    check_backends(metrics)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    if count < 1:
        raise ValueError("Dataset is empty")
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    training = model.training
    rows, failures = [], []
    total_seconds, total_duration = 0.0, 0.0
    try:
        model.eval()
        with torch.inference_mode():
            for index in range(count):
                sample = dataset[index]  # Data/integrity errors abort rather than silently dropping an utterance.
                if sample["sample_rate"] != getattr(model, "sample_rate", 16000):
                    raise ValueError("Model/dataset sample rate mismatch")
                noisy = sample["noisy_audio"].unsqueeze(0).to(device=device, dtype=dtype)
                for _ in range(warmup):
                    model(noisy)
                timings = []
                for _ in range(repeats):
                    synchronize()
                    start = time.perf_counter()
                    enhanced = model(noisy)
                    synchronize()
                    timings.append(time.perf_counter() - start)
                if enhanced.shape != noisy.shape or not torch.isfinite(enhanced).all():
                    raise ValueError("Model returned misaligned/non-finite output")
                result = compute_audio_metrics(sample["clean_audio"], enhanced[0], sample["noisy_audio"], metrics=metrics)
                baseline = compute_audio_metrics(sample["clean_audio"], sample["noisy_audio"], sample["noisy_audio"], metrics=metrics)
                seconds = statistics.median(timings)
                duration = noisy.shape[-1] / 16000
                total_seconds += seconds
                total_duration += duration
                row = {"utterance_id": sample["utterance_id"], "samples": noisy.shape[-1],
                       "enhanced": result["values"], "noisy": baseline["values"],
                       "errors": {"enhanced": result["errors"], "noisy": baseline["errors"]},
                       "median_inference_seconds": seconds, "rtf": seconds / duration}
                rows.append(row)
                if result["errors"] or baseline["errors"]:
                    failures.append(row["utterance_id"])
                if (index + 1) % 20 == 0 or index + 1 == count:
                    print(f"Evaluated {index + 1}/{count} (manifest: {len(dataset)})", flush=True)
    finally:
        model.train(training)
    summary, coverage = {}, {}
    for role in ["enhanced", "noisy"]:
        summary[role], coverage[role] = {}, {}
        for metric in metrics:
            scores = [r[role][metric] for r in rows if r[role][metric] is not None]
            summary[role][metric] = statistics.mean(scores) if scores else None
            coverage[role][metric] = len(scores)
    full = count == len(dataset) and bool(getattr(dataset, "protocol_complete", False)) and getattr(dataset, "split", None) == "test"
    versions = {}
    for package in ["torch", "numpy", "soundfile", "pyarrow", "pystoi", "pesq"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"scope": "full_fixed_test" if full else "subset_or_fixture",
            "status": "complete_selected_metrics" if not failures else "metric_failures",
            "all_quality_metrics_complete": full and not failures and set(metrics) == set(QUALITY_METRICS),
            "quality_metrics": list(metrics), "not_computed": [m for m in QUALITY_METRICS if m not in metrics],
            "evaluated_count": count, "manifest_count": len(dataset),
            "summary": summary, "metric_coverage": coverage, "failed_utterances": failures,
            "rtf": total_seconds / total_duration, "total_audio_seconds": total_duration,
            "total_median_inference_seconds": total_seconds,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "measurement": {"platform": platform.platform(), "processor": processor_name(), "machine": platform.machine(),
                            "device": str(device), "dtype": str(dtype), "threads": torch.get_num_threads(),
                            "batch_size": 1, "warmup_per_utterance": warmup, "repeats": repeats,
                            "rtf_aggregation": "sum per-utterance median inference seconds / total audio duration",
                            "includes": "adapter STFT, network, iSTFT", "excludes": "I/O, transfer, metrics",
                            "versions": versions},
            "dataset": {k: v for k, v in getattr(dataset, "metadata", {}).items() if k != "utterances"},
            "utterances": rows}
