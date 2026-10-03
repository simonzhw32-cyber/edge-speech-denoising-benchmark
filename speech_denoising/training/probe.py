"""Bounded resource checks, separate from training and checkpoint selection."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from speech_denoising.utils.utils import REPO_ROOT
from .data import load_training_inputs
from .protocol import fingerprint


def source_hashes(names):
    return {name: hashlib.sha256((REPO_ROOT / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in names}


def bind_preflight(manifest_path, preflight_path, model):
    """Reuse a complete audio scan across models only with the same current inputs."""
    report_bytes = Path(preflight_path).read_bytes()
    report = json.loads(report_bytes)
    path, data, manifest, plan = load_training_inputs(manifest_path, model)
    _, old_data, _, old_plan = load_training_inputs(path, report["plan"]["model"])
    expected = {
        "schema_version": 1, "scope": "training_data_preflight", "status": "verified_local_audio",
        "manifest_sha256": hashlib.sha256(data).hexdigest(),
        "rows_sha256": fingerprint(manifest["utterances"]), "source_sha256": fingerprint(plan["source"]),
        "protocol_sha256": plan["protocol_sha256"], "split_sha256": plan["split_sha256"],
        "plan_sha256": fingerprint(old_plan), "plan": old_plan, "audio_files_verified": True,
        "test_audio_read": False, "model_executed": False, "training_performed": False,
        "quality_scores_generated": False, "execution_ready": False,
        "hardware_checked": False, "crop_feasibility_checked": False,
        "checks": {"byte_hashes": True, "mono_16khz": True, "paired_lengths": True,
                   "finite_float32": True, "nonconstant_full_clean_reference": True},
    }
    if old_data != data or any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Preflight does not match the complete current train source and draft split")
    rows = manifest["utterances"]
    lengths = [row["samples"] for row in rows]
    audio = {"verified_pairs": len(rows), "verified_files": 2 * len(rows), "sample_rate": 16000,
             "min_samples": min(lengths), "max_samples": max(lengths),
             "total_samples": sum(lengths), "duration_seconds": sum(lengths) / 16000}
    if report.get("audio") != audio:
        raise ValueError("Incomplete or inconsistent preflight audio coverage")
    names = ["configs/voicebank.json", "configs/training_protocol.yaml",
             old_plan["model_settings"]["config"], "configs/default.yaml",
             "scripts/plan_training.py", "scripts/preflight_training.py",
             "speech_denoising/training/protocol.py", "speech_denoising/training/data.py",
             "speech_denoising/training/preflight.py", "speech_denoising/datasets/voicebank.py",
             "speech_denoising/utils/utils.py"]
    if report.get("source_fingerprints") != source_hashes(names):
        raise ValueError("Preflight source fingerprints differ from this checkout")
    return path, data, manifest, plan, report_bytes


def probe_selection(manifest, plan):
    """One lexical train batch; longest validation item, lexical tie break."""
    training = plan["protocol"]["training"]
    train_ids = plan["split"]["train_ids"][:training["batch_size"]]
    if len(train_ids) != training["batch_size"]:
        raise ValueError("The probe requires a complete configured training batch")
    widths = {row["utterance_id"]: row["samples"] for row in manifest["utterances"]}
    validation_id = min(plan["split"]["validation_ids"], key=lambda key: (-widths[key], key))
    return {"train_ids": train_ids, "validation_id": validation_id,
            "validation_samples": widths[validation_id], "segment_samples": training["segment_samples"],
            "seed": training["seeds"][0], "epoch": 1}


def output_path(output, protected=()):
    path = Path(output).absolute()
    root = (REPO_ROOT / "results/training_probe").resolve()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Probe output must not use symlinks")
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root or path.suffix != ".json":
        raise ValueError("Write a new .json below results/training_probe/")
    if resolved in {Path(item).resolve() for item in protected}:
        raise ValueError("Probe output must differ from every input")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    return path


def publish_report(output, report, protected=()):
    path = output_path(output, protected)
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".probe_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def probe_model(model, train_batch, validation_batch, *, epsilon):
    """Fresh FP32 model only; restore buffers/modes and discard computed gradients.

    CPU snapshots keep preservation overhead out of the GPU memory measurements.
    Timings include transfers and finite checks; these are not benchmark RTF.
    """
    import torch
    from speech_denoising.losses.loss import EnhancementLoss
    from .state import capture_rng, restore_rng
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    tensors = dict(parameters)
    tensors.update({"buffer:" + name: value for name, value in buffers.items()})
    if not parameters or any(value.grad is not None for value in parameters.values()):
        raise ValueError("Use a fresh model with no existing gradients")
    device = next(iter(parameters.values())).device
    if any(value.device != device or (value.is_floating_point() and value.dtype != torch.float32)
           for value in tensors.values()):
        raise ValueError("Probe model tensors must share the FP32 device")
    before = {name: value.detach().cpu().clone() for name, value in tensors.items()}
    modes = [(module, module.training) for module in model.modules()]
    rng = capture_rng()
    criterion = EnhancementLoss(eps=epsilon)
    result = {"status": "passed", "stages": {}, "optimizer_steps": 0}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def restore_tensors():
        with torch.no_grad():
            for name, value in tensors.items():
                value.copy_(before[name])
        model.zero_grad(set_to_none=True)

    try:
        for name, batch, backward in (("train_batch", train_batch, True),
                                      ("validation_utterance", validation_batch, False)):
            restore_tensors()
            model.train(backward)
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            stage = {"status": "running", "input_shape": list(batch["noisy_audio"].shape),
                     "valid_lengths": batch["lengths"].tolist(), "backward": backward}
            if device.type == "cuda":
                stage["baseline_allocated_bytes"] = torch.cuda.memory_allocated(device)
            result["stages"][name] = stage
            noisy = clean = lengths = enhanced = loss = None
            try:
                noisy, clean, lengths = [batch[key].to(device) for key in
                                         ("noisy_audio", "clean_audio", "lengths")]
                with torch.autocast(device_type=device.type, enabled=False), torch.set_grad_enabled(backward):
                    enhanced = model(noisy)
                    if enhanced.shape != noisy.shape or enhanced.dtype != torch.float32:
                        raise ValueError("Model returned a different waveform shape or dtype")
                    if not torch.isfinite(enhanced).all().item():
                        raise ValueError("Non-finite model output")
                    loss = criterion(enhanced, clean, lengths)
                    if backward:
                        loss.backward()
                if backward:
                    gradients = [value.grad for value in parameters.values() if value.grad is not None]
                    if not gradients or not all(torch.isfinite(value).all().item() for value in gradients):
                        raise ValueError("Absent or non-finite gradients")
                    stage["parameters_with_gradients"] = len(gradients)
                    # Do not retain tensor references between the two cases.
                    del gradients
                if any(not torch.equal(value.detach().cpu(), before[key])
                       for key, value in parameters.items()):
                    raise ValueError("Parameters changed despite no optimizer steps")
                stage.update(status="passed", finite_output=True, finite_loss=True)
            except (RuntimeError, ValueError) as error:
                stage.update(status="out_of_memory" if isinstance(error, torch.cuda.OutOfMemoryError)
                             else "failed", error=str(error))
                result["status"] = stage["status"]
            finally:
                synchronize()
                stage["seconds"] = time.perf_counter() - started
                if device.type == "cuda":
                    stage["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    stage["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                del noisy, clean, lengths, enhanced, loss
            if result["status"] != "passed":
                break
    finally:
        restore_tensors()
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
    result["parameters_unchanged"] = all(torch.equal(value.detach().cpu(), before[key])
                                         for key, value in parameters.items())
    result["buffers_restored"] = all(torch.equal(value.detach().cpu(), before["buffer:" + key])
                                    for key, value in buffers.items())
    return result
