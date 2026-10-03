"""Read-only verification of local train audio; no model or optimizer execution."""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

from speech_denoising.utils.utils import REPO_ROOT
from .data import subset_indices, validate_bound_plan
from .protocol import fingerprint


def _audio_path(root, relative):
    path = root / relative
    # Reject symlinks even if their target stays inside the dataset directory.
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p.is_relative_to(root)):
        raise ValueError(f"Audio must not use a symlink: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Audio path escapes the dataset directory: {relative}")
    return resolved


def _decode(data):
    import soundfile as sf
    try:
        return sf.read(io.BytesIO(data), dtype="float32")
    except (RuntimeError, OSError, ValueError) as error:
        raise ValueError(f"Cannot decode audio: {error}") from error


def verify_training_audio(manifest_path, manifest, plan, source, protocol, settings, *, progress=None):
    """Check every pair from one canonical plan; fail rather than skip bad audio."""
    import numpy as np
    validate_bound_plan(manifest, plan, source, protocol, settings)
    for split in ("train", "validation"):
        subset_indices(manifest, plan, split)
    path = Path(manifest_path).resolve(strict=True)
    before = path.read_bytes()
    if json.loads(before) != manifest:
        raise ValueError("Input manifest changed before audio verification")
    root = path.parent
    rows = manifest["utterances"]
    lengths = []
    for index, row in enumerate(rows, 1):
        width = row["samples"]
        if width < 2:
            raise ValueError(f"At least two samples are required: {row['utterance_id']}")
        for role in ("noisy", "clean"):
            audio_path = _audio_path(root, row[role + "_path"])
            data = audio_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != row[role + "_sha256"]:
                raise ValueError(f"Audio hash mismatch: {row['utterance_id']} {role}")
            # Decode the same bytes that were hashed, avoiding a second file read.
            audio, rate = _decode(data)
            if rate != 16000 or audio.ndim != 1 or len(audio) != width:
                raise ValueError(f"Audio format/length mismatch: {row['utterance_id']} {role}")
            if audio.dtype != np.float32 or not np.isfinite(audio).all():
                raise ValueError(f"Expected finite FP32 audio: {row['utterance_id']} {role}")
            if role == "clean":
                with np.errstate(over="ignore", invalid="ignore"):
                    centered = audio - audio.mean(dtype=np.float32)
                    energy = float(np.sum(centered * centered, dtype=np.float32))
                if not math.isfinite(energy) or energy <= protocol["loss"]["epsilon"]:
                    raise ValueError(f"Silent/constant or overflowing clean reference: {row['utterance_id']}")
        lengths.append(width)
        if progress:
            progress(index, len(rows))
    if path.read_bytes() != before:
        raise ValueError("Manifest changed during verification")
    report = {
        "schema_version": 1, "scope": "training_data_preflight", "status": "verified_local_audio",
        "manifest_sha256": hashlib.sha256(before).hexdigest(), "rows_sha256": fingerprint(rows),
        "source_sha256": fingerprint(source), "protocol_sha256": plan["protocol_sha256"],
        "split_sha256": plan["split_sha256"], "plan_sha256": fingerprint(plan), "plan": plan,
        "audio": {"verified_pairs": len(rows), "verified_files": 2 * len(rows),
                  "sample_rate": 16000, "min_samples": min(lengths), "max_samples": max(lengths),
                  "total_samples": sum(lengths), "duration_seconds": sum(lengths) / 16000},
        "checks": {"byte_hashes": True, "mono_16khz": True, "paired_lengths": True,
                   "finite_float32": True, "nonconstant_full_clean_reference": True},
        "audio_files_verified": True, "test_audio_read": False,
        "model_executed": False, "training_performed": False, "quality_scores_generated": False,
        "hardware_checked": False, "crop_feasibility_checked": False, "execution_ready": False,
    }
    return report


def report_path(output, *, protected=()):
    """Only new JSON artifacts below the dedicated generated-output directory."""
    path = Path(output).absolute()
    results = REPO_ROOT / "results" / "training_preflight"
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Report output must not use symlinks")
    resolved = path.resolve()
    if not resolved.is_relative_to(results.resolve()) or resolved == results.resolve() or path.suffix != ".json":
        raise ValueError("Write a new .json below results/training_preflight/")
    if resolved in {Path(p).resolve() for p in protected}:
        raise ValueError("Report must not overwrite an input")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    return path


def write_preflight_report(output, report, *, protected=()):
    """Publish complete bytes without overwriting an existing output."""
    path = report_path(output, protected=protected)
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".preflight_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link publishes complete bytes while refusing an existing target.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path
