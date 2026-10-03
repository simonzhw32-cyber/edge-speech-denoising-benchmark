"""Owned fixture-run files and validation history; no model execution."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .protocol import fingerprint
from .selection import select_checkpoint, validate_identity, validate_report
from speech_denoising.utils.utils import REPO_ROOT


def run_path(path):
    raw = Path(path).absolute()
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError("Run paths must not use symlinks")
    path = raw.resolve()
    root = (REPO_ROOT / "results/training_fixture").resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Use a separate run directory below results/training_fixture/")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, replace=False):
    """Publish complete bytes; only the owned selection index is replaceable."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refusing a symlink output")
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if replace and path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".run-json-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def checkpoint_path(run, epoch):
    return run / "checkpoints" / f"epoch_{epoch:04d}.pt"


def _record_path(run, kind, epoch):
    return run / kind / f"epoch_{epoch:04d}.json"


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection(spec, reports):
    best = None
    for report in reports:
        best = select_checkpoint(best, report, spec["identity"], spec["validation_ids"], synthetic=True)
    return {"schema_version": 1, "scope": "synthetic_fixture", "spec_sha256": fingerprint(spec),
            "validated_epochs": len(reports), "reports_sha256": fingerprint(reports),
            "best": best, "checkpoint": f"checkpoints/epoch_{best['epoch']:04d}.pt" if best else None}


def audit_run(run, spec, *, require_selection=False):
    """Reject incomplete/corrupt records; preserve them for investigation."""
    run = run_path(run)
    validate_identity(spec["identity"])
    if read_json(run / "run.json") != spec or spec.get("scope") != "synthetic_fixture":
        raise ValueError("Fixture spec or source/config fingerprints changed")
    expected = {"run.json"}
    records, reports = [], []
    for epoch in range(1, spec["max_epochs"] + 1):
        record_file = _record_path(run, "training", epoch)
        if not record_file.exists():
            break
        record = read_json(record_file)
        checkpoint = checkpoint_path(run, epoch)
        if record.get("scope") != "synthetic_fixture" or record.get("epoch") != epoch or (
                record.get("spec_sha256") != fingerprint(spec)):
            raise ValueError("Invalid epoch training record")
        if record.get("count") != len(spec["train_ids"]) or record.get("global_step") != epoch * spec["steps_per_epoch"]:
            raise ValueError("Incomplete fixture epoch or step count")
        loss = record.get("mean_loss")
        if type(loss) not in (int, float) or not math.isfinite(loss):
            raise ValueError("Non-finite training record")
        if not checkpoint.is_file() or record.get("checkpoint_sha256") != _file_hash(checkpoint):
            raise ValueError("Epoch checkpoint is missing or changed")
        records.append(record)
        expected.update({str(record_file.relative_to(run)), str(checkpoint.relative_to(run))})
        report_file = _record_path(run, "validation", epoch)
        if report_file.exists():
            if len(reports) != epoch - 1:
                raise ValueError("Validation history has a gap")
            report = read_json(report_file)
            validate_report(report, spec["identity"], spec["validation_ids"], synthetic=True)
            if report["epoch"] != epoch or report["checkpoint_sha256"] != record["checkpoint_sha256"]:
                raise ValueError("Validation report does not name this epoch checkpoint")
            reports.append(report)
            expected.add(str(report_file.relative_to(run)))
    selection_file = run / "selection.json"
    if selection_file.exists():
        stored = read_json(selection_file)
        count = stored.get("validated_epochs")
        if type(count) is not int or not 1 <= count <= len(reports) or stored != _selection(spec, reports[:count]):
            raise ValueError("Selection index differs from its saved validation history")
        expected.add("selection.json")
    else:
        stored = None
    for path in run.rglob("*"):
        if path.is_symlink():
            raise ValueError("Run files must not use symlinks")
        if path.is_file() and str(path.relative_to(run)) not in expected:
            raise ValueError(f"Unexpected or incomplete run artifact: {path.relative_to(run)}")
    if require_selection and reports and stored != _selection(spec, reports):
        raise ValueError("Validation selection is incomplete; rerun the separate validation command")
    return records, reports


def create_run(run, spec):
    run = run_path(run)
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "run.json", spec)
    return run


def persist_training(run, spec, epoch, summary, checkpoint_sha256):
    record = {"scope": "synthetic_fixture", "epoch": epoch, "spec_sha256": fingerprint(spec),
              "checkpoint_sha256": checkpoint_sha256, **summary}
    write_json(_record_path(run, "training", epoch), record)
    audit_run(run, spec)
    return record


def persist_validation(run, spec, report):
    records, _ = audit_run(run, spec)
    epoch = report["epoch"]
    validate_report(report, spec["identity"], spec["validation_ids"], synthetic=True)
    if not 1 <= epoch <= len(records) or report["checkpoint_sha256"] != records[epoch - 1]["checkpoint_sha256"]:
        raise ValueError("Report checkpoint is not a recorded training epoch")
    path = _record_path(run, "validation", epoch)
    if path.exists():
        if read_json(path) != report:
            raise ValueError("Refusing to replace a different validation report")
    else:
        write_json(path, report)
    _, reports = audit_run(run, spec)
    selection = _selection(spec, reports)
    write_json(run / "selection.json", selection, replace=True)
    audit_run(run, spec, require_selection=True)
    return selection
