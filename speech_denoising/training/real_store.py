"""Immutable epoch records for verified VoiceBank train/validation runs."""

import hashlib
import math
from pathlib import Path

from speech_denoising.assets.gtcrn import CHECKPOINT_SHA256
from speech_denoising.utils.utils import REPO_ROOT
from .protocol import fingerprint
from .run_store import read_json, write_json
from .selection import execution_identity, select_checkpoint, validate_identity, validate_report


def run_path(path):
    raw = Path(path).absolute()
    if any(parent.is_symlink() for parent in (raw, *raw.parents)):
        raise ValueError("Run paths must not use symlinks")
    resolved = raw.resolve()
    root = (REPO_ROOT / "results/training_runs").resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Use a separate run directory below results/training_runs/")
    return resolved


def checkpoint_path(run, epoch):
    return run_path(run) / "checkpoints" / f"epoch_{epoch:04d}.pt"


def _record_path(run, kind, epoch):
    return run_path(run) / kind / f"epoch_{epoch:04d}.json"


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_spec(spec):
    if spec.get("schema_version") != 1 or spec.get("scope") != "real_voicebank_training":
        raise ValueError("Expected a real VoiceBank training run specification")
    identity = spec.get("identity")
    validate_identity(identity)
    plan = spec.get("plan")
    if not isinstance(plan, dict) or fingerprint(plan) != spec.get("plan_sha256"):
        raise ValueError("Run plan fingerprint differs")
    if plan.get("model") != "gtcrn" or identity.get("model") != "gtcrn":
        raise ValueError("The first real-training milestone supports GTCRN only")
    train_ids = plan.get("split", {}).get("train_ids")
    validation_ids = plan.get("split", {}).get("validation_ids")
    if any(not isinstance(ids, list) or not ids or ids != sorted(set(ids))
           for ids in (train_ids, validation_ids)):
        raise ValueError("Run specification requires complete ordered train/validation IDs")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Train and validation IDs overlap")
    settings = spec.get("settings", {})
    training = plan.get("protocol", {}).get("training", {})
    expected = {name: training.get(name) for name in (
        "segment_samples", "batch_size", "epochs", "optimizer", "learning_rate",
        "weight_decay", "betas", "gradient_clip_norm", "precision")}
    if any(settings.get(name) != value for name, value in expected.items()):
        raise ValueError("Run settings differ from the bound training plan")
    if settings.get("device") not in ("cpu", "cuda") or type(settings.get("deterministic_algorithms")) is not bool:
        raise ValueError("Run device and deterministic mode must be explicit")
    initialization = spec.get("initialization", {})
    if initialization.get("mode") not in ("random", "pretrained"):
        raise ValueError("Initialization must be random or pretrained")
    digest = initialization.get("checkpoint_sha256")
    if initialization["mode"] == "random":
        if digest is not None or initialization.get("checkpoint_source") is not None:
            raise ValueError("Random initialization cannot name a checkpoint")
    elif digest != CHECKPOINT_SHA256 or initialization.get("checkpoint_source") != "gtcrn_vctk_demand_official":
        raise ValueError("Pretrained initialization must name the pinned GTCRN checkpoint")
    expected_identity = execution_identity(
        plan, identity["seed"], initialization["mode"], digest)
    if identity != expected_identity:
        raise ValueError("Run identity differs from its plan and initialization")
    for name in ("manifest_sha256", "preflight_sha256"):
        value = spec.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Invalid {name}")
    return spec


def _selection(spec, reports):
    best = None
    ids = spec["plan"]["split"]["validation_ids"]
    for report in reports:
        best = select_checkpoint(best, report, spec["identity"], ids, synthetic=False)
    return {
        "schema_version": 1, "scope": "real_voicebank_training",
        "spec_sha256": fingerprint(spec), "validated_epochs": len(reports),
        "reports_sha256": fingerprint(reports), "best": best,
        "checkpoint": f"checkpoints/epoch_{best['epoch']:04d}.pt" if best else None,
    }


def create_run(run, spec):
    run = run_path(run)
    validate_spec(spec)
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "run.json", spec)
    return run


def audit_run(run, spec=None, *, require_selection=False):
    run = run_path(run)
    stored = read_json(run / "run.json")
    validate_spec(stored)
    if spec is not None and stored != spec:
        raise ValueError("Run specification changed")
    spec = stored
    expected_files = {"run.json"}
    records, reports = [], []
    train_count = len(spec["plan"]["split"]["train_ids"])
    batch_size = spec["settings"]["batch_size"]
    steps_per_epoch = (train_count + batch_size - 1) // batch_size
    max_epochs = spec["settings"]["epochs"]
    for epoch in range(1, max_epochs + 1):
        record_path = _record_path(run, "training", epoch)
        if not record_path.exists():
            break
        record = read_json(record_path)
        checkpoint = checkpoint_path(run, epoch)
        if record.get("scope") != "real_voicebank_training" or record.get("epoch") != epoch or (
                record.get("spec_sha256") != fingerprint(spec)):
            raise ValueError("Invalid real-training epoch record")
        if record.get("count") != train_count or record.get("global_step") != epoch * steps_per_epoch:
            raise ValueError("Incomplete real-training epoch")
        loss = record.get("mean_loss")
        if type(loss) not in (int, float) or not math.isfinite(loss):
            raise ValueError("Non-finite real-training loss")
        if not checkpoint.is_file() or record.get("checkpoint_sha256") != _file_hash(checkpoint):
            raise ValueError("Real-training checkpoint is missing or changed")
        records.append(record)
        expected_files.update({str(record_path.relative_to(run)), str(checkpoint.relative_to(run))})
        report_path = _record_path(run, "validation", epoch)
        if report_path.exists():
            if len(reports) != epoch - 1:
                raise ValueError("Validation history has a gap")
            report = read_json(report_path)
            validate_report(report, spec["identity"], spec["plan"]["split"]["validation_ids"])
            if report["epoch"] != epoch or report["checkpoint_sha256"] != record["checkpoint_sha256"]:
                raise ValueError("Validation report does not name this epoch checkpoint")
            reports.append(report)
            expected_files.add(str(report_path.relative_to(run)))
        elif epoch < max_epochs and _record_path(run, "training", epoch + 1).exists():
            raise ValueError("A later epoch exists before independent validation completed")
    selection_path = run / "selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        count = selection.get("validated_epochs")
        if type(count) is not int or not 1 <= count <= len(reports) or selection != _selection(spec, reports[:count]):
            raise ValueError("Selection index differs from validation history")
        expected_files.add("selection.json")
    else:
        selection = None
    for path in run.rglob("*"):
        if path.is_symlink():
            raise ValueError("Run files must not use symlinks")
        if path.is_file() and str(path.relative_to(run)) not in expected_files:
            raise ValueError(f"Unexpected or incomplete run artifact: {path.relative_to(run)}")
    if require_selection and reports and selection != _selection(spec, reports):
        raise ValueError("Validation selection is incomplete; rerun the separate validation command")
    return spec, records, reports


def persist_training(run, spec, epoch, summary, checkpoint_sha256):
    record = {"scope": "real_voicebank_training", "epoch": epoch,
              "spec_sha256": fingerprint(spec), "checkpoint_sha256": checkpoint_sha256, **summary}
    write_json(_record_path(run, "training", epoch), record)
    audit_run(run, spec)
    return record


def persist_validation(run, spec, report):
    _, records, _ = audit_run(run, spec)
    epoch = report["epoch"]
    ids = spec["plan"]["split"]["validation_ids"]
    validate_report(report, spec["identity"], ids)
    if not 1 <= epoch <= len(records) or report["checkpoint_sha256"] != records[epoch - 1]["checkpoint_sha256"]:
        raise ValueError("Validation report is not bound to a recorded training checkpoint")
    path = _record_path(run, "validation", epoch)
    if path.exists():
        if read_json(path) != report:
            raise ValueError("Refusing to replace a different validation report")
    else:
        write_json(path, report)
    _, _, reports = audit_run(run, spec)
    selection = _selection(spec, reports)
    write_json(run_path(run) / "selection.json", selection, replace=True)
    audit_run(run, spec, require_selection=True)
    return selection
