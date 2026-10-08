"""Evaluate the validation-selected GTCRN/LiSenNet checkpoint on the fixed test split."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile

import torch

from scripts.evaluate import implementation_record
from speech_denoising.datasets import VoiceBankDataset
from speech_denoising.metrics.audio_metrics import QUALITY_METRICS, evaluate
from speech_denoising.training.real_store import audit_run, checkpoint_path, read_json, run_path
from speech_denoising.training.real_runner import build_training_model, verify_run_sources
try:
    from speech_denoising.training.state import load_model_state_for_evaluation
except ImportError:
    # Older completed runs predate this helper in state.py. Keep evaluation
    # compatible without changing the source file fingerprint recorded by the run.
    from speech_denoising.training.selection import validate_identity

    def _model_signature(model):
        return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in model.state_dict().items()}

    def load_model_state_for_evaluation(path, model, identity):
        validate_identity(identity)
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if (payload.get("schema_version") != 1 or
                payload.get("kind") != "local_epoch_training_state"):
            raise ValueError("Not a local epoch-boundary training checkpoint")
        if (payload.get("identity") != identity or
                payload.get("model_signature") != _model_signature(model)):
            raise ValueError("Checkpoint experiment or model signature differs")
        model_state = payload.get("model")
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint is missing model state")
        for value in model_state.values():
            if not isinstance(value, torch.Tensor):
                raise ValueError("Checkpoint model state contains a non-tensor")
            if ((value.is_floating_point() or value.is_complex()) and
                    not torch.isfinite(value).all()):
                raise ValueError("Non-finite checkpoint tensor")
        epoch, step = payload.get("completed_epoch"), payload.get("global_step")
        if type(epoch) is not int or epoch < 1 or type(step) is not int or step < 0:
            raise ValueError("Invalid checkpoint progress")
        model.load_state_dict(model_state, strict=True)
        model.zero_grad(set_to_none=True)
        return {"completed_epoch": epoch, "global_step": step,
                "checkpoint_sha256": digest}
from speech_denoising.training.protocol import fingerprint
from speech_denoising.utils.utils import REPO_ROOT


def _sha256_text(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def _evaluation_runtime(device, threads, warmup, repeats):
    import numpy as np
    import soundfile
    result = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "soundfile": soundfile.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "device": device, "dtype": "float32", "threads": threads,
        "interop_threads": torch.get_num_interop_threads(),
        "warmup_per_utterance": warmup, "repeats": repeats,
        "cuda_build": torch.version.cuda,
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        result.update(gpu=properties.name,
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      gpu_total_bytes=properties.total_memory)
    return result


def _output_path(value, run, manifest, checkpoint):
    path = Path(value).absolute()
    root = (REPO_ROOT / "results/trained_reports").resolve()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Evaluation output must not use symlinks")
    resolved = path.resolve()
    run_root = Path(run).resolve()
    if (not resolved.is_relative_to(root) or resolved == root or path.suffix.lower() != ".json" or
            resolved.is_relative_to(run_root) or
            resolved in {Path(manifest).resolve(), Path(checkpoint).resolve(), run_root}):
        raise ValueError("Write a new .json below results/trained_reports/")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    return path


def _publish(path, report):
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".trained-report-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Complete prepared VoiceBank test manifest")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--metrics", nargs="+", choices=QUALITY_METRICS,
                        default=list(QUALITY_METRICS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("Require threads >= 1, warmup >= 0 and repeats >= 1")
    if len(set(args.metrics)) != len(args.metrics):
        parser.error("--metrics must not contain duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    run = run_path(args.run_dir)
    spec, records, reports = audit_run(run, require_selection=True)
    if len(reports) != len(records):
        raise ValueError("Every completed training epoch must have independent validation before test evaluation")
    selection = read_json(run / "selection.json")
    best = selection.get("best")
    if not best or not reports or best["epoch"] > len(records):
        raise ValueError("The run has no complete validation-only checkpoint selection")
    if spec["model"] not in ("gtcrn", "lisennet") or spec["identity"]["profile"] != "native_default":
        raise ValueError("Trained evaluation currently supports native GTCRN/LiSenNet profiles only")
    verify_run_sources(spec)
    checkpoint = checkpoint_path(run, best["epoch"])
    if selection.get("checkpoint") != str(checkpoint.relative_to(run)):
        raise ValueError("Selection checkpoint path differs from the selected epoch")
    output = _output_path(args.output, run, args.manifest, checkpoint)
    dataset = VoiceBankDataset(args.manifest, split="test", verify_files=True, require_complete=True)
    if len(dataset) != 824 or not dataset.protocol_complete:
        raise ValueError("Trained evaluation requires the complete fixed 824-item test manifest")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; there is no CPU fallback")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = build_training_model(spec)
    model.to(device=device, dtype=torch.float32)
    loaded = load_model_state_for_evaluation(checkpoint, model, spec["identity"])
    if (loaded["completed_epoch"] != best["epoch"] or
            loaded["global_step"] != records[best["epoch"] - 1]["global_step"] or
            loaded["checkpoint_sha256"] != best["checkpoint_sha256"]):
        raise ValueError("Loaded checkpoint differs from the validation-selected epoch")
    report = evaluate(model, dataset, metrics=args.metrics, warmup=args.warmup, repeats=args.repeats)
    implementation = implementation_record(spec["model"])
    implementation["source_fingerprints"].update({
        "scripts/evaluate_trained.py": _sha256_text(REPO_ROOT / "scripts/evaluate_trained.py"),
        "speech_denoising/training/real_store.py": _sha256_text(REPO_ROOT / "speech_denoising/training/real_store.py"),
        "speech_denoising/training/state.py": _sha256_text(REPO_ROOT / "speech_denoising/training/state.py"),
        "speech_denoising/training/selection.py": _sha256_text(REPO_ROOT / "speech_denoising/training/selection.py"),
    })
    report.update(
        model=spec["model"], profile=spec["identity"]["profile"],
        comparison_type=("unified_training_draft; not final comparison"
                         if spec["initialization"]["mode"] == "random"
                         else "pretrained_finetuning_integration"),
        checkpoint={"path": str(checkpoint.relative_to(run)),
                    "sha256": loaded["checkpoint_sha256"],
                    "completed_epoch": loaded["completed_epoch"],
                    "global_step": loaded["global_step"]},
        training_run={
            "run_name": run.name, "run_spec_sha256": fingerprint(spec),
            "selection_sha256": fingerprint(selection), "selected_epoch": best["epoch"],
            "validated_epochs": len(reports),
            "completed_epochs": len(records),
            "global_step": records[best["epoch"] - 1]["global_step"],
            "identity": spec["identity"], "initialization": spec["initialization"],
            "manifest_sha256": spec["manifest_sha256"],
            "preflight_sha256": spec["preflight_sha256"],
            "plan_sha256": spec["plan_sha256"], "source_fingerprints": spec["source_fingerprints"],
            "git": spec["git"], "training_runtime": spec["runtime"],
        },
        evaluation_runtime=_evaluation_runtime(args.device, args.threads, args.warmup, args.repeats),
        implementation=implementation,
    )
    _publish(output, report)
    print("Selected epoch:", best["epoch"])
    print("Scope:", report["scope"])
    print("Evaluated:", report["evaluated_count"], "/", report["manifest_count"])
    print("Enhanced metrics:", report["summary"]["enhanced"])
    print("Report:", output)
    if report["failed_utterances"]:
        raise SystemExit("Report contains metric failures; inspect coverage before using scores")


if __name__ == "__main__":
    main()
