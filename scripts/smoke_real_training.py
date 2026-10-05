"""Check guarded real-run persistence; optional tensors use only generated audio."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from scripts.plan_training import model_settings
from speech_denoising.training.protocol import fingerprint
from speech_denoising.training.real_store import (
    audit_run, checkpoint_path, create_run, persist_training, persist_validation,
    validate_spec,
)
from speech_denoising.training.selection import execution_identity
from speech_denoising.utils.utils import REPO_ROOT, load_config


def rejected(operation):
    try:
        operation()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return
    raise AssertionError("Invalid real-training input was accepted")


def tiny_spec(initialization="random"):
    protocol = load_config("configs/training_protocol.yaml")
    protocol["training"].update(segment_samples=8, batch_size=2, epochs=2)
    settings = model_settings(protocol, "gtcrn")
    split = {
        "algorithm": protocol["split"]["algorithm"], "seed": protocol["split"]["seed"],
        "parent_rows_sha256": "1" * 64, "train_speakers": ["p226"],
        "validation_speakers": ["p256"],
        "train_ids": ["p226_00001", "p226_00002", "p226_00003"],
        "validation_ids": ["p256_00001", "p256_00002"],
    }
    plan = {
        "schema_version": 1, "scope": "training_metadata_plan", "status": "draft",
        "model": "gtcrn", "model_settings": settings, "protocol": protocol,
        "protocol_sha256": fingerprint(protocol), "source": {"fixture": True},
        "split": split, "split_sha256": fingerprint(split),
        "counts": {"parent_train": 5, "train": 3, "validation": 2},
        "audio_files_verified": False, "training_performed": False,
        "checkpoint_loaded": False, "quality_scores_generated": False,
        "execution_ready": False,
    }
    checkpoint = None
    source = None
    if initialization == "pretrained":
        from speech_denoising.assets.gtcrn import CHECKPOINT_SHA256
        checkpoint = CHECKPOINT_SHA256
        source = "gtcrn_vctk_demand_official"
    identity = execution_identity(plan, 42, initialization, checkpoint)
    training = protocol["training"]
    runtime_settings = {name: training[name] for name in (
        "segment_samples", "batch_size", "epochs", "optimizer", "learning_rate",
        "weight_decay", "betas", "gradient_clip_norm", "precision")}
    runtime_settings.update(device="cpu", deterministic_algorithms=False)
    return {
        "schema_version": 1, "scope": "real_voicebank_training", "model": "gtcrn",
        "purpose": ("unified_training_from_scratch" if initialization == "random"
                    else "pretrained_finetuning_integration"),
        "identity": identity,
        "initialization": {"mode": initialization, "checkpoint_source": source,
                           "checkpoint_sha256": checkpoint},
        "plan": plan, "plan_sha256": fingerprint(plan), "settings": runtime_settings,
        "manifest_sha256": "2" * 64, "preflight_sha256": "3" * 64,
        "source_fingerprints": {"fixture": "4" * 64},
        "git": {"commit": "5" * 40, "dirty": True, "changed_paths": []},
        "runtime": {"device": "cpu"}, "manifest_name": "fixture.json",
        "fixed_test_used": False, "complete_validation_required": True,
    }


def metadata_checks(folder):
    spec = tiny_spec()
    validate_spec(spec)
    validate_spec(tiny_spec("pretrained"))
    changed = deepcopy(spec)
    changed["identity"]["split_sha256"] = "9" * 64
    rejected(lambda: validate_spec(changed))
    changed = tiny_spec("pretrained")
    changed["initialization"]["checkpoint_sha256"] = "9" * 64
    changed["identity"]["initial_checkpoint_sha256"] = "9" * 64
    rejected(lambda: validate_spec(changed))

    run = folder / "metadata"
    create_run(run, spec)
    checkpoint = checkpoint_path(run, 1)
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"metadata-only real-run checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    persist_training(run, spec, 1, {"count": 3, "global_step": 2, "mean_loss": -1.0}, digest)
    report = {
        "schema_version": 1, "loss": "negative_si_snr", "reduction": "utterance_mean",
        "scope": "complete_train_validation", "identity": spec["identity"], "epoch": 1,
        "checkpoint_sha256": digest,
        "validation_ids_sha256": fingerprint(spec["plan"]["split"]["validation_ids"]),
        "count": 2, "mean_loss": -2.0,
    }
    selection = persist_validation(run, spec, report)
    stored, records, reports = audit_run(run, spec, require_selection=True)
    assert stored == spec and len(records) == len(reports) == 1
    assert selection["best"] == report and selection["checkpoint"] == "checkpoints/epoch_0001.pt"
    rejected(lambda: create_run(REPO_ROOT / "results/training_runs", spec))
    rejected(lambda: create_run(REPO_ROOT / "benchmark_reports/real", spec))
    print("PASS: real-run identity, pinned initialization, immutable epoch and selection metadata")


def tensor_checks(folder):
    import torch
    from torch import nn
    from speech_denoising.datasets.voicebank import VoiceBankDataset
    from speech_denoising.training.data import TrainingSubset
    from speech_denoising.training import protocol as protocol_module
    from speech_denoising.training import real_runner

    spec = tiny_spec()
    plan = spec["plan"]
    rows = {}
    for index, identifier in enumerate(
            plan["split"]["train_ids"] + plan["split"]["validation_ids"]):
        clean = torch.linspace(-1.0, 1.0, 12 + index, dtype=torch.float32)
        noisy = clean + 0.05 * torch.sin(torch.arange(clean.numel(), dtype=torch.float32))
        rows[identifier] = {"utterance_id": identifier, "sample_rate": 16000,
                            "noisy_audio": noisy, "clean_audio": clean}

    parent = VoiceBankDataset.__new__(VoiceBankDataset)
    parent.split = "train"
    parent.verify_files = parent.protocol_complete = True
    parent.metadata = {"source": plan["source"]}

    class FixtureSubset(TrainingSubset):
        def __init__(self, manifest_path, supplied_plan, split):
            assert supplied_plan == plan and split in ("train", "validation")
            self.split = split
            self.utterance_ids = tuple(plan["split"][split + "_ids"])
            self.plan_sha256 = fingerprint(plan)
            self._parent = parent

        def __len__(self):
            return len(self.utterance_ids)

        def __getitem__(self, index):
            return rows[self.utterance_ids[index]]

    class TinyWaveform(nn.Module):
        def __init__(self):
            super().__init__()
            self.filter = nn.Conv1d(1, 1, 3, padding=1, bias=False)

        def forward(self, audio):
            return self.filter(audio[:, None])[:, 0]

    initial = None

    def context(saved_spec, **kwargs):
        nonlocal initial
        torch.manual_seed(7)
        model = TinyWaveform()
        if initial is None:
            initial = deepcopy(model.state_dict())
        settings = saved_spec["settings"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]))
        return model, optimizer

    run = folder / "tensor"
    manifest = folder / "manifest.json"
    preflight = folder / "preflight.json"
    manifest.write_text("{}", encoding="utf-8")
    preflight.write_text("{}", encoding="utf-8")
    with patch.object(real_runner, "build_run_spec", return_value=deepcopy(spec)), \
            patch.object(real_runner, "_verify_current_inputs", return_value=(manifest, plan)), \
            patch.object(real_runner, "_context", side_effect=context), \
            patch.object(real_runner, "TrainingSubset", FixtureSubset), \
            patch.object(protocol_module, "build_training_plan", return_value=plan):
        training = real_runner.train_real_epoch(
            manifest, preflight, run, 1, seed=42, device="cpu", initialization="random")
        assert training["count"] == 3 and training["global_step"] == 2
        payload = torch.load(checkpoint_path(run, 1), map_location="cpu", weights_only=True)
        assert any(not torch.equal(value, initial[name]) for name, value in payload["model"].items())
        report, selection = real_runner.validate_real_epoch(
            manifest, preflight, run, 1, device="cpu")
        assert report["count"] == 2 and selection["best"] == report
        rejected(lambda: real_runner.train_real_epoch(
            manifest, preflight, run, 1, seed=42, device="cpu", initialization="random"))
    _, records, reports = audit_run(run, spec, require_selection=True)
    assert len(records) == len(reports) == 1
    print("PASS: generated-audio real path updates parameters, saves state and validates separately")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    root = REPO_ROOT / "results/training_runs"
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="smoke_", dir=root))
    try:
        metadata_checks(folder)
        if not args.metadata_only:
            tensor_checks(folder)
        else:
            print("Metadata only; no framework import, model update or audio access.")
    finally:
        shutil.rmtree(folder)


if __name__ == "__main__":
    main()
