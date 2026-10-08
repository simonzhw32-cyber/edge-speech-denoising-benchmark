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


def tiny_spec(initialization="random", model="gtcrn"):
    protocol = load_config("configs/training_protocol.yaml")
    protocol["training"].update(segment_samples=8, batch_size=2, epochs=2)
    settings = model_settings(protocol, model)
    split = {
        "algorithm": protocol["split"]["algorithm"], "seed": protocol["split"]["seed"],
        "parent_rows_sha256": "1" * 64, "train_speakers": ["p226"],
        "validation_speakers": ["p256"],
        "train_ids": ["p226_00001", "p226_00002", "p226_00003"],
        "validation_ids": ["p256_00001", "p256_00002"],
    }
    plan = {
        "schema_version": 1, "scope": "training_metadata_plan", "status": "draft",
        "model": model, "model_settings": settings, "protocol": protocol,
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
        "schema_version": 1, "scope": "real_voicebank_training", "model": model,
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
    validate_spec(tiny_spec(model="lisennet"))
    changed = tiny_spec(model="lisennet")
    changed["model"] = "gtcrn"
    rejected(lambda: validate_spec(changed))
    rejected(lambda: tiny_spec("pretrained", model="lisennet"))
    from speech_denoising.training.real_runner import verify_resume_spec, source_files
    same = deepcopy(spec)
    same["git"] = {"commit": "6" * 40, "dirty": False, "changed_paths": []}
    verify_resume_spec(spec, same)
    for field in ("runtime", "source_fingerprints", "settings", "plan", "identity",
                  "manifest_sha256", "preflight_sha256"):
        changed = deepcopy(same)
        changed[field] = {"changed": True}
        rejected(lambda: verify_resume_spec(spec, changed))
    for model in ("gtcrn", "lisennet"):
        names = source_files(model)
        assert len(names) == len(set(names))
        assert all((REPO_ROOT / name).is_file() for name in names)
    names = source_files("lisennet")
    assert "speech_denoising/models/lisennet/dpr_layer.py" in names
    assert not any("models/gtcrn/" in name for name in names)
    rejected(lambda: source_files("tfgridnet"))
    from speech_denoising.training import real_runner
    from speech_denoising.training.probe import source_hashes
    current = tiny_spec(model="lisennet")
    current["source_fingerprints"] = source_hashes(source_files("lisennet"))
    real_runner.verify_run_sources(current)
    current["source_fingerprints"]["speech_denoising/models/lisennet/dpr_layer.py"] = "0" * 64
    rejected(lambda: real_runner.verify_run_sources(current))
    fixture = tiny_spec(model="lisennet")
    with patch.object(real_runner, "bind_preflight", return_value=(
            Path("fixture.json"), b"manifest", {}, fixture["plan"], b"preflight")) as binding, \
            patch.object(real_runner, "_runtime", return_value={"device": "cpu"}), \
            patch.object(real_runner, "_git_state", return_value=fixture["git"]):
        built = real_runner.build_run_spec("manifest", "preflight", seed=42,
                                          device="cpu", initialization="random", model="lisennet")
        binding.assert_called_once_with("manifest", "preflight", "lisennet")
        validate_spec(built)
        assert built["model"] == built["identity"]["model"] == "lisennet"
        rejected(lambda: real_runner.build_run_spec("manifest", "preflight", seed=42,
            device="cpu", initialization="pretrained", model="lisennet"))
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


def tensor_checks(folder, model_name="gtcrn"):
    import torch
    from torch import nn
    from speech_denoising.datasets.voicebank import VoiceBankDataset
    from speech_denoising.training.data import TrainingSubset
    from speech_denoising.training import protocol as protocol_module
    from speech_denoising.training import real_runner

    spec = tiny_spec(model=model_name)
    if model_name == "lisennet":
        spec["plan"]["protocol"]["training"]["segment_samples"] = 1024
        spec["plan"]["protocol_sha256"] = fingerprint(spec["plan"]["protocol"])
        spec["identity"] = execution_identity(spec["plan"], 42, "random")
        spec["plan_sha256"] = fingerprint(spec["plan"])
        spec["settings"]["segment_samples"] = 1024
    plan = spec["plan"]
    rows = {}
    for index, identifier in enumerate(
            plan["split"]["train_ids"] + plan["split"]["validation_ids"]):
        if model_name == "lisennet":
            generator = torch.Generator().manual_seed(100 + index)
            clean = 0.2 * torch.randn(1200 + index, generator=generator)
        else:
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
        model = real_runner.build_training_model(saved_spec) if model_name == "lisennet" else TinyWaveform()
        if initial is None:
            initial = deepcopy(model.state_dict())
        settings = saved_spec["settings"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]))
        return model, optimizer

    run = folder / ("tensor_" + model_name)
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
            manifest, preflight, run, 1, seed=42, device="cpu", initialization="random", model=model_name)
        assert training["count"] == 3 and training["global_step"] == 2
        rejected(lambda: real_runner.train_real_epoch(
            manifest, preflight, run, 2, seed=42, device="cpu", initialization="random", model=model_name))
        payload = torch.load(checkpoint_path(run, 1), map_location="cpu", weights_only=True)
        assert any(not torch.equal(value, initial[name]) for name, value in payload["model"].items())
        report, selection = real_runner.validate_real_epoch(
            manifest, preflight, run, 1, device="cpu")
        assert report["count"] == 2 and selection["best"] == report
        proposed = deepcopy(spec)
        proposed["git"] = {"commit": "6" * 40, "dirty": False, "changed_paths": []}
        with patch.object(real_runner, "build_run_spec", return_value=proposed):
            second = real_runner.train_real_epoch(
                manifest, preflight, run, 2, seed=42, device="cpu", initialization="random", model=model_name)
        assert second["global_step"] == 4
        real_runner.validate_real_epoch(manifest, preflight, run, 2, device="cpu")
        assert audit_run(run)[0] == spec  # Original provenance is never rewritten.
        rejected(lambda: real_runner.train_real_epoch(
            manifest, preflight, run, 1, seed=42, device="cpu", initialization="random", model=model_name))
    _, records, reports = audit_run(run, spec, require_selection=True)
    assert len(records) == len(reports) == 2
    print(f"PASS: generated-audio {model_name} updates, pending-validation gate and post-commit resume")
    if model_name == "lisennet":
        from scripts import evaluate_trained
        class FixedTestFixture:
            protocol_complete = True
            def __init__(self, *args, **kwargs):
                assert kwargs == {"split": "test", "verify_files": True, "require_complete": True}
            def __len__(self):
                return 824
        output = REPO_ROOT / "results/trained_reports" / (folder.name + "_lisennet.json")
        fixture_report = {"scope": "synthetic_test_routing_fixture", "evaluated_count": 824,
                          "manifest_count": 824, "summary": {"enhanced": {}}, "failed_utterances": []}
        try:
            with patch.object(evaluate_trained, "verify_run_sources"), \
                    patch.object(evaluate_trained, "VoiceBankDataset", FixedTestFixture), \
                    patch.object(evaluate_trained, "evaluate", return_value=fixture_report):
                evaluate_trained.main(["--run-dir", str(run), "--manifest", str(manifest),
                                       "--output", str(output), "--device", "cpu"])
            from speech_denoising.training.run_store import read_json
            result = read_json(output)
            assert result["model"] == "lisennet"
            assert result["checkpoint"]["sha256"] == audit_run(run)[2][result["checkpoint"]["completed_epoch"] - 1]["checkpoint_sha256"]
            assert result["implementation"]["source_fingerprints"].get("speech_denoising/models/lisennet/dpr_layer.py")
            assert result["training_run"]["identity"] == spec["identity"]
        finally:
            output.unlink(missing_ok=True)
        print("PASS: LiSenNet selected weights load and route to the shared test runner (metrics mocked)")


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
            import torch
            torch.set_num_threads(2)
            tensor_checks(folder)
            tensor_checks(folder, "lisennet")
        else:
            print("Metadata only; no framework import, model update or audio access.")
    finally:
        shutil.rmtree(folder)


if __name__ == "__main__":
    main()
