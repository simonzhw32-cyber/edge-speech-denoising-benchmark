"""Exercise entry-point persistence; synthetic GTCRN updates, never VoiceBank training."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

from speech_denoising.training.fixture import fixture_spec
from speech_denoising.training.protocol import fingerprint
from speech_denoising.training import run_store
from speech_denoising.utils.utils import REPO_ROOT


def rejected(operation):
    try:
        operation()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return
    raise AssertionError("Invalid run input was accepted")


def command(*args, success=True):
    result = subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    if success and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not success and not result.returncode:
        raise AssertionError("CLI accepted an invalid operation: " + repr(args))
    return result


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def metadata_checks(folder):
    # Snapshot lookups use POSIX keys even when Path uses Windows separators.
    class WindowsFile(PureWindowsPath):
        def is_file(self):
            return True

        def read_bytes(self):
            return b"path fixture"

    class WindowsDirectory(PureWindowsPath):
        def rglob(self, pattern):
            return [WindowsFile(self / "validation/epoch_0001.json"),
                    WindowsFile(self / "checkpoints/epoch_0001.pt")]

    assert snapshot(WindowsDirectory("C:/fixture/run")) == {
        "validation/epoch_0001.json": b"path fixture", "checkpoints/epoch_0001.pt": b"path fixture"}
    spec = fixture_spec()
    assert spec["scope"] == "synthetic_fixture" and not spec["execution_ready_for_real_data"]
    assert spec["max_epochs"] == 2 and len(spec["train_ids"]) == 4
    rejected(lambda: fixture_spec(99))
    rejected(lambda: run_store.run_path(REPO_ROOT / "benchmark_reports"))
    rejected(lambda: run_store.run_path(REPO_ROOT / "results/training_fixture"))
    run = folder / "metadata"
    run_store.create_run(run, spec)
    rejected(lambda: run_store.create_run(run, spec))
    bytes_before = (run / "run.json").read_bytes()
    rejected(lambda: run_store.write_json(run / "run.json", {}))
    assert (run / "run.json").read_bytes() == bytes_before
    checkpoint = run_store.checkpoint_path(run, 1)
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"metadata-only checkpoint fixture; never deserialized")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rejected(lambda: run_store.audit_run(run, spec))  # Unpublished checkpoint is an orphan.
    run_store.persist_training(run, spec, 1, {"count": 4, "global_step": 2, "mean_loss": -1.}, digest)
    report = {"schema_version": 1, "loss": "negative_si_snr", "reduction": "utterance_mean",
              "scope": "synthetic_fixture", "identity": spec["identity"], "epoch": 1,
              "checkpoint_sha256": digest, "validation_ids_sha256": fingerprint(spec["validation_ids"]),
              "count": 2, "mean_loss": -2.}
    original_writer = run_store.write_json

    def fail_selection(path, value, **kwargs):
        if Path(path).name == "selection.json":
            raise OSError("Injected selection publication failure")
        return original_writer(path, value, **kwargs)

    with patch.object(run_store, "write_json", fail_selection):
        rejected(lambda: run_store.persist_validation(run, spec, report))
    assert run_store.read_json(run / "validation/epoch_0001.json") == report
    rejected(lambda: run_store.audit_run(run, spec, require_selection=True))
    best = run_store.persist_validation(run, spec, report)
    before = snapshot(run)
    assert run_store.persist_validation(run, spec, report) == best
    assert snapshot(run) == before
    rejected(lambda: run_store.persist_validation(run, spec, dict(report, mean_loss=-3.)))
    rejected(lambda: run_store.persist_validation(run, spec, dict(report, scope="full_fixed_test")))
    second_checkpoint = run_store.checkpoint_path(run, 2)
    second_checkpoint.write_bytes(b"second metadata-only checkpoint fixture")
    second_digest = hashlib.sha256(second_checkpoint.read_bytes()).hexdigest()
    run_store.persist_training(run, spec, 2, {"count": 4, "global_step": 4, "mean_loss": -1.5}, second_digest)
    second_report = dict(report, epoch=2, checkpoint_sha256=second_digest)
    with patch.object(run_store, "write_json", fail_selection):
        rejected(lambda: run_store.persist_validation(run, spec, second_report))
    rejected(lambda: run_store.audit_run(run, spec, require_selection=True))
    final_selection = run_store.persist_validation(run, spec, second_report)
    assert final_selection["validated_epochs"] == 2 and final_selection["best"]["epoch"] == 1
    before = snapshot(run)
    first_report_path = run / "validation/epoch_0001.json"
    first_report_path.unlink()
    rejected(lambda: run_store.audit_run(run, spec))
    first_report_path.write_bytes(before["validation/epoch_0001.json"])
    unexpected = run / "unexpected.txt"
    unexpected.write_text("fixture", encoding="utf-8")
    rejected(lambda: run_store.audit_run(run, spec))
    unexpected.unlink()
    assert snapshot(run) == before
    checkpoint.write_bytes(b"changed fixture")
    rejected(lambda: run_store.audit_run(run, spec))
    checkpoint.write_bytes(before["checkpoints/epoch_0001.pt"])
    changed = deepcopy(spec)
    changed["settings"]["segment_samples"] = 2048
    rejected(lambda: run_store.audit_run(run, changed))
    assert snapshot(run) == before
    for model in ("gtcrn", "lisennet", "tfgridnet"):
        description = json.loads(command("scripts.train", "--describe", "--model", model).stdout)
        assert description["model"] == model and not description["execution_ready"]
    command("scripts.train", success=False)
    command("scripts.validate", "--run-dir", str(run), "--epoch", "1", success=False)
    print("PASS: fixture scope, output ownership, report replay, hash checks and selection recovery")
    print("PASS: metadata-only train descriptions and explicit execution-scope guards")


def tensor_checks(folder):
    import torch
    from speech_denoising.training.fixture import fixture_audio, fixture_context
    from speech_denoising.training.runner import train_one_epoch
    from speech_denoising.training.state import capture_rng, restore_rng

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor) and torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right

    outer_rng = capture_rng()
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        spec = fixture_spec()
        model, optimizer = fixture_context(spec)
        original = deepcopy(model.state_dict())
        dataset = fixture_audio(spec, "train")
        first = train_one_epoch(model, optimizer, dataset, spec["train_ids"], seed=42,
                                epoch=1, settings=spec["settings"])
        second = train_one_epoch(model, optimizer, dataset, spec["train_ids"], seed=42,
                                 epoch=2, settings=spec["settings"], global_step=first["global_step"])
        assert first["global_step"] == 2 and second["global_step"] == 4
        trainable = dict(model.named_parameters())
        assert any(not torch.equal(value, original[name]) for name, value in trainable.items() if value.requires_grad)
        for name, value in trainable.items():
            if not value.requires_grad:
                equal(value, original[name])
        expected = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
                    "rng": capture_rng(), "module_modes": {name: m.training for name, m in model.named_modules()}}
        run = folder / "cli"
        arguments = ("--fixture", "--run-dir", str(run))
        command("scripts.train", *arguments, "--epoch", "1")
        files = snapshot(run)
        assert not (run / "validation").exists() and not (run / "selection.json").exists()
        command("scripts.train", *arguments, "--epoch", "2", success=False)
        command("scripts.train", *arguments, "--epoch", "1", success=False)
        command("scripts.train", *arguments, "--epoch", "3", success=False)
        assert snapshot(run) == files
        command("scripts.validate", *arguments, "--epoch", "1")
        command("scripts.train", *arguments, "--epoch", "2")
        command("scripts.validate", *arguments, "--epoch", "2")
        records, reports = run_store.audit_run(run, spec, require_selection=True)
        assert len(records) == len(reports) == 2
        assert records[0]["mean_loss"] == first["mean_loss"] and records[1]["mean_loss"] == second["mean_loss"]
        payload = torch.load(run_store.checkpoint_path(run, 2), map_location="cpu", weights_only=True)
        for key in expected:
            equal(payload[key], expected[key])
        assert payload["global_step"] == 4 and payload["completed_epoch"] == 2
        selection = run_store.read_json(run / "selection.json")
        best = min(reports, key=lambda report: (report["mean_loss"], report["epoch"]))
        assert selection["best"] == best
        assert hashlib.sha256((run / selection["checkpoint"]).read_bytes()).hexdigest() == best["checkpoint_sha256"]
        files = snapshot(run)
        command("scripts.validate", *arguments, "--epoch", "2")
        command("scripts.train", *arguments, "--epoch", "2", success=False)
        command("scripts.train", *arguments, "--epoch", "2", "--seed", "43", success=False)
        assert snapshot(run) == files
        from torch import nn
        from speech_denoising.losses.loss import EnhancementLoss
        from speech_denoising.training import runner

        class TinyWaveform(nn.Module):
            def __init__(self):
                super().__init__()
                self.filter = nn.Conv1d(1, 1, 3, padding=1, bias=False)
                self.batch_sizes = []

            def forward(self, audio):
                self.batch_sizes.append(audio.shape[0])
                return self.filter(audio[:, None])[:, 0]

        observed = []

        class RecordedLoss(EnhancementLoss):
            def forward(self, *args, **kwargs):
                values = super().forward(*args, **kwargs)
                observed.extend(values.detach().cpu().tolist())
                return values

        tiny = TinyWaveform()
        zero_lr_optimizer = torch.optim.AdamW(tiny.parameters(), lr=0.)
        with patch.object(runner, "EnhancementLoss", RecordedLoss):
            partial = train_one_epoch(tiny, zero_lr_optimizer, dataset[:3], spec["train_ids"][:3],
                                      seed=42, epoch=1, settings=spec["settings"])
        assert tiny.batch_sizes == [2, 1] and partial["count"] == 3 and partial["global_step"] == 2
        assert abs(partial["mean_loss"] - sum(observed) / 3) < 1e-6
        print("PASS: retained partial batch and utterance-weighted training loss")
        print("PASS: separate CLI validation, pending-validation gate and complete report/selection history")
        print("PASS: resumed GTCRN synthetic updates match uninterrupted parameters, moments, RNG and progress")
    finally:
        restore_rng(outer_rng)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="No PyTorch or model execution")
    args = parser.parse_args()
    root = REPO_ROOT / "results/training_fixture"
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="smoke_", dir=root))
    try:
        metadata_checks(folder)
        if args.metadata_only:
            print("Metadata checks only; optimizer and model checks were not run.")
        else:
            tensor_checks(folder)
            print("Synthetic random-GTCRN updates only; no assets, VoiceBank training or quality benchmark.")
    finally:
        shutil.rmtree(folder)


if __name__ == "__main__":
    main()
