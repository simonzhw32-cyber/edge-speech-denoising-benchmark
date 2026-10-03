"""Probe guards and toy backward passes; no assets or speech-model updates."""

import argparse
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from scripts.smoke_training_plan import synthetic_manifest
from speech_denoising.training.data import load_training_inputs
from speech_denoising.training.probe import (bind_preflight, probe_selection, publish_report,
                                            source_hashes)
from speech_denoising.training.protocol import fingerprint
from speech_denoising.utils.utils import REPO_ROOT


def reject(operation):
    try:
        operation()
    except (OSError, ValueError, KeyError, TypeError):
        return
    raise AssertionError("Invalid probe input was accepted")


def metadata_checks(folder):
    source = json.loads((REPO_ROOT / "configs/voicebank.json").read_text())
    manifest = synthetic_manifest(source)
    parent = folder / "manifest.json"
    parent.write_text(json.dumps(manifest), encoding="utf-8")
    _, data, _, plan = load_training_inputs(parent, "gtcrn")
    lengths = [row["samples"] for row in manifest["utterances"]]
    names = ["configs/voicebank.json", "configs/training_protocol.yaml", "configs/gtcrn.yaml",
             "configs/default.yaml", "scripts/plan_training.py", "scripts/preflight_training.py",
             "speech_denoising/training/protocol.py", "speech_denoising/training/data.py",
             "speech_denoising/training/preflight.py", "speech_denoising/datasets/voicebank.py",
             "speech_denoising/utils/utils.py"]
    report = {
        "schema_version": 1, "scope": "training_data_preflight", "status": "verified_local_audio",
        "manifest_sha256": hashlib.sha256(data).hexdigest(), "rows_sha256": manifest["rows_sha256"],
        "source_sha256": fingerprint(source), "protocol_sha256": plan["protocol_sha256"],
        "split_sha256": plan["split_sha256"], "plan_sha256": fingerprint(plan), "plan": plan,
        "audio_files_verified": True, "model_executed": False, "training_performed": False,
        "quality_scores_generated": False, "test_audio_read": False, "hardware_checked": False,
        "crop_feasibility_checked": False, "execution_ready": False,
        "checks": {"byte_hashes": True, "mono_16khz": True, "paired_lengths": True,
                   "finite_float32": True, "nonconstant_full_clean_reference": True},
        "audio": {"verified_pairs": len(lengths), "verified_files": 2 * len(lengths),
                  "sample_rate": 16000, "min_samples": min(lengths), "max_samples": max(lengths),
                  "total_samples": sum(lengths), "duration_seconds": sum(lengths) / 16000},
        "source_fingerprints": source_hashes(names),
    }
    preflight = folder / "preflight.json"
    def write(value):
        preflight.write_text(json.dumps(value), encoding="utf-8")
    write(report)
    selections = []
    for model in ("gtcrn", "lisennet", "tfgridnet"):
        _, _, _, bound, _ = bind_preflight(parent, preflight, model)
        selected = probe_selection(manifest, bound)
        assert selected["train_ids"] == bound["split"]["train_ids"][:4]
        validation_ids = set(bound["split"]["validation_ids"])
        validation_rows = [row for row in manifest["utterances"]
                           if row["utterance_id"] in validation_ids]
        expected = min(validation_rows, key=lambda row: (-row["samples"], row["utterance_id"]))
        assert selected["validation_id"] == expected["utterance_id"]
        assert selected["segment_samples"] == 64000 and selected["seed"] == 42
        assert selected["epoch"] == 1
        selections.append(selected)
        if model == "tfgridnet":
            assert bound["model_settings"]["constructor_kwargs"]["n_layers"] == 6
    assert selections[0] == selections[1] == selections[2]
    for change in (
        lambda value: value.update(scope="full_fixed_test"),
        lambda value: value.update(audio_files_verified=False),
        lambda value: value.update(manifest_sha256="0" * 64),
        lambda value: value["audio"].update(verified_pairs=1),
        lambda value: value["checks"].update(byte_hashes=False),
        lambda value: value["source_fingerprints"].update({names[0]: "0" * 64}),
        lambda value: value["plan"]["split"]["train_ids"].pop(),
        lambda value: value.update(execution_ready=True),
    ):
        bad = deepcopy(report)
        change(bad)
        write(bad)
        reject(lambda: bind_preflight(parent, preflight, "gtcrn"))
    write(report)
    parent.write_bytes(data + b"\n")
    reject(lambda: bind_preflight(parent, preflight, "gtcrn"))
    parent.write_bytes(data)
    root = REPO_ROOT / "results/training_probe"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke_", dir=root) as temporary:
        output = Path(temporary) / "report.json"
        publish_report(output, {"scope": "synthetic_probe_output"})
        original = output.read_bytes()
        reject(lambda: publish_report(output, {}))
        reject(lambda: publish_report(folder / "outside.json", {}))
        reject(lambda: publish_report(REPO_ROOT / "benchmark_reports/new.json", {}))
        protected = Path(temporary) / "input.json"
        reject(lambda: publish_report(protected, {}, (protected,)))
        reject(lambda: publish_report(Path(temporary) / "nan.json", {"x": float("nan")}))
        assert not (Path(temporary) / "nan.json").exists()
        command = [sys.executable, "-m", "scripts.probe_training", "--model", "gtcrn",
                   "--manifest", str(parent), "--preflight", str(preflight), "--device", "cuda",
                   "--output", str(output)]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True)
        assert result.returncode != 0 and b"overwrite" in result.stderr
        assert output.read_bytes() == original
    assert "torch" not in sys.modules and "soundfile" not in sys.modules
    return parent, preflight


def tensor_checks(parent, preflight):
    import torch
    from speech_denoising.training.probe import probe_model
    from speech_denoising.training.state import capture_rng
    class Toy(torch.nn.Module):
        def __init__(self, failure=None):
            super().__init__()
            self.conv = torch.nn.Conv1d(1, 1, 3, padding=1)
            self.bn = torch.nn.BatchNorm1d(1)
            self.failure = failure
        def forward(self, audio):
            result = self.bn(self.conv(audio[:, None]))[:, 0]
            if self.failure == "oom":
                raise torch.cuda.OutOfMemoryError("synthetic OOM; no GPU allocation")
            if self.failure == "nonfinite":
                return result * float("nan")
            if self.failure == "parameter":
                with torch.no_grad():
                    self.conv.bias.add_(1)
            return result
    generator = torch.Generator().manual_seed(77)
    def batch(size, width):
        clean = torch.randn((size, width), generator=generator)
        return {"noisy_audio": clean + 0.15 * torch.randn((size, width), generator=generator),
                "clean_audio": clean, "lengths": torch.full((size,), width, dtype=torch.long)}
    train, validation = batch(4, 64), batch(1, 105)
    for failure, expected in ((None, "passed"), ("oom", "out_of_memory"),
                              ("nonfinite", "failed"), ("parameter", "failed")):
        model = Toy(failure).eval()
        model.bn.train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        modes = [module.training for module in model.modules()]
        rng = capture_rng()
        result = probe_model(model, train, validation, epsilon=1e-12)
        after = capture_rng()
        assert result["status"] == expected and result["optimizer_steps"] == 0
        assert result["parameters_unchanged"] and result["buffers_restored"]
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        assert modes == [module.training for module in model.modules()]
        assert all(value.grad is None for value in model.parameters())
        assert rng["python"] == after["python"] and rng["numpy"] == after["numpy"]
        assert torch.equal(rng["torch_cpu"], after["torch_cpu"])
        assert all(torch.equal(a, b) for a, b in zip(rng["torch_cuda"], after["torch_cuda"]))
        if not failure:
            assert list(result["stages"]) == ["train_batch", "validation_utterance"]
            assert result["stages"]["train_batch"]["parameters_with_gradients"] > 0
            assert result["stages"]["validation_utterance"]["backward"] is False
        else:
            assert list(result["stages"]) == ["train_batch"]
    model = Toy()
    model.conv.weight.grad = torch.zeros_like(model.conv.weight)
    reject(lambda: probe_model(model, train, validation, epsilon=1e-12))
    # Exercise the actual CLI/report path with explicit synthetic-only loaders.
    from scripts import probe_training
    from speech_denoising.models import MODEL_REGISTRY
    manifest = json.loads(parent.read_bytes())
    widths = {row["utterance_id"]: row["samples"] for row in manifest["utterances"]}
    class FixtureView:
        def __init__(self, path, plan, split):
            self.utterance_ids = tuple(plan["split"][split + "_ids"])
        def __getitem__(self, index):
            identifier = self.utterance_ids[index]
            clean = torch.sin(torch.arange(widths[identifier], dtype=torch.float32) * 0.17)
            return {"utterance_id": identifier, "sample_rate": 16000, "clean_audio": clean,
                    "noisy_audio": clean + 0.1 * torch.cos(torch.arange(len(clean)) * 0.13)}
    with tempfile.TemporaryDirectory(prefix="cli_", dir=REPO_ROOT / "results/training_probe") as temporary:
        output = Path(temporary) / "toy.json"
        argv = ["probe_training", "--model", "gtcrn", "--manifest", str(parent),
                "--preflight", str(preflight), "--device", "cpu", "--output", str(output)]
        with patch.object(sys, "argv", argv), patch.object(probe_training, "TrainingSubset", FixtureView), \
                patch.dict(MODEL_REGISTRY, {"gtcrn": Toy}), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("Optimizer forbidden")):
                probe_training.main()
        stored = json.loads(output.read_bytes())
        assert stored["scope"] == "bounded_training_resource_probe"
        assert stored["result"]["status"] == "passed" and not stored["execution_ready"]
        assert not stored["quality_scores_generated"] and not stored["complete_validation"]
        assert not stored["optimizer_created"] and stored["result"]["optimizer_steps"] == 0
        assert stored["result"]["stages"]["train_batch"]["input_shape"] == [4, 64000]
        assert stored["result"]["stages"]["validation_utterance"]["input_shape"][0] == 1
        failure_output = Path(temporary) / "oom.json"
        argv[-1] = str(failure_output)
        with patch.object(sys, "argv", argv), patch.object(probe_training, "TrainingSubset", FixtureView), \
                patch.dict(MODEL_REGISTRY, {"gtcrn": lambda: Toy("oom")}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                probe_training.main()
            except SystemExit as error:
                assert error.code == 1
            else:
                raise AssertionError("Failed probe returned success")
        assert json.loads(failure_output.read_bytes())["result"]["status"] == "out_of_memory"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="training_probe_") as temporary:
        parent, preflight = metadata_checks(Path(temporary))
        print("PASS: current preflight binding, shared bounded selection, six-layer profile and tamper rejection")
        print("PASS: output ownership, atomic publication and early CLI refusal")
        if not args.metadata_only:
            tensor_checks(parent, preflight)
            print("PASS: toy forward/backward, unchanged parameters, restored buffers/modes/RNG and failure cleanup")
            print("PASS: CPU fixture CLI, configured batch/crop and diagnostic report; optimizer creation forbidden")
    print("Synthetic fixtures only; no real audio, speech-model updates or quality benchmark.")


if __name__ == "__main__":
    main()
