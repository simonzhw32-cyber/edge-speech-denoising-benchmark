"""Synthetic metadata/WAV checks; no real data, model updates or downloads."""

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
import wave

from scripts.plan_training import model_settings
from scripts.smoke_training_plan import synthetic_manifest
from speech_denoising.training.data import (TrainingSubset, load_training_inputs,
                                           subset_indices, validate_bound_plan)
from speech_denoising.training.preflight import verify_training_audio, write_preflight_report
from speech_denoising.training.protocol import build_training_plan, fingerprint
from speech_denoising.utils.utils import REPO_ROOT, load_config


def reject(operation):
    try:
        operation()
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return
    raise AssertionError("Invalid input was accepted")


def metadata_checks(folder):
    source = json.loads((REPO_ROOT / "configs/voicebank.json").read_text())
    protocol = load_config("configs/training_protocol.yaml")
    manifest = synthetic_manifest(source)
    parent = folder / "metadata_manifest.json"
    parent.write_text(json.dumps(manifest), encoding="utf-8")
    plans = {}
    for model in protocol["models"]:
        path, data, loaded, plan = load_training_inputs(parent, model)
        assert path == parent and data == parent.read_bytes() and loaded == manifest
        plans[model] = plan
        settings = model_settings(protocol, model)
        assert validate_bound_plan(manifest, plan, source, protocol, settings) == plan
        train = subset_indices(manifest, plan, "train")
        validation = subset_indices(manifest, plan, "validation")
        assert set(train).isdisjoint(validation) and set(train) | set(validation) == set(range(11572))
        assert not plan["execution_ready"] and not plan["audio_files_verified"]
        reject(lambda: subset_indices(manifest, plan, "test"))
        for mutate in (
            lambda p: p["model_settings"].update(initialization="pretrained"),
            lambda p: p["split"]["train_ids"].pop(),
            lambda p: p["split"]["validation_ids"].append(p["split"]["train_ids"][0]),
            lambda p: p.update(execution_ready=True),
            lambda p: p.update(protocol_sha256="0" * 64),
        ):
            bad = deepcopy(plan)
            mutate(bad)
            bad["split_sha256"] = fingerprint(bad["split"])
            reject(lambda: validate_bound_plan(manifest, bad, source, protocol, settings))
        bad = deepcopy(plan)
        a, b = bad["split"]["train_ids"][0], bad["split"]["validation_ids"][0]
        bad["split"]["train_ids"] = sorted([b] + bad["split"]["train_ids"][1:])
        bad["split"]["validation_ids"] = sorted([a] + bad["split"]["validation_ids"][1:])
        bad["split_sha256"] = fingerprint(bad["split"])
        reject(lambda: validate_bound_plan(manifest, bad, source, protocol, settings))
    assert len({p["split_sha256"] for p in plans.values()}) == 1
    assert plans["tfgridnet"]["model_settings"]["constructor_kwargs"]["n_layers"] == 6
    dns = deepcopy(protocol)
    dns["models"]["tfgridnet"] = "configs/tfgridnet_dns.yaml"
    reject(lambda: model_settings(dns, "tfgridnet"))
    bad = deepcopy(manifest)
    bad["split"] = "test"
    parent.write_text(json.dumps(bad), encoding="utf-8")
    reject(lambda: load_training_inputs(parent, "gtcrn"))
    parent.write_text(json.dumps(manifest), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="smoke_", dir=REPO_ROOT / "results/training_preflight") as outputs:
        output = Path(outputs) / "check.json"
        stored = {"scope": "synthetic_output_test", "execution_ready": False}
        assert write_preflight_report(output, stored) == output
        content = output.read_bytes()
        reject(lambda: write_preflight_report(output, stored))
        assert output.read_bytes() == content
        reject(lambda: write_preflight_report(Path(outputs) / "protected.json", stored,
                                             protected=(Path(outputs) / "protected.json",)))
        reject(lambda: write_preflight_report(folder / "outside.json", stored))
        reject(lambda: write_preflight_report(REPO_ROOT / "benchmark_reports/new.json", stored))
        # CLI guard errors must leave existing/protected output untouched.
        command = [sys.executable, "-m", "scripts.preflight_training", "--model", "gtcrn",
                   "--manifest", str(parent), "--output"]
        for dest in (output, folder / "outside.json", REPO_ROOT / "benchmark_reports/new.json"):
            result = subprocess.run(command + [str(dest)], cwd=REPO_ROOT, capture_output=True)
            assert result.returncode != 0
        assert output.read_bytes() == content and not (folder / "outside.json").exists()
        target = folder / "external"
        target.mkdir()
        link = Path(outputs) / "linked"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pass  # Windows may require a privilege unavailable on the test machine.
        else:
            reject(lambda: write_preflight_report(link / "report.json", stored))
            assert not (target / "report.json").exists()
    assert "torch" not in sys.modules and "soundfile" not in sys.modules


def wav_bytes(*, rate=16000, channels=1, samples=16, constant=False):
    import numpy as np
    # Exact PCM values permit equality checks with SoundFile and Torch loaders.
    audio = np.array([0 if constant else (1200 if i % 2 else -1200)
                      for i in range(samples)], dtype="<i2")
    if channels == 2:
        audio = np.column_stack((audio, audio)).reshape(-1)
    handle = io.BytesIO()
    with wave.open(handle, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(audio.tobytes())
    return handle.getvalue()


def audio_checks(folder, *, dataset_views=True):
    import numpy as np
    import soundfile  # Require the actual backend for the full fixture check.
    source = json.loads((REPO_ROOT / "configs/voicebank.json").read_text())
    source["expected_counts"]["train"] = 4
    protocol = load_config("configs/training_protocol.yaml")
    settings = model_settings(protocol, "gtcrn")
    manifest = synthetic_manifest(source)
    for row in manifest["utterances"]:
        row["samples"] = 16
        for role in ("noisy", "clean"):
            path = folder / row[role + "_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            data = wav_bytes()
            path.write_bytes(data)
            row[role + "_sha256"] = hashlib.sha256(data).hexdigest()
    manifest["rows_sha256"] = fingerprint(manifest["utterances"])
    parent = folder / "train_manifest.json"
    parent.write_text(json.dumps(manifest), encoding="utf-8")
    plan = build_training_plan(manifest, source, protocol, "gtcrn", settings)
    def verify(m=manifest, p=plan, progress=None):
        return verify_training_audio(parent, m, p, source, protocol, settings, progress=progress)
    events = []
    original_files = {p.relative_to(folder).as_posix(): p.read_bytes()
                      for p in folder.rglob("*") if p.is_file()}
    report = verify(progress=lambda n, total: events.append((n, total)))
    assert report["audio"]["verified_files"] == 8 and events == [(i, 4) for i in range(1, 5)]
    assert report["audio_files_verified"] and not report["plan"]["audio_files_verified"]
    assert not report["execution_ready"] and not report["model_executed"] and not report["test_audio_read"]
    assert {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in folder.rglob("*") if p.is_file()} == original_files
    import scripts.preflight_training as cli
    with tempfile.TemporaryDirectory(prefix="cli_", dir=REPO_ROOT / "results/training_preflight") as outputs:
        output = Path(outputs) / "report.json"
        inputs = (parent, parent.read_bytes(), deepcopy(manifest), deepcopy(plan))
        command = ["preflight", "--model", "gtcrn", "--manifest", str(parent), "--output", str(output)]
        with patch.object(cli, "load_training_inputs", return_value=inputs), patch.object(sys, "argv", command):
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main()
        saved = json.loads(output.read_text())
        assert saved["plan_sha256"] == fingerprint(plan) and not saved["execution_ready"]
        assert saved["audio"]["verified_pairs"] == 4 and saved["source_fingerprints"]
        # A changed input/plan at the end of the scan cannot publish a success.
        changed = deepcopy(inputs)
        changed[-1]["counts"]["train"] += 1
        blocked = Path(outputs) / "changed.json"
        with patch.object(cli, "load_training_inputs", side_effect=[inputs, changed]), patch.object(
                sys, "argv", command[:-1] + [str(blocked)]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    cli.main()
                except SystemExit as error:
                    assert error.code == 1
                else:
                    raise AssertionError("Changed input was accepted by CLI")
        assert not blocked.exists()
    row = manifest["utterances"][0]
    clean = folder / row["clean_path"]
    original = clean.read_bytes()
    clean.unlink()
    reject(verify)
    clean.write_bytes(original + b"changed")
    reject(verify)
    clean.write_bytes(original)
    for data in (wav_bytes(rate=8000), wav_bytes(channels=2), wav_bytes(samples=8),
                 wav_bytes(constant=True), b"invalid audio"):
        changed = deepcopy(manifest)
        clean.write_bytes(data)
        changed["utterances"][0]["clean_sha256"] = hashlib.sha256(data).hexdigest()
        changed["rows_sha256"] = fingerprint(changed["utterances"])
        parent.write_text(json.dumps(changed), encoding="utf-8")
        changed_plan = build_training_plan(changed, source, protocol, "gtcrn", settings)
        reject(lambda: verify(changed, changed_plan))
    clean.write_bytes(original)
    parent.write_text(json.dumps(manifest), encoding="utf-8")
    import speech_denoising.training.preflight as module
    for value in (float("nan"), float("inf"), 1e30):
        with patch.object(module, "_decode", return_value=(np.full(16, value, dtype=np.float32), 16000)):
            reject(verify)
    def change_manifest(n, total):
        if n == total:
            parent.write_text(parent.read_text() + " ", encoding="utf-8")
    reject(lambda: verify(progress=change_manifest))
    parent.write_text(json.dumps(manifest), encoding="utf-8")
    alias = folder / "original.wav"
    clean.rename(alias)
    try:
        clean.symlink_to(alias)
    except OSError:
        alias.rename(clean)
    else:
        reject(verify)
        clean.unlink()
        alias.rename(clean)
    if dataset_views:
        # Tiny synthetic source is injected only inside this test. Production keeps
        # the pinned 11,572-item source and does not accept a fixture CLI override.
        import speech_denoising.training.data as module_data
        with patch.object(module_data, "load_training_inputs", return_value=(
                parent, parent.read_bytes(), deepcopy(manifest), deepcopy(plan))):
            from speech_denoising.datasets.voicebank import VoiceBankDataset
            def fixture_loader(path, **kwargs):
                assert kwargs == {"split": "train", "verify_files": True, "require_complete": True}
                return VoiceBankDataset(path, split="train", verify_files=True, require_complete=False)
            with patch("speech_denoising.datasets.voicebank.VoiceBankDataset", side_effect=fixture_loader) as loader:
                train = TrainingSubset(parent, plan, "train")
                validation = TrainingSubset(parent, plan, "validation")
                assert set(train.utterance_ids).isdisjoint(validation.utterance_ids)
                assert len(train) + len(validation) == 4
                assert [train[i]["utterance_id"] for i in range(len(train))] == list(train.utterance_ids)
                sample = train[0]
                assert sample["clean_audio"].numel() == 16 and sample["noisy_audio"].numel() == 16
                reject(lambda: train[-1])
                reject(lambda: TrainingSubset(parent, plan, "test"))
                corrupt = folder / manifest["utterances"][train._indices[0]]["clean_path"]
                previous = corrupt.read_bytes()
                corrupt.write_bytes(previous + b"changed after preflight")
                reject(lambda: train[0])
                corrupt.write_bytes(previous)
                plan["split"]["train_ids"].clear()
                assert len(train) == 2
                train._parent.rows[train._indices[0]]["samples"] = 99
                reject(lambda: train[0])
                assert loader.call_args.kwargs == {"split": "train", "verify_files": True, "require_complete": True}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="No SoundFile or Torch required")
    args = parser.parse_args()
    (REPO_ROOT / "results/training_preflight").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="training_preflight_") as temporary:
        folder = Path(temporary)
        metadata_checks(folder)
        print("PASS: canonical three-model plans, shared split, test/DNS distinction and tamper rejection")
        print("PASS: new-report publication, input/output protection and CLI rejection")
        if not args.metadata_only:
            audio_checks(folder / "audio")
            print("PASS: local WAV hashes, formats, alignment, finite/silent references and manifest-change rejection")
            print("PASS: ordered train/validation views and unchanged full-utterance loading contract")
    print("Synthetic fixtures only; no real data, model updates, assets or quality benchmark.")


if __name__ == "__main__":
    main()
