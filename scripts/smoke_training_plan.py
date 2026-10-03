"""Check planning rules with synthetic metadata; no audio or model imports."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.plan_training import model_settings
from speech_denoising.training.protocol import (
    build_training_plan, fingerprint, validate_protocol, validation_speakers,
)
from speech_denoising.utils.utils import REPO_ROOT, load_config


def expect_rejection(operation):
    try:
        operation()
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError("Invalid planning input was accepted")


def synthetic_manifest(source):
    rows = []
    for index in range(source["expected_counts"]["train"]):
        identifier = f"p{226 + index % 4:03d}_{index + 1:05d}"
        row = {"utterance_id": identifier, "samples": 16000 + index}
        for role in ("clean", "noisy"):
            row[f"{role}_path"] = f"audio/train/{role}/{identifier}.wav"
            row[f"{role}_sha256"] = "0" * 64
        rows.append(row)
    rows.sort(key=lambda row: row["utterance_id"])
    return {
        "schema_version": 1, "split": "train", "sample_rate": 16000,
        "protocol_complete": True, "source": deepcopy(source),
        "source_files": [f for f in source["files"] if f["path"].startswith("data/train-")],
        "count": len(rows), "rows_sha256": fingerprint(rows), "utterances": rows,
    }


def main():
    protocol = load_config("configs/training_protocol.yaml")
    validate_protocol(protocol)
    source = json.loads((REPO_ROOT / "configs/voicebank.json").read_text())
    manifest = synthetic_manifest(source)
    original = deepcopy(manifest)
    plans = {name: build_training_plan(manifest, source, protocol, name,
                                      model_settings(protocol, name))
             for name in protocol["models"]}
    assert manifest == original
    assert len({p["split_sha256"] for p in plans.values()}) == 1
    assert len({p["protocol_sha256"] for p in plans.values()}) == 1
    plan = plans["gtcrn"]
    assert plan == build_training_plan(manifest, source, protocol, "gtcrn",
                                       model_settings(protocol, "gtcrn"))
    split = plan["split"]
    assert not set(split["train_ids"]) & set(split["validation_ids"])
    assert set(split["train_ids"]) | set(split["validation_ids"]) == {
        row["utterance_id"] for row in manifest["utterances"]
    }
    assert not set(split["train_speakers"]) & set(split["validation_speakers"])
    assert plan["counts"]["train"] + plan["counts"]["validation"] == 11572
    assert len(split["validation_speakers"]) == 2
    assert not plan["execution_ready"] and not plan["audio_files_verified"]
    assert not plan["training_performed"] and not plan["quality_scores_generated"]
    speakers = split["train_speakers"] + split["validation_speakers"]
    assert validation_speakers(speakers, 42, 2) == validation_speakers(reversed(speakers), 42, 2)
    expect_rejection(lambda: validation_speakers(speakers, 42, len(speakers)))
    for key, value in (("split", "test"), ("sample_rate", 8000), ("count", 1),
                       ("rows_sha256", "0" * 64), ("protocol_complete", False)):
        bad = deepcopy(manifest)
        bad[key] = value
        expect_rejection(lambda: build_training_plan(bad, source, protocol, "gtcrn", {}))
    bad = deepcopy(manifest)
    bad["source_files"] = source["files"][:1]
    expect_rejection(lambda: build_training_plan(bad, source, protocol, "gtcrn", {}))
    for mutation in (
        lambda m: m["utterances"][0].update(samples=0),
        lambda m: m["utterances"][0].update(clean_path="../test/audio.wav"),
        lambda m: m["utterances"][0].update(noisy_sha256="invalid"),
        lambda m: m["utterances"][0].update(utterance_id="unknown_speaker"),
        lambda m: m["utterances"].__setitem__(1, deepcopy(m["utterances"][0])),
        lambda m: m["utterances"].reverse(),
    ):
        bad = deepcopy(manifest)
        mutation(bad)
        bad["rows_sha256"] = fingerprint(bad["utterances"])
        expect_rejection(lambda: build_training_plan(bad, source, protocol, "gtcrn", {}))
    for key, value in (("epochs", True), ("learning_rate", float("nan")),
                       ("seeds", [42, 42]), ("weight_decay", -1)):
        bad = deepcopy(protocol)
        bad["training"][key] = value
        expect_rejection(lambda: validate_protocol(bad))
    bad = deepcopy(protocol)
    bad["models"]["tfgridnet"] = "configs/tfgridnet_dns.yaml"
    expect_rejection(lambda: model_settings(bad, "tfgridnet"))
    assert plans["tfgridnet"]["model_settings"]["constructor_kwargs"]["n_layers"] == 6
    for name in protocol["models"]:
        result = subprocess.run([sys.executable, "-m", "scripts.plan_training",
                                 "--model", name, "--describe"], cwd=REPO_ROOT,
                                text=True, capture_output=True, check=True)
        description = json.loads(result.stdout)
        assert not description["execution_ready"] and description["model"] == name
    # Exercise new-file output and protected-path refusal with synthetic metadata.
    with tempfile.TemporaryDirectory() as folder:
        parent = Path(folder) / "train_manifest.json"
        parent.write_text(json.dumps(manifest), encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="plan_fixture_", dir=REPO_ROOT / "results") as outputs:
            output = Path(outputs) / "plan.json"
            command = [sys.executable, "-m", "scripts.plan_training", "--model", "gtcrn",
                       "--manifest", str(parent), "--output", str(output)]
            subprocess.run(command, cwd=REPO_ROOT, capture_output=True, check=True)
            contents = output.read_bytes()
            stored = json.loads(contents)
            assert stored["split_sha256"] == plan["split_sha256"]
            assert subprocess.run(command, cwd=REPO_ROOT, capture_output=True).returncode != 0
            assert output.read_bytes() == contents
            outside = Path(folder) / "historical_report.json"
            assert subprocess.run(command[:-1] + [str(outside)], cwd=REPO_ROOT,
                                  capture_output=True).returncode != 0
            assert not outside.exists()
    assert "torch" not in sys.modules
    print("PASS: shared speaker split, determinism, coverage and manifest rejection")
    print("PASS: three model configs, DNS-profile rejection and CLI output protection")
    print("Synthetic metadata only; no audio verification, model execution, download or training.")


if __name__ == "__main__":
    main()
