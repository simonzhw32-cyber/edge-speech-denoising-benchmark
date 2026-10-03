"""Validate metadata and plan a speaker-disjoint split of the official train set."""

from copy import deepcopy
import hashlib
import json
import math
import re


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_protocol(protocol):
    if protocol.get("schema_version") != 1 or protocol.get("status") != "draft":
        raise ValueError("Expected a schema-1 draft training protocol")
    if protocol.get("comparison_type") != "unified_training_from_scratch":
        raise ValueError("This plan is for unified training from random weights")
    if protocol.get("sample_rate") != 16000:
        raise ValueError("Training protocol must use 16 kHz")
    split = protocol["split"]
    if split["algorithm"] != "speaker_sha256_v1":
        raise ValueError("Unknown speaker split algorithm")
    _integer(split["seed"], "split.seed", 0)
    _integer(split["validation_speakers"], "split.validation_speakers")
    if set(protocol["models"]) != {"gtcrn", "lisennet", "tfgridnet"}:
        raise ValueError("The protocol must specify all three model configurations")
    training = protocol["training"]
    seeds = training["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be a nonempty unique list")
    for seed in seeds:
        _integer(seed, "training seed", 0)
    for name in ("segment_samples", "batch_size", "epochs"):
        _integer(training[name], f"training.{name}")
    for name in ("learning_rate", "gradient_clip_norm"):
        value = training[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{name} must be finite and positive")
    weight_decay = training["weight_decay"]
    if type(weight_decay) not in (int, float) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    betas = training["betas"]
    if not isinstance(betas, list) or len(betas) != 2 or any(
        type(v) not in (int, float) or not 0 <= v < 1 for v in betas
    ):
        raise ValueError("AdamW betas must be two numbers in [0, 1)")
    expected = {
        "initialization": "random", "short_utterance": "right_zero_pad_with_valid_length",
        "drop_last": False, "optimizer": "AdamW", "precision": "float32",
        "scheduler": "none", "early_stopping": False, "device": "unselected",
    }
    if any(training.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported draft training setting")
    if protocol["loss"] != {
        "name": "negative_si_snr", "zero_mean": True, "epsilon": 1e-12,
        "reduction": "utterance_mean", "padded_samples": "excluded",
        "silent_clean_reference": "error",
    }:
        raise ValueError("Unsupported draft loss contract")
    if protocol["validation"] != {
        "segment": "complete_utterance", "batch_size": 1, "selection_metric": "loss",
        "selection_direction": "minimize", "tie_break": "earliest_epoch",
        "frequency": "every_epoch_via_separate_entrypoint",
    }:
        raise ValueError("Unsupported validation/selection policy")
    if protocol["test"] != {
        "split": "fixed_824", "use": "final_selected_checkpoint_only", "tuning": "forbidden"
    }:
        raise ValueError("Test must be reserved for final checkpoint evaluation")


def validation_speakers(speakers, seed, count):
    _integer(seed, "split seed", 0)
    _integer(count, "validation speaker count")
    speakers = sorted(set(speakers))
    if count >= len(speakers):
        raise ValueError("Holdout must leave at least one speaker for training")
    # Hash ranking avoids Python hash randomization and RNG-version dependence.
    ranked = sorted(speakers, key=lambda s: (
        hashlib.sha256(f"voicebank-speaker-split-v1:{seed}:{s}".encode()).hexdigest(), s
    ))
    return sorted(ranked[:count])


def validate_train_manifest(manifest, source):
    if (manifest.get("schema_version") != 1 or manifest.get("split") != "train"
            or manifest.get("sample_rate") != 16000
            or manifest.get("protocol_complete") is not True):
        raise ValueError("A complete prepared 16 kHz train manifest is required; test is forbidden")
    if manifest.get("source") != source:
        raise ValueError("Manifest source differs from the pinned dataset configuration")
    expected_files = [f for f in source["files"] if f["path"].startswith("data/train-")]
    if not expected_files or manifest.get("source_files") != expected_files:
        raise ValueError("Train parquet provenance mismatch")
    rows = manifest["utterances"]
    if len(rows) != source["expected_counts"]["train"] or manifest.get("count") != len(rows):
        raise ValueError("Train utterance count mismatch")
    if manifest.get("rows_sha256") != fingerprint(rows):
        raise ValueError("Manifest rows fingerprint mismatch")
    ids = [row["utterance_id"] for row in rows]
    if ids != sorted(ids) or len(set(ids)) != len(ids) or not ids:
        raise ValueError("Train IDs must be sorted, unique and nonempty")
    speakers = {}
    for row in rows:
        identifier = row["utterance_id"]
        match = re.fullmatch(r"(p[0-9]{3})_[0-9]+", identifier)
        if not match:
            raise ValueError(f"Cannot identify VoiceBank speaker: {identifier}")
        _integer(row["samples"], "utterance samples")
        for role in ("noisy", "clean"):
            if row[f"{role}_path"] != f"audio/train/{role}/{identifier}.wav":
                raise ValueError(f"Unexpected train audio path: {identifier} {role}")
            if not re.fullmatch(r"[0-9a-f]{64}", row[f"{role}_sha256"]):
                raise ValueError(f"Invalid audio hash: {identifier} {role}")
        speakers[identifier] = match[1]
    return speakers


def build_training_plan(manifest, source, protocol, model, model_settings):
    validate_protocol(protocol)
    if model not in protocol["models"]:
        raise ValueError(f"Unknown model: {model}")
    speakers = validate_train_manifest(manifest, source)
    held_out = validation_speakers(speakers.values(), protocol["split"]["seed"],
                                  protocol["split"]["validation_speakers"])
    validation_ids = [identifier for identifier, speaker in speakers.items() if speaker in held_out]
    training_ids = [identifier for identifier, speaker in speakers.items() if speaker not in held_out]
    split = {
        "algorithm": protocol["split"]["algorithm"], "seed": protocol["split"]["seed"],
        "parent_rows_sha256": manifest["rows_sha256"],
        "train_speakers": sorted(set(speakers.values()) - set(held_out)),
        "validation_speakers": held_out,
        "train_ids": training_ids, "validation_ids": validation_ids,
    }
    return {
        "schema_version": 1, "scope": "training_metadata_plan", "status": "draft",
        "model": model, "model_settings": deepcopy(model_settings),
        "protocol": deepcopy(protocol), "protocol_sha256": fingerprint(protocol),
        "source": deepcopy(source), "split": split, "split_sha256": fingerprint(split),
        "counts": {"parent_train": len(speakers), "train": len(training_ids),
                   "validation": len(validation_ids)},
        "audio_files_verified": False, "training_performed": False,
        "checkpoint_loaded": False, "quality_scores_generated": False,
        "execution_ready": False,
    }
