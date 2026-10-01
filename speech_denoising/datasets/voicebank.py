"""Verified paired VoiceBank-DEMAND manifests with unchanged 16 kHz audio."""

import hashlib
import io
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class AudioPair(TypedDict):
    utterance_id: str
    noisy_audio: torch.Tensor
    clean_audio: torch.Tensor
    sample_rate: int


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows):
    content = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


class VoiceBankDataset(Dataset):
    SAMPLE_RATE = 16000
    TRAIN_COUNT = 11572
    TEST_COUNT = 824

    def __init__(self, manifest=None, split="test", verify_files=True, require_complete=True):
        if split not in {"train", "test", "validation"}:
            raise ValueError(f"Unsupported split: {split}")
        if manifest is None:
            raise ValueError("Provide a prepared manifest; run python -m scripts.prepare_data")
        self.manifest_path = Path(manifest).resolve()
        self.metadata = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.rows = self.metadata["utterances"]
        self.split = split
        self.verify_files = verify_files
        if self.metadata.get("schema_version") != 1 or self.metadata.get("split") != split:
            raise ValueError("Manifest schema/split mismatch")
        if self.metadata.get("sample_rate") != 16000:
            raise ValueError("Manifest must specify 16 kHz")
        if self.metadata.get("rows_sha256") != rows_sha256(self.rows):
            raise ValueError("Manifest rows fingerprint mismatch")
        ids = [r["utterance_id"] for r in self.rows]
        if ids != sorted(ids) or len(set(ids)) != len(ids) or not ids:
            raise ValueError("Manifest IDs must be sorted, unique and nonempty")
        if any(r["samples"] <= 0 for r in self.rows):
            raise ValueError("Manifest contains an empty utterance")
        expected = {"train": 11572, "test": 824}.get(split)
        self.protocol_complete = require_complete and bool(self.metadata.get("protocol_complete"))
        if require_complete:
            canonical = json.loads((Path(__file__).resolve().parents[2] / "configs/voicebank.json").read_text(encoding="utf-8"))
            if self.metadata.get("source") != canonical:
                raise ValueError("Manifest source differs from pinned dataset configuration")
            expected_files = [x for x in canonical["files"] if x["path"].startswith(f"data/{split}-")]
            if self.metadata.get("source_files") != expected_files:
                raise ValueError("Manifest parquet provenance differs from pinned split")
        if require_complete and (not self.protocol_complete or expected is None or len(ids) != expected):
            raise ValueError("A complete official train/test manifest is required")

    def __len__(self):
        return len(self.rows)

    def _path(self, relative):
        path = (self.manifest_path.parent / relative).resolve()
        if not path.is_relative_to(self.manifest_path.parent):
            raise ValueError("Audio path escapes manifest directory")
        return path

    def __getitem__(self, index) -> AudioPair:
        row = self.rows[index]
        signals = {}
        for role in ("noisy", "clean"):
            data = self._path(row[f"{role}_path"]).read_bytes()
            if self.verify_files and hashlib.sha256(data).hexdigest() != row[f"{role}_sha256"]:
                raise ValueError(f"Audio hash mismatch: {row['utterance_id']} {role}")
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            if sr != 16000 or audio.ndim != 1 or len(audio) != row["samples"]:
                raise ValueError(f"Audio format/length mismatch: {row['utterance_id']}")
            if not np.isfinite(audio).all():
                raise ValueError("Audio contains non-finite samples")
            signals[role] = torch.from_numpy(audio.copy())
        return {"utterance_id": row["utterance_id"], "sample_rate": 16000,
                "noisy_audio": signals["noisy"], "clean_audio": signals["clean"]}
