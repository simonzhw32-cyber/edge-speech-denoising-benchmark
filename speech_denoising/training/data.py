"""Bind a draft plan to the complete local train manifest and ordered subsets."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts.plan_training import model_settings
from speech_denoising.utils.utils import REPO_ROOT, load_config
from .protocol import build_training_plan, fingerprint


def load_training_inputs(manifest_path, model, config="configs/training_protocol.yaml"):
    path = Path(manifest_path).resolve(strict=True)
    data = path.read_bytes()
    manifest = json.loads(data)
    source = json.loads((REPO_ROOT / "configs/voicebank.json").read_text(encoding="utf-8"))
    protocol = load_config(config)
    settings = model_settings(protocol, model)
    plan = build_training_plan(manifest, source, protocol, model, settings)
    return path, data, manifest, plan


def validate_bound_plan(manifest, plan, source, protocol, settings):
    """Rebuild the entire plan; a self-consistent edited split is still rejected."""
    expected = build_training_plan(manifest, source, protocol, plan["model"], settings)
    if plan != expected:
        raise ValueError("Plan differs from the current manifest, protocol or model configuration")
    return expected


def subset_indices(manifest, plan, split):
    if split not in ("train", "validation"):
        raise ValueError("Training views accept train or validation only; test is reserved")
    rows = manifest["utterances"]
    if plan["split"]["parent_rows_sha256"] != fingerprint(rows):
        raise ValueError("Plan parent rows differ from the manifest")
    parent = [row["utterance_id"] for row in rows]
    training, validation = plan["split"]["train_ids"], plan["split"]["validation_ids"]
    if any(ids != sorted(set(ids)) or not ids for ids in (parent, training, validation)):
        raise ValueError("Subset IDs must be sorted, unique and nonempty")
    if set(training) & set(validation) or set(training) | set(validation) != set(parent):
        raise ValueError("Train/validation must partition the complete parent train set")
    lookup = {identifier: index for index, identifier in enumerate(parent)}
    return tuple(lookup[identifier] for identifier in plan["split"][split + "_ids"])


class TrainingSubset:
    """Complete utterances only; paired cropping remains the runner's responsibility.

    Byte hashes are rechecked by VoiceBankDataset on every access. A prior preflight
    report never substitutes for that check because local audio may have changed.
    """

    def __init__(self, manifest_path, plan, split, config="configs/training_protocol.yaml"):
        path, data, manifest, expected = load_training_inputs(manifest_path, plan["model"], config)
        if plan != expected:
            raise ValueError("Training view requires the canonical current draft plan")
        indices = subset_indices(manifest, plan, split)
        from speech_denoising.datasets.voicebank import VoiceBankDataset
        parent = VoiceBankDataset(path, split="train", verify_files=True, require_complete=True)
        if parent.metadata != manifest or path.read_bytes() != data:
            raise ValueError("Manifest changed while binding the dataset")
        self._parent = parent
        self._indices = indices
        self.utterance_ids = tuple(plan["split"][split + "_ids"])
        self.split = split
        self.plan_sha256 = fingerprint(plan)
        self.manifest_sha256 = hashlib.sha256(data).hexdigest()
        self._rows = deepcopy(manifest["utterances"])

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Training subset index out of range")
        parent_index = self._indices[index]
        if self._parent.rows[parent_index] != self._rows[parent_index]:
            raise ValueError("Dataset row changed after binding")
        sample = self._parent[parent_index]
        if sample["utterance_id"] != self.utterance_ids[index]:
            raise ValueError("Dataset order differs from the planned IDs")
        return sample
