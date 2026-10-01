"""Strict, checksum-verified loading of the pinned official VCTK-DEMAND weights."""

import hashlib
from pathlib import Path

import torch

REVISION = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
CHECKPOINT_SHA256 = "a0f0e04421d9fc1efe734d44191f5dc4b973ce59d6546af8d64ad9167e2a9944"
CHECKPOINT_URL = (
    f"https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/{REVISION}/"
    "checkpoints/model_trained_on_vctk.tar"
)
DEMO_URL = f"https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/{REVISION}/test_wavs/mix.wav"
DEMO_SHA256 = "8d47e1d03eeb457c2549be79c8ec33a349ccd79f21c3add05f946f8f760c5a99"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gtcrn_checkpoint(model, path):
    """Load official network keys into model.network; reject partial matches."""
    path = Path(path)
    actual_hash = file_sha256(path)
    if actual_hash != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint checksum mismatch. Use the pinned official VCTK file.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Expected the official checkpoint's 'model' state dict.")
    expected = model.network.state_dict()
    if set(state) != set(expected):
        raise ValueError("Checkpoint keys do not exactly match the GTCRN network.")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape:
            raise ValueError(f"Invalid checkpoint tensor: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    model.network.load_state_dict(state, strict=True)
    return {
        "source_url": CHECKPOINT_URL,
        "source_revision": REVISION,
        "sha256": actual_hash,
        "training_dataset": "VCTK-DEMAND (upstream pretrained)",
        "epoch": payload.get("epoch"),
        "matched_state_entries": len(state),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
