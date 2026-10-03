"""Strict loading of the audited original TF-GridNet DNS checkpoint."""

from collections.abc import Mapping

import torch

from .assets import MODEL_ID, REVISION, CHECKPOINT_URL, verify_checkpoint_file
from .model import TFGridNetModel
from .profiles import DNS_MODEL_PARAMS, DNS_PARAMETERS, DNS_PROFILE, DNS_STATE_ENTRIES


def _prepare_dns_state_dict(payload, expected):
    """Validate the entire tensor inventory before any model assignment."""
    if not isinstance(payload, Mapping) or len(payload) != DNS_STATE_ENTRIES:
        raise ValueError("Expected the raw 362-entry ESPnet model state dict.")
    if any(not isinstance(name, str) or not name.startswith("separator.") for name in payload):
        raise ValueError("Only separator.* checkpoint keys are accepted.")
    state = {name.removeprefix("separator."): value for name, value in payload.items()}
    if set(state) != set(expected):
        raise ValueError("Checkpoint keys do not exactly match the DNS network.")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("Invalid checkpoint tensor: " + name)
        if tensor.dtype != torch.float32 or tensor.shape != expected[name].shape:
            raise ValueError("Checkpoint tensor shape/dtype mismatch: " + name)
        if not torch.isfinite(tensor).all().item():
            raise ValueError("Non-finite checkpoint tensor: " + name)
    if sum(t.numel() for t in state.values()) != DNS_PARAMETERS:
        raise ValueError("DNS checkpoint parameter count mismatch.")
    return state


def load_tfgridnet_dns_checkpoint(model, path):
    """Verify identity, safe deserialize, check every tensor, then load strictly.

    Load into the explicit CPU FP32 DNS profile, then move/cast if needed.
    No fallback to random weights or partial state matching is permitted.
    """
    if not isinstance(model, TFGridNetModel) or model.config != dict(DNS_MODEL_PARAMS):
        raise ValueError("Use build_tfgridnet_dns_model(); the six-layer default is incompatible.")
    expected = model.network.state_dict()
    if len(expected) != DNS_STATE_ENTRIES or sum(p.numel() for p in model.parameters()) != DNS_PARAMETERS:
        raise ValueError("Model structure differs from the pinned DNS profile.")
    if any(t.device.type != "cpu" or t.dtype != torch.float32 for t in expected.values()):
        raise ValueError("Load DNS weights into a CPU FP32 model before moving/casting.")
    digest = verify_checkpoint_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = _prepare_dns_state_dict(payload, expected)
    model.network.load_state_dict(state, strict=True)
    return {
        "model_id": MODEL_ID, "profile": DNS_PROFILE,
        "source_revision": REVISION, "source_url": CHECKPOINT_URL,
        "sha256": digest, "matched_state_entries": len(state),
        "parameter_count": DNS_PARAMETERS,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_dataset": "DNS Challenge 2020 synthetic recipe; external pretrained",
        "epoch_from_filename": 33,
        "license_card_declaration": "bsd-2-clause",
        "sample_rate": 16000,
        "sample_rate_basis": "Official recipe defaults to 16k; exact training audio manifests unavailable.",
        "mode": "offline_whole_utterance",
        "normalize_output_wav": False,
        "comparison_type": "external_pretrained; not controlled unified training",
        "audit_record": "docs/tfgridnet_checkpoint_audit.json",
    }
