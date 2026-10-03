"""Independent complete-utterance validation; never uses the fixed test set."""

import math

import torch

from speech_denoising.losses.loss import EnhancementLoss
from .protocol import fingerprint
from .selection import validate_report
from .state import capture_rng, restore_rng


def run_validation(model, dataset, identity, validation_ids, *, epoch,
                   checkpoint_sha256, synthetic=False, plan=None):
    """Compute macro negative SI-SNR, batch one, without modifying train state.

    Production callers must supply a verified full-train dataset subset, with
    IDs from the metadata plan. Synthetic fixtures require an explicit flag.
    A checkpoint hash labels the state that the caller loaded before this call.
    """
    ids = list(validation_ids)
    report = {
        "schema_version": 1, "loss": "negative_si_snr", "reduction": "utterance_mean",
        "scope": "synthetic_fixture" if synthetic else "complete_train_validation",
        "identity": identity.copy(), "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_ids_sha256": fingerprint(ids), "count": len(ids), "mean_loss": 0.0,
    }
    validate_report(report, identity, ids, synthetic=synthetic)
    if len(dataset) != len(ids):
        raise ValueError("Validation subset must cover every planned validation utterance")
    if not synthetic:
        from speech_denoising.datasets.voicebank import VoiceBankDataset
        from torch.utils.data import Subset
        parent = dataset.dataset if isinstance(dataset, Subset) else None
        if not isinstance(parent, VoiceBankDataset) or parent.split != "train" or not (
                parent.verify_files and parent.protocol_complete):
            raise ValueError("Use a subset of the verified complete train dataset, never test")
        from .protocol import build_training_plan
        from .selection import experiment_identity
        if plan is None:
            raise ValueError("Production validation requires the parent training plan")
        rebuilt = build_training_plan(parent.metadata, parent.metadata["source"],
                                      plan["protocol"], plan["model"], plan["model_settings"])
        if experiment_identity(rebuilt, identity["seed"]) != identity or (
                rebuilt["split"]["validation_ids"] != ids):
            raise ValueError("Validation subset differs from the recomputed train-only split")
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    device = (parameters or buffers or [torch.empty(0)])[0].device
    if any(value.device != device for value in parameters + buffers):
        raise ValueError("Validation model must reside on one device")
    loss = EnhancementLoss(reduction="none")
    modes = [(module, module.training) for module in model.modules()]
    rng = capture_rng()
    values = []
    try:
        model.eval()
        with torch.inference_mode():
            for index, expected_id in enumerate(ids):
                sample = dataset[index]
                if sample["utterance_id"] != expected_id or sample["sample_rate"] != 16000:
                    raise ValueError("Validation ID order or sample rate differs from plan")
                if "valid_samples" in sample or "crop_start" in sample:
                    raise ValueError("Validation requires complete utterances, not crops")
                noisy, clean = sample["noisy_audio"], sample["clean_audio"]
                if noisy.ndim != 1 or noisy.shape != clean.shape or noisy.numel() < 2:
                    raise ValueError("Expected aligned mono utterances")
                if noisy.dtype != torch.float32 or clean.dtype != torch.float32:
                    raise ValueError("Validation uses FP32 audio")
                if not torch.isfinite(noisy).all() or not torch.isfinite(clean).all():
                    raise ValueError("Non-finite validation audio")
                estimate = model(noisy[None].to(device))
                value = loss(estimate, clean[None].to(device)).item()
                values.append(value)
        report["mean_loss"] = math.fsum(values) / len(values)
        validate_report(report, identity, ids, synthetic=synthetic)
        return report
    finally:
        for module, flag in modes:
            module.training = flag
        restore_rng(rng)
