"""Select epoch-boundary checkpoints using complete validation loss."""

import copy
import math
import re

from .protocol import fingerprint


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def experiment_identity(plan, seed):
    """Bind state to one metadata plan; this does not verify local audio."""
    if plan.get("scope") != "training_metadata_plan" or plan.get("status") != "draft":
        raise ValueError("Expected a draft training metadata plan")
    from .protocol import validate_protocol
    validate_protocol(plan["protocol"])
    for name in ("protocol", "split"):
        if fingerprint(plan[name]) != plan[name + "_sha256"]:
            raise ValueError(f"Changed {name} fingerprint")
    model = plan["model"]
    profile = "local_6layer" if model == "tfgridnet" else "native_default"
    settings = plan["model_settings"]
    if model not in ("gtcrn", "lisennet", "tfgridnet") or settings["profile"] != profile:
        raise ValueError("Use the random-weight training architecture, not the DNS profile")
    if settings["initialization"] != "random" or not _hash(settings["config_sha256"]):
        raise ValueError("Invalid random-initialization config")
    if model == "tfgridnet" and settings["constructor_kwargs"].get("n_layers") != 6:
        raise ValueError("Training uses six-layer TF-GridNet")
    if type(seed) is not int or seed not in plan["protocol"]["training"]["seeds"]:
        raise ValueError("Seed is outside the protocol")
    return {
        "model": model, "profile": profile, "seed": seed,
        "protocol_sha256": plan["protocol_sha256"],
        "split_sha256": plan["split_sha256"],
        "model_settings_sha256": fingerprint(settings),
    }


def validate_identity(identity):
    expected = {"model", "profile", "seed", "protocol_sha256", "split_sha256", "model_settings_sha256"}
    if not isinstance(identity, dict) or set(identity) != expected:
        raise ValueError("Invalid experiment identity fields")
    profile = "local_6layer" if identity["model"] == "tfgridnet" else "native_default"
    if identity["model"] not in ("gtcrn", "lisennet", "tfgridnet") or identity["profile"] != profile:
        raise ValueError("Invalid training model/profile")
    if type(identity["seed"]) is not int or identity["seed"] < 0:
        raise ValueError("Invalid training seed")
    if not all(_hash(identity[k]) for k in expected if k.endswith("sha256")):
        raise ValueError("Invalid experiment fingerprints")


def validate_report(report, identity, validation_ids, *, synthetic=False):
    validate_identity(identity)
    ids = list(validation_ids)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Expected nonempty unique validation IDs")
    if report.get("schema_version") != 1 or report.get("loss") != "negative_si_snr" or (
            report.get("reduction") != "utterance_mean"):
        raise ValueError("Wrong validation loss/report format")
    scope = "synthetic_fixture" if synthetic else "complete_train_validation"
    if report.get("scope") != scope or report.get("identity") != identity:
        raise ValueError("Wrong validation scope or experiment")
    if report.get("validation_ids_sha256") != fingerprint(ids) or (
            type(report.get("count")) is not int or report["count"] != len(ids)):
        raise ValueError("Incomplete or different validation set")
    if type(report.get("epoch")) is not int or report["epoch"] < 1:
        raise ValueError("Validation epoch must be a completed positive epoch")
    if not _hash(report.get("checkpoint_sha256")):
        raise ValueError("Validation must identify the checkpoint bytes")
    value = report.get("mean_loss")
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("Validation loss must be finite")


def select_checkpoint(best, candidate, identity, validation_ids, *, synthetic=False):
    """Return a copy of the best report; callers retain its named checkpoint."""
    validate_report(candidate, identity, validation_ids, synthetic=synthetic)
    if best is None:
        return copy.deepcopy(candidate)
    validate_report(best, identity, validation_ids, synthetic=synthetic)
    if candidate["epoch"] == best["epoch"] and candidate != best:
        raise ValueError("Conflicting reports for the same epoch")
    key = lambda report: (report["mean_loss"], report["epoch"])
    return copy.deepcopy(candidate if key(candidate) < key(best) else best)
