"""Verified GTCRN real-data epoch and independent validation coordination."""

import hashlib
import platform
import random
import subprocess

from speech_denoising.utils.utils import REPO_ROOT
from .data import TrainingSubset
from .probe import bind_preflight, source_hashes
from .protocol import fingerprint
from .real_store import (audit_run, checkpoint_path, create_run, persist_training,
                         persist_validation, run_path)
from .selection import execution_identity
from .state import load_training_state, save_training_state


SOURCE_FILES = [
    "configs/default.yaml", "configs/gtcrn.yaml", "configs/training_protocol.yaml",
    "configs/voicebank.json", "scripts/plan_training.py", "scripts/train.py",
    "scripts/validate.py", "speech_denoising/datasets/voicebank.py",
    "speech_denoising/assets/gtcrn.py", "speech_denoising/losses/loss.py",
    "speech_denoising/models/base.py", "speech_denoising/models/gtcrn/checkpoint.py",
    "speech_denoising/models/gtcrn/model.py",
    "speech_denoising/models/gtcrn/network.py", "speech_denoising/training/batching.py",
    "speech_denoising/training/data.py", "speech_denoising/training/probe.py",
    "speech_denoising/training/protocol.py", "speech_denoising/training/real_runner.py",
    "speech_denoising/training/real_store.py", "speech_denoising/training/runner.py",
    "speech_denoising/training/selection.py", "speech_denoising/training/state.py",
    "speech_denoising/training/validation.py", "speech_denoising/utils/utils.py",
]


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _git_state():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "changed_paths": sorted(status)}


def _runtime(device, deterministic):
    import numpy as np
    import torch
    result = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "platform": platform.platform(),
        "device": device, "cuda_build": torch.version.cuda,
        "deterministic_algorithms": deterministic, "tensor_dtype": "float32",
        "autocast": False,
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; there is no CPU fallback")
        properties = torch.cuda.get_device_properties(0)
        result.update(gpu=properties.name,
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      gpu_total_bytes=properties.total_memory)
    return result


def build_run_spec(manifest_path, preflight_path, *, seed, device, initialization,
                   initial_checkpoint=None, deterministic_algorithms=False):
    path, data, _, plan, preflight_bytes = bind_preflight(manifest_path, preflight_path, "gtcrn")
    checkpoint_sha256 = None
    initialization_record = {"mode": initialization, "checkpoint_source": None,
                             "checkpoint_sha256": None}
    if initialization == "pretrained":
        from speech_denoising.models.gtcrn.checkpoint import CHECKPOINT_SHA256, file_sha256
        if initial_checkpoint is None:
            raise ValueError("Pretrained fine-tuning requires --initial-checkpoint")
        checkpoint_sha256 = file_sha256(initial_checkpoint)
        if checkpoint_sha256 != CHECKPOINT_SHA256:
            raise ValueError("Use the pinned official GTCRN VCTK-DEMAND checkpoint")
        initialization_record.update(checkpoint_source="gtcrn_vctk_demand_official",
                                     checkpoint_sha256=checkpoint_sha256)
    elif initialization != "random" or initial_checkpoint is not None:
        raise ValueError("Random initialization must not receive an initial checkpoint")
    identity = execution_identity(plan, seed, initialization, checkpoint_sha256)
    training = plan["protocol"]["training"]
    settings = {name: training[name] for name in (
        "segment_samples", "batch_size", "epochs", "optimizer", "learning_rate",
        "weight_decay", "betas", "gradient_clip_norm", "precision")}
    settings.update(device=device, deterministic_algorithms=deterministic_algorithms)
    return {
        "schema_version": 1, "scope": "real_voicebank_training", "model": "gtcrn",
        "purpose": ("unified_training_from_scratch" if initialization == "random"
                    else "pretrained_finetuning_integration"),
        "identity": identity, "initialization": initialization_record,
        "plan": plan, "plan_sha256": fingerprint(plan), "settings": settings,
        "manifest_sha256": _sha256(data), "preflight_sha256": _sha256(preflight_bytes),
        "source_fingerprints": source_hashes(SOURCE_FILES), "git": _git_state(),
        "runtime": _runtime(device, deterministic_algorithms),
        "manifest_name": path.name,
        "fixed_test_used": False, "complete_validation_required": True,
    }


def _context(spec, *, initial_checkpoint=None, load_initial=False):
    import numpy as np
    import torch
    from speech_denoising.models.gtcrn.model import GTCRNModel
    seed = spec["identity"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(spec["settings"]["deterministic_algorithms"])
    device = torch.device("cuda:0" if spec["settings"]["device"] == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; there is no CPU fallback")
    model = GTCRNModel(**spec["plan"]["model_settings"]["constructor_kwargs"])
    if load_initial and spec["initialization"]["mode"] == "pretrained":
        if initial_checkpoint is None:
            raise ValueError("The first pretrained epoch requires --initial-checkpoint")
        from speech_denoising.models.gtcrn.checkpoint import load_gtcrn_checkpoint
        metadata = load_gtcrn_checkpoint(model, initial_checkpoint)
        if metadata["sha256"] != spec["initialization"]["checkpoint_sha256"]:
            raise ValueError("Initial checkpoint differs from the run specification")
    model.to(device=device, dtype=torch.float32)
    settings = spec["settings"]
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"],
                                  betas=tuple(settings["betas"]))
    return model, optimizer


def _verify_current_inputs(spec, manifest_path, preflight_path):
    path, data, _, plan, preflight_bytes = bind_preflight(manifest_path, preflight_path, "gtcrn")
    if (_sha256(data) != spec["manifest_sha256"] or
            _sha256(preflight_bytes) != spec["preflight_sha256"] or
            plan != spec["plan"] or source_hashes(SOURCE_FILES) != spec["source_fingerprints"]):
        raise ValueError("Manifest, preflight, plan or training source changed since run creation")
    return path, plan


def train_real_epoch(manifest_path, preflight_path, directory, epoch, *, seed, device,
                     initialization, initial_checkpoint=None, deterministic_algorithms=False):
    from .runner import train_one_epoch
    proposed = build_run_spec(manifest_path, preflight_path, seed=seed, device=device,
                              initialization=initialization, initial_checkpoint=initial_checkpoint,
                              deterministic_algorithms=deterministic_algorithms)
    run = run_path(directory)
    if run.exists():
        spec, records, reports = audit_run(run, require_selection=True)
        if proposed != spec:
            raise ValueError("Requested inputs or runtime differ from the existing run")
    else:
        if epoch != 1:
            raise ValueError("A new real-data run must start at epoch 1")
        spec, records, reports = proposed, [], []
    if type(epoch) is not int or not 1 <= epoch <= spec["settings"]["epochs"]:
        raise ValueError("Epoch is outside the bound training plan")
    if epoch != len(records) + 1:
        raise ValueError("Epoch already exists or skips saved progress")
    if len(reports) != len(records):
        raise ValueError("Run independent validation for the previous epoch first")
    model, optimizer = _context(spec, initial_checkpoint=initial_checkpoint,
                                load_initial=not records)
    dataset = TrainingSubset(manifest_path, spec["plan"], "train")
    step = 0
    if records:
        progress = load_training_state(checkpoint_path(run, epoch - 1), model, optimizer, spec["identity"])
        if progress["next_epoch"] != epoch or progress["global_step"] != records[-1]["global_step"]:
            raise ValueError("Checkpoint progress differs from the recorded run")
        if progress["checkpoint_sha256"] != records[-1]["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes changed during restoration")
        step = progress["global_step"]
    else:
        create_run(run, spec)
    summary = train_one_epoch(model, optimizer, dataset, dataset.utterance_ids,
                              seed=seed, epoch=epoch, settings=spec["settings"], global_step=step)
    digest = save_training_state(checkpoint_path(run, epoch), model, optimizer, spec["identity"],
                                 completed_epoch=epoch, global_step=summary["global_step"])
    return persist_training(run, spec, epoch, summary, digest)


def validate_real_epoch(manifest_path, preflight_path, directory, epoch, *, device):
    from .validation import run_validation
    run = run_path(directory)
    spec, records, _ = audit_run(run)
    if device != spec["settings"]["device"]:
        raise ValueError("Validation device must match the saved training runtime")
    path, plan = _verify_current_inputs(spec, manifest_path, preflight_path)
    if type(epoch) is not int or not 1 <= epoch <= len(records):
        raise ValueError("Validate an existing complete training epoch")
    model, optimizer = _context(spec)
    progress = load_training_state(checkpoint_path(run, epoch), model, optimizer, spec["identity"])
    if progress["completed_epoch"] != epoch or progress["global_step"] != records[epoch - 1]["global_step"]:
        raise ValueError("Checkpoint progress differs from the recorded epoch")
    dataset = TrainingSubset(path, plan, "validation")
    report = run_validation(model, dataset, spec["identity"], dataset.utterance_ids,
                            epoch=epoch, checkpoint_sha256=progress["checkpoint_sha256"],
                            synthetic=False, plan=plan)
    selection = persist_validation(run, spec, report)
    return report, selection
