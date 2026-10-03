"""Small CPU GTCRN experiment for exercising entry points, not quality scores."""

import hashlib

from .protocol import fingerprint, validate_protocol
from speech_denoising.utils.utils import REPO_ROOT, load_config

TRAIN_IDS = ["fixture_train_0", "fixture_train_1", "fixture_train_2", "fixture_train_3"]
VALIDATION_IDS = ["fixture_validation_0", "fixture_validation_1"]


def fixture_spec(seed=42):
    protocol = load_config("configs/training_protocol.yaml")
    validate_protocol(protocol)
    if type(seed) is not int or seed not in protocol["training"]["seeds"]:
        raise ValueError("Use a seed declared by the draft protocol")
    entry = load_config("configs/gtcrn.yaml")["model"]
    if entry.get("name") != "gtcrn" or entry.get("checkpoint") is not None or entry.get("parameters"):
        raise ValueError("This fixture pins the native random-weight GTCRN constructor")
    paths = [
        "configs/training_protocol.yaml", "configs/gtcrn.yaml", "configs/default.yaml",
        "speech_denoising/utils/utils.py",
        "speech_denoising/models/base.py", "speech_denoising/models/gtcrn/model.py",
        "speech_denoising/models/gtcrn/network.py", "speech_denoising/losses/loss.py",
        "speech_denoising/training/batching.py", "speech_denoising/training/state.py",
        "speech_denoising/training/selection.py", "speech_denoising/training/validation.py",
        "speech_denoising/training/fixture.py", "speech_denoising/training/runner.py",
        "speech_denoising/training/run_store.py", "scripts/train.py", "scripts/validate.py",
    ]
    hashes = {name: hashlib.sha256((REPO_ROOT / name).read_text(encoding="utf-8").encode()).hexdigest()
              for name in paths}
    settings = {"scope": "synthetic_fixture", "model": "gtcrn", "initialization": "random",
                "segment_samples": 1024, "batch_size": 2, "sample_rate": 16000,
                "device": "cpu", "threads": 1, "deterministic_algorithms": True,
                "optimizer": protocol["training"]["optimizer"],
                "learning_rate": protocol["training"]["learning_rate"],
                "weight_decay": protocol["training"]["weight_decay"],
                "betas": protocol["training"]["betas"],
                "gradient_clip_norm": protocol["training"]["gradient_clip_norm"],
                "source_fingerprints": hashes}
    split = {"scope": "synthetic_fixture", "version": 1,
             "train_ids": TRAIN_IDS, "validation_ids": VALIDATION_IDS}
    identity = {"model": "gtcrn", "profile": "native_default", "seed": seed,
                "protocol_sha256": fingerprint(protocol), "split_sha256": fingerprint(split),
                "model_settings_sha256": fingerprint(settings)}
    return {"schema_version": 1, "scope": "synthetic_fixture", "identity": identity,
            "settings": settings, "train_ids": TRAIN_IDS.copy(), "validation_ids": VALIDATION_IDS.copy(),
            "steps_per_epoch": 2, "max_epochs": 2, "generator_version": 1,
            "execution_ready_for_real_data": False}


def fixture_context(spec):
    import random
    import numpy as np
    import torch
    from speech_denoising.models.gtcrn.model import GTCRNModel
    seed = spec["identity"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = GTCRNModel().to(device="cpu", dtype=torch.float32)
    settings = spec["settings"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
                                 betas=tuple(settings["betas"]))
    return model, optimizer


def fixture_audio(spec, split):
    import torch
    if split not in ("train", "validation"):
        raise ValueError("The fixture has no test set")
    ids = spec[split + "_ids"]
    widths = [1001, 1100, 1217, 900] if split == "train" else [1191, 901]
    generator = torch.Generator().manual_seed(spec["identity"]["seed"] + (100 if split == "train" else 200))
    pairs = []
    for index, (identifier, width) in enumerate(zip(ids, widths)):
        time = torch.arange(width, dtype=torch.float32) / 16000
        clean = .6 * torch.sin(2 * torch.pi * (190 + index * 47) * time)
        clean += .2 * torch.sin(2 * torch.pi * (430 + index * 31) * time)
        noisy = clean + .07 * torch.randn(width, generator=generator)
        pairs.append({"utterance_id": identifier, "sample_rate": 16000,
                      "noisy_audio": noisy, "clean_audio": clean})
    return pairs
