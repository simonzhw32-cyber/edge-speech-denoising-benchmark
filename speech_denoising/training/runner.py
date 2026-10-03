"""One-epoch updates and fixture coordination; validation runs separately."""

import math
import random

import torch

from speech_denoising.losses.loss import EnhancementLoss
from .batching import collate_audio_pairs, paired_crop
from .fixture import fixture_audio, fixture_context, fixture_spec
from .run_store import (audit_run, checkpoint_path, create_run, persist_training,
                        run_path)
from .state import load_training_state, save_training_state


def train_one_epoch(model, optimizer, dataset, ids, *, seed, epoch, settings, global_step=0):
    """Use deterministic paired crops; no validation or checkpoint selection here."""
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or len(dataset) != len(ids):
        raise ValueError("A complete, ordered, unique training subset is required")
    batch_size, segment = settings["batch_size"], settings["segment_samples"]
    clip = settings["gradient_clip_norm"]
    if type(epoch) is not int or epoch < 1 or type(seed) is not int or seed < 0 or (
            type(batch_size) is not int or batch_size < 1 or type(segment) is not int or segment < 2):
        raise ValueError("Invalid epoch, seed or batch settings")
    if not math.isfinite(clip) or clip <= 0 or type(global_step) is not int or global_step < 0:
        raise ValueError("Invalid clipping or progress")
    device = next(model.parameters()).device
    order = list(range(len(ids)))
    random.Random(f"fixture_shuffle_v1:{seed}:{epoch}").shuffle(order)
    loss_function = EnhancementLoss(reduction="none")
    total, count = 0.0, 0
    model.train()
    for start in range(0, len(order), batch_size):
        samples = []
        for index in order[start:start + batch_size]:
            sample = dataset[index]
            if sample["utterance_id"] != ids[index]:
                raise ValueError("Training dataset order differs from the planned IDs")
            samples.append(paired_crop(sample, segment_samples=segment, seed=seed, epoch=epoch))
        batch = collate_audio_pairs(samples)
        if batch["noisy_audio"].dtype != torch.float32:
            raise ValueError("The draft update loop uses FP32")
        optimizer.zero_grad(set_to_none=True)
        estimate = model(batch["noisy_audio"].to(device))
        losses = loss_function(estimate, batch["clean_audio"].to(device), batch["lengths"].to(device))
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                     clip, error_if_nonfinite=True)
        optimizer.step()
        total += math.fsum(losses.detach().cpu().tolist())
        count += len(samples)
        global_step += 1
    return {"mean_loss": total / count, "count": count, "global_step": global_step}


def train_fixture_epoch(directory, epoch, seed=42):
    spec = fixture_spec(seed)
    if type(epoch) is not int or not 1 <= epoch <= spec["max_epochs"]:
        raise ValueError("This fixture supports epochs 1 and 2 only")
    run = run_path(directory)
    if run.exists():
        records, reports = audit_run(run, spec, require_selection=True)
    else:
        if epoch != 1:
            raise ValueError("A new fixture must start at epoch 1")
        records, reports = [], []
    if epoch != len(records) + 1:
        raise ValueError("Epoch already exists or skips saved progress")
    if len(reports) != len(records):
        raise ValueError("Run the independent validation command for the previous epoch first")
    # Build everything before creating a new run, so missing dependencies leave no output.
    model, optimizer = fixture_context(spec)
    dataset = fixture_audio(spec, "train")
    step = 0
    if records:
        progress = load_training_state(checkpoint_path(run, epoch - 1), model, optimizer, spec["identity"])
        if progress["next_epoch"] != epoch or progress["global_step"] != records[-1]["global_step"]:
            raise ValueError("Checkpoint progress differs from the saved record")
        if progress["checkpoint_sha256"] != records[-1]["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes changed during restoration")
        step = progress["global_step"]
    else:
        create_run(run, spec)
    summary = train_one_epoch(model, optimizer, dataset, spec["train_ids"], seed=seed,
                              epoch=epoch, settings=spec["settings"], global_step=step)
    digest = save_training_state(checkpoint_path(run, epoch), model, optimizer, spec["identity"],
                                 completed_epoch=epoch, global_step=summary["global_step"])
    return persist_training(run, spec, epoch, summary, digest)
