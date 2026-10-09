"""FP32 micro-batch-one updates for the six-layer TF-GridNet pilot."""

import math
import random
import torch

from speech_denoising.losses.loss import EnhancementLoss
from .batching import collate_audio_pairs, paired_crop


def train_accumulated_epoch(model, optimizer, dataset, ids, *, seed, epoch,
                            settings, global_step=0):
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or len(dataset) != len(ids):
        raise ValueError("A complete ordered unique training subset is required")
    group_size = settings["batch_size"]
    segment, clip = settings["segment_samples"], settings["gradient_clip_norm"]
    if (type(group_size) is not int or group_size != 4 or
            type(settings.get("micro_batch_size")) is not int or settings["micro_batch_size"] != 1 or
            type(settings.get("gradient_accumulation_steps")) is not int or
            settings["gradient_accumulation_steps"] != 4 or
            settings.get("accumulation_reduction") != "actual_group_utterance_mean" or
            settings.get("precision") != "float32"):
        raise ValueError("Expected FP32 micro-batch 1 and accumulation 4")
    if (type(epoch) is not int or epoch < 1 or type(seed) is not int or seed < 0 or
            type(segment) is not int or segment < 2 or not math.isfinite(clip) or clip <= 0 or
            type(global_step) is not int or global_step < 0):
        raise ValueError("Invalid crop, clipping or progress settings")
    device = next(model.parameters()).device
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("TF-GridNet training requires FP32 parameters")
    order = list(range(len(ids)))
    random.Random(f"fixture_shuffle_v1:{seed}:{epoch}").shuffle(order)
    criterion = EnhancementLoss(reduction="none")
    total, count = 0.0, 0
    model.train()
    for start in range(0, len(order), group_size):
        group = order[start:start + group_size]
        optimizer.zero_grad(set_to_none=True)
        try:
            for index in group:
                sample = dataset[index]
                if sample["utterance_id"] != ids[index]:
                    raise ValueError("Training dataset order differs from planned IDs")
                batch = collate_audio_pairs([paired_crop(
                    sample, segment_samples=segment, seed=seed, epoch=epoch)])
                if batch["noisy_audio"].dtype != torch.float32:
                    raise ValueError("Expected FP32 audio")
                estimate = model(batch["noisy_audio"].to(device))
                losses = criterion(estimate, batch["clean_audio"].to(device),
                                   batch["lengths"].to(device))
                # The final partial group is normalized by its actual size.
                (losses.sum() / len(group)).backward()
                total += math.fsum(losses.detach().cpu().tolist())
                count += 1
                del estimate, losses, batch
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), clip, error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
        except BaseException:
            optimizer.zero_grad(set_to_none=True)
            raise
        if global_step % 100 == 0:
            print(f"TF-GridNet epoch {epoch}: {count}/{len(ids)} utterances; "
                  f"optimizer steps {global_step}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    return {"mean_loss": total / count, "count": count, "global_step": global_step}
