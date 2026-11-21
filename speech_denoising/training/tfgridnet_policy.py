"""Small, reproducible quality-proxy trainer for TF-GridNet compute policies.

This is deliberately separate from the benchmark protocol. It trains on a
random subset of the verified VoiceBank training manifest so that checkpointing
and precision policies can be compared at an equal optimizer-step budget.
"""

from __future__ import annotations

from collections import OrderedDict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import yaml

from speech_denoising.datasets.voicebank import VoiceBankDataset
from speech_denoising.losses.loss import EnhancementLoss
from speech_denoising.metrics.audio_metrics import compute_audio_metrics
from speech_denoising.models.tfgridnet.model import TFGridNetModel
from speech_denoising.utils.utils import REPO_ROOT


def install_checkpointing(network, blocks_to_checkpoint) -> None:
    """Checkpoint selected TF-GridNet blocks in place.

    ``blocks_to_checkpoint`` may be an integer (the first N blocks) or an
    iterable of zero-based block indices.
    """
    from speech_denoising.models.tfgridnet import network as netmod

    if isinstance(blocks_to_checkpoint, int):
        selected = set(range(blocks_to_checkpoint))
    else:
        selected = {int(index) for index in blocks_to_checkpoint}
    if not selected:
        return
    if min(selected) < 0 or max(selected) >= len(network.blocks):
        raise ValueError("checkpoint index outside the block range")
    blocks = network.blocks

    def forward(self, input, ilens, additional=None):
        n_samples = input.shape[1]
        if self.n_imics == 1:
            input = input[..., None]
        mix_std = torch.std(input, dim=(1, 2), keepdim=True).clamp_min(1e-8)
        input = input / mix_std
        batch = self.enc(input, ilens)[0]
        batch0 = batch.transpose(1, 2)
        batch = torch.cat((batch0.real, batch0.imag), dim=1)
        batch = self.conv(batch)
        for index in range(self.n_layers):
            if index in selected:
                batch = cp.checkpoint(blocks[index], batch, use_reentrant=False)
            else:
                batch = blocks[index](batch)
        batch = self.deconv(batch)
        batch = batch.view(batch.shape[0], self.n_srcs, 2, batch.shape[2], batch.shape[3])
        batch = netmod.new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))
        batch = self.dec(batch.view(-1, batch.shape[2], batch.shape[3]), ilens)[0]
        batch = self.pad2(batch.view(batch.shape[0], self.n_srcs, -1), n_samples)
        batch = batch * mix_std
        return [batch[:, src] for src in range(self.n_srcs)], ilens, OrderedDict()

    network.forward = forward.__get__(network, type(network))


def _crop(audio, length: int, rng: random.Random) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < length:
        repeats = int(math.ceil(length / max(1, len(audio))))
        audio = np.tile(audio, repeats)
    start = rng.randrange(0, len(audio) - length + 1)
    return audio[start:start + length]


def _crop_pair(noisy, clean, length: int, rng: random.Random):
    """Crop noisy and clean at the same offset; wrap-pad short pairs."""
    noisy = np.asarray(noisy, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    if len(noisy) < length:
        repeats = int(math.ceil(length / max(1, len(noisy))))
        noisy = np.tile(noisy, repeats)
        clean = np.tile(clean, repeats)
    start = rng.randrange(0, len(noisy) - length + 1)
    return noisy[start:start + length], clean[start:start + length]


def _batch(samples, length: int, rng: random.Random, device: torch.device):
    pairs = [_crop_pair(item["noisy_audio"], item["clean_audio"], length, rng)
             for item in samples]
    noisy = np.stack([pair[0] for pair in pairs])
    clean = np.stack([pair[1] for pair in pairs])
    return (
        torch.from_numpy(noisy).to(device),
        torch.from_numpy(clean).to(device),
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _validate(model, dataset, indices, limit: int | None, device: torch.device, metrics):
    selected = list(indices)
    if limit is not None and len(selected) > limit:
        selected = selected[::max(1, len(selected) // limit)][:limit]
    values = {name: [] for name in metrics}
    failures = 0
    model.eval()
    with torch.no_grad():
        for index in selected:
            sample = dataset[index]
            noisy_np = np.asarray(sample["noisy_audio"], dtype=np.float32)
            clean_np = np.asarray(sample["clean_audio"], dtype=np.float32)
            noisy = torch.from_numpy(noisy_np[None]).to(device)
            enhanced = model(noisy)[0].cpu().numpy()
            result = compute_audio_metrics(clean_np, enhanced, noisy_np, metrics=metrics)
            for name in metrics:
                value = result["values"].get(name)
                if value is not None and math.isfinite(float(value)):
                    values[name].append(float(value))
                else:
                    failures += 1
    model.train()
    return {
        "count": len(selected),
        "failures": failures,
        "metrics": {
            name: (float(np.mean(scores)) if scores else None)
            for name, scores in values.items()
        },
    }


def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest_path = Path(args.manifest).resolve(strict=True)
    dataset = VoiceBankDataset(manifest_path, split="train", verify_files=not args.skip_audio_verify,
                               require_complete=True)
    total = len(dataset)
    order = list(range(total))
    random.Random(args.seed).shuffle(order)
    validation_size = min(args.validation_size, max(1, total // 10))
    validation_indices = sorted(order[:validation_size])
    train_indices = order[validation_size:]
    if args.train_pool:
        train_indices = train_indices[:args.train_pool]

    config = yaml.safe_load((REPO_ROOT / "configs/tfgridnet.yaml").read_text(encoding="utf-8"))
    params = dict(config["model_params"])
    params["stride"] = args.stride
    model = TFGridNetModel(**params).cuda().float()
    install_checkpointing(model.network, args.checkpoint_blocks)
    if args.subblock_intra or args.subblock_inter or args.subblock_attention:
        install_subblock_checkpointing(
            model.network,
            intra=args.subblock_intra,
            inter=args.subblock_inter,
            attention=args.subblock_attention,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.0
    )
    loss_function = EnhancementLoss()
    device = torch.device("cuda")

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("run directory must be new or empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    _write_json(run_dir / "run.json", {
        "kind": "tfgridnet_policy_proxy",
        "args": serializable_args,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_pool": len(train_indices),
        "validation": len(validation_indices),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "git_commit": os.popen("git rev-parse HEAD").read().strip() or None,
        "git_dirty": bool(os.popen("git status --porcelain").read().strip()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    curve_path = run_dir / "learning_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "seconds", "peak_alloc_gib", "peak_reserved_gib"])
        rng = random.Random(args.seed)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        running = 0.0
        window_count = 0
        for step in range(1, args.steps + 1):
            samples = [dataset[train_indices[rng.randrange(len(train_indices))]]
                       for _ in range(args.batch_size)]
            noisy, clean = _batch(samples, args.segment_samples, rng, device)
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            if args.amp == "fp16":
                with torch.autocast("cuda", dtype=torch.float16):
                    enhanced = model(noisy)
                    loss = loss_function(enhanced.float(), clean)
            else:
                enhanced = model(noisy)
                loss = loss_function(enhanced, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_started
            running += float(loss.detach())
            window_count += 1
            if step % args.log_every == 0 or step == 1:
                print(f"step {step}/{args.steps} loss={running / window_count:.5f} "
                      f"sec={elapsed:.2f} mem={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
                running = 0.0
                window_count = 0
            writer.writerow([step, float(loss.detach()), elapsed,
                             torch.cuda.max_memory_allocated() / 2**30,
                             torch.cuda.max_memory_reserved() / 2**30])
            handle.flush()
            if args.validation_every and step % args.validation_every == 0:
                result = _validate(model, dataset, validation_indices, args.validation_limit,
                                   device, ("pesq", "stoi", "si_snr", "si_snr_improvement"))
                _write_json(run_dir / f"validation_step_{step:06d}.json", result)
                print("validation", json.dumps(result, sort_keys=True), flush=True)
                # Validation holds full-utterance activations. On a 16 GB card
                # the retained blocks can push reserved memory into thrash;
                # release them only when the allocator is near the limit.
                if torch.cuda.is_available() and torch.cuda.memory_reserved() > 14 * 2**30:
                    torch.cuda.empty_cache()
            if args.max_seconds and time.perf_counter() - started >= args.max_seconds:
                print(f"stopping at step {step}: max_seconds reached", flush=True)
                break

    final = _validate(model, dataset, validation_indices, args.validation_limit, device,
                      ("pesq", "stoi", "si_snr", "si_snr_improvement"))
    _write_json(run_dir / "validation_final.json", final)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "parameters": sum(p.numel() for p in model.parameters()),
    }, run_dir / "checkpoint.pt")
    summary = {
        "run_dir": str(run_dir),
        "steps": args.steps,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "validation": final,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _subblock_intra(self, x):
    """Intra-frame RNN sub-block, factored for selective checkpointing."""
    B, C, T, Q = x.shape
    input_ = x
    intra_rnn = self.intra_norm(input_)
    intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
    intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
    intra_rnn = intra_rnn.transpose(1, 2)
    intra_rnn, _ = self.intra_rnn(intra_rnn)
    intra_rnn = intra_rnn.transpose(1, 2)
    intra_rnn = self.intra_linear(intra_rnn)
    intra_rnn = intra_rnn.view(B, T, C, Q)
    intra_rnn = intra_rnn.transpose(1, 2).contiguous()
    return intra_rnn + input_


def _subblock_inter(self, x):
    """Inter-frame RNN sub-block, factored for selective checkpointing."""
    B, C, T, Q = x.shape
    input_ = x
    inter_rnn = self.inter_norm(input_)
    inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
    inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
    inter_rnn = inter_rnn.transpose(1, 2)
    inter_rnn, _ = self.inter_rnn(inter_rnn)
    inter_rnn = inter_rnn.transpose(1, 2)
    inter_rnn = self.inter_linear(inter_rnn)
    inter_rnn = inter_rnn.view(B, Q, C, T)
    inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()
    return inter_rnn + input_


def _subblock_attention(self, x):
    """Multi-head attention sub-block, factored for selective checkpointing."""
    B, C, T, Q = x.shape
    input_ = x
    all_q, all_k, all_v = [], [], []
    for head in range(self.n_head):
        all_q.append(self["attn_conv_Q_%d" % head](input_))
        all_k.append(self["attn_conv_K_%d" % head](input_))
        all_v.append(self["attn_conv_V_%d" % head](input_))
    q = torch.cat(all_q, dim=0).transpose(1, 2).flatten(start_dim=2)
    k = torch.cat(all_k, dim=0).transpose(1, 2).flatten(start_dim=2)
    v = torch.cat(all_v, dim=0).transpose(1, 2)
    old_shape = v.shape
    v = v.flatten(start_dim=2)
    emb_dim = q.shape[-1]
    attn = torch.matmul(q, k.transpose(1, 2)) / (emb_dim ** 0.5)
    attn = F.softmax(attn, dim=2)
    v = torch.matmul(attn, v).reshape(old_shape).transpose(1, 2)
    out = v.view(self.n_head, B, v.shape[1], T, Q).transpose(0, 1)
    out = out.contiguous().view(B, self.n_head * v.shape[1], T, Q)
    out = self["attn_concat_proj"](out)
    return out + input_


def _subblock_forward(self, x):
    B, C, old_T, old_Q = x.shape
    import math as _math
    T = _math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
    Q = _math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
    x = F.pad(x, (0, Q - old_Q, 0, T - old_T))
    intra, inter, attention = self._subblock_flags
    x = cp.checkpoint(_subblock_intra, self, x, use_reentrant=False) if intra else _subblock_intra(self, x)
    x = cp.checkpoint(_subblock_inter, self, x, use_reentrant=False) if inter else _subblock_inter(self, x)
    x = x[..., :old_T, :old_Q]
    x = cp.checkpoint(_subblock_attention, self, x, use_reentrant=False) if attention else _subblock_attention(self, x)
    return x


def install_subblock_checkpointing(network, intra=False, inter=False, attention=False) -> None:
    """Checkpoint selected sub-blocks inside every GridNetBlock.

    Whole-block checkpointing recomputes a full block; these flags allow
    recomputing only the intra RNN, inter RNN, or attention section. This is
    the experimental axis for a selective, sub-block policy.
    """
    import types

    if not (intra or inter or attention):
        return
    for block in network.blocks:
        block._subblock_flags = (bool(intra), bool(inter), bool(attention))
        block.forward = types.MethodType(_subblock_forward, block)
