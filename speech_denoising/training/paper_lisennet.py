"""Paper-faithful LiSenNet training loop (VoiceBank+DEMAND, 16 kHz).

The repository's unified benchmark runner deliberately uses one optimizer and
one negative-SI-SNR loss for every model.  This module is a separate, explicit
LiSenNet paper track: AdamW 5e-4 / betas (0.8, 0.99), StepLR gamma 0.98,
2 s crops, batch 4, gradient clipping 5.0, and the official
``0.9*L_mag + 0.1*L_comp + 0.05*L_adv`` generator objective with a
PESQ-conditioned discriminator.

It writes to ``results/paper_runs/<name>`` and never touches the immutable
``training_reports`` artifacts of the unified benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from pesq import pesq as pesq_score
from joblib import Parallel, delayed

from speech_denoising.datasets.voicebank import VoiceBankDataset
from speech_denoising.metrics.audio_metrics import compute_audio_metrics, evaluate
from speech_denoising.models.lisennet.discriminator import Discriminator
from speech_denoising.models.lisennet.model import LiSenNetModel
from speech_denoising.training.data import load_training_inputs, subset_indices
from speech_denoising.utils.utils import REPO_ROOT

SAMPLE_RATE = 16000
CUT_LEN = 32000
BATCH_SIZE = 4
EPOCHS = 100
LR = 5.0e-4
BETAS = (0.8, 0.99)
GAMMA = 0.98
CLIP = 5.0
SEED = 42

G_LOSS_WEIGHTS = {"complex": 0.1, "magnitude": 0.9, "adversarial": 0.05}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    head = REPO_ROOT / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = REPO_ROOT / ".git" / value[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
    return value


def git_dirty():
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def random_crop_pair(noisy, clean, cut_len, rng):
    if noisy.shape != clean.shape or noisy.ndim != 1:
        raise ValueError("Noisy and clean must be aligned mono signals")
    # Official data_module.py crops first, then divides both signals by the
    # cropped noisy standard deviation.
    length = noisy.shape[0]
    if length < cut_len:
        pad = cut_len - length
        noisy = np.pad(noisy, (0, pad), mode="wrap")
        clean = np.pad(clean, (0, pad), mode="wrap")
    else:
        start = rng.randrange(0, length - cut_len + 1)
        noisy = noisy[start:start + cut_len]
        clean = clean[start:start + cut_len]
    scale = float(np.std(noisy) + 1e-8)
    return noisy / scale, clean / scale


def make_batch(dataset, indices, cut_len, rng, device):
    noisy_batch, clean_batch = [], []
    for index in indices:
        sample = dataset[index]
        noisy = sample["noisy_audio"].numpy()
        clean = sample["clean_audio"].numpy()
        noisy, clean = random_crop_pair(noisy, clean, cut_len, rng)
        noisy_batch.append(torch.from_numpy(noisy.astype(np.float32, copy=False)))
        clean_batch.append(torch.from_numpy(clean.astype(np.float32, copy=False)))
    return (torch.stack(noisy_batch).to(device=device, dtype=torch.float32),
            torch.stack(clean_batch).to(device=device, dtype=torch.float32))


def _scaled_pesq(clean, enhanced):
    try:
        value = pesq_score(SAMPLE_RATE, clean, enhanced, "wb")
    except Exception:
        return None
    if not math.isfinite(float(value)):
        return None
    return min(max((float(value) - 1.0) / 3.5, 0.0), 1.0)


def batch_pesq(clean, enhanced):
    """Return PESQ rescaled to (0, 1), or None if any pair is invalid."""
    clean_np = clean.detach().cpu().numpy()
    enhanced_np = enhanced.detach().cpu().numpy()
    scores = Parallel(n_jobs=min(4, clean_np.shape[0]), prefer="processes")(
        delayed(_scaled_pesq)(clean_np[index], enhanced_np[index])
        for index in range(clean_np.shape[0]))
    if any(score is None for score in scores):
        return None
    return [float(score) for score in scores]


def generator_loss(discriminator, results):
    est_spec, tgt_spec = results["est_spec"], results["tgt_spec"]
    complex_loss = (F.mse_loss(est_spec.real, tgt_spec.real) +
                    F.mse_loss(est_spec.imag, tgt_spec.imag))
    magnitude_loss = F.mse_loss(results["est_mag"], results["tgt_mag"])
    fake = discriminator(results["tgt_mag"], results["est_mag"])
    adversarial = F.mse_loss(fake[-1], torch.ones_like(fake[-1]))
    total = (G_LOSS_WEIGHTS["complex"] * complex_loss +
             G_LOSS_WEIGHTS["magnitude"] * magnitude_loss +
             G_LOSS_WEIGHTS["adversarial"] * adversarial)
    return total, {
        "generator_loss": float(total.detach()),
        "complex_loss": float(complex_loss.detach()),
        "magnitude_loss": float(magnitude_loss.detach()),
        "adversarial_loss": float(adversarial.detach()),
    }


def discriminator_loss(discriminator, results, pesq_targets):
    fake = discriminator(results["tgt_mag"], results["est_mag"].detach())
    real = discriminator(results["tgt_mag"], results["tgt_mag"])
    loss = F.mse_loss(real[-1], torch.ones_like(real[-1]))
    if pesq_targets is not None:
        targets = torch.as_tensor(pesq_targets, dtype=real[-1].dtype,
                                  device=real[-1].device).reshape(-1, 1)
        loss = loss + F.mse_loss(fake[-1].reshape(-1, 1), targets)
    return loss, {"discriminator_loss": float(loss.detach())}


def _audio_metrics_arrays(clean, enhanced, noisy):
    return compute_audio_metrics(clean, enhanced, noisy)["values"]


def select_validation_indices(indices, limit=None):
    """Return a deterministic, ID-ranked subset spread over the whole split."""
    indices = list(indices)
    if limit is None or limit >= len(indices):
        return indices
    if limit < 1:
        raise ValueError("limit must be positive")
    ranked = sorted(
        indices,
        key=lambda index: hashlib.sha256(
            f"lisennet_validation_v2:{index}".encode("ascii")
        ).hexdigest(),
    )
    return ranked[:limit]


@torch.inference_mode()
def validate(model, dataset, indices, device, limit=None, chunk_size=32):
    model.eval()
    totals = {name: [] for name in ("pesq", "stoi", "si_snr", "si_snr_improvement")}
    selected = select_validation_indices(indices, limit)
    for start in range(0, len(selected), chunk_size):
        chunk = selected[start:start + chunk_size]
        arrays = []
        for index in chunk:
            sample = dataset[index]
            noisy = sample["noisy_audio"].unsqueeze(0).to(device=device,
                                                        dtype=torch.float32)
            enhanced = model(noisy)
            arrays.append((sample["clean_audio"].numpy(),
                           enhanced[0].detach().cpu().numpy(),
                           sample["noisy_audio"].numpy()))
        results = Parallel(n_jobs=min(4, len(arrays)), prefer="processes")(
            delayed(_audio_metrics_arrays)(*item) for item in arrays)
        for values in results:
            for name, value in values.items():
                if value is not None:
                    totals[name].append(float(value))
    model.train()
    summary = {name: (float(np.mean(values)) if values else None)
               for name, values in totals.items()}
    summary["count"] = len(selected)
    summary["metric_coverage"] = {name: len(values) for name, values in totals.items()}
    return summary


def capture_rng_state(device):
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        # Store the NumPy state as built-ins so the checkpoint stays loadable
        # with torch.load(weights_only=True); an ndarray is not allowed there.
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "pos": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, dict):
        numpy_state = (
            numpy_state["name"], np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["pos"], numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(model, discriminator, g_optimizer, d_optimizer,
                       g_scheduler, d_scheduler, epoch, global_step, seed, recipe,
                       rng_state=None):
    return {
        "schema_version": 1,
        "kind": "lisennet_paper_checkpoint",
        "epoch": epoch,
        "global_step": global_step,
        "seed": seed,
        "recipe": recipe,
        "model": model.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "g_scheduler": g_scheduler.state_dict(),
        "d_scheduler": d_scheduler.state_dict(),
        "rng_state": rng_state,
    }


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, discriminator, g_optimizer, d_optimizer,
                    g_scheduler, d_scheduler):
    # Local, trusted run artifact; the payload contains NumPy/Python RNG state.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "lisennet_paper_checkpoint":
        raise ValueError("Not a LiSenNet paper checkpoint")
    model.load_state_dict(payload["model"])
    discriminator.load_state_dict(payload["discriminator"])
    g_optimizer.load_state_dict(payload["g_optimizer"])
    d_optimizer.load_state_dict(payload["d_optimizer"])
    g_scheduler.load_state_dict(payload["g_scheduler"])
    d_scheduler.load_state_dict(payload["d_scheduler"])
    return payload


def environment_record(device):
    record = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        record.update(gpu=properties.name,
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      gpu_total_bytes=properties.total_memory,
                      cuda_build=torch.version.cuda)
    return record


def run(args):
    manifest_path = Path(args.manifest).resolve(strict=True)
    path, manifest_bytes, manifest, plan = load_training_inputs(manifest_path, "lisennet")
    if path != manifest_path:
        raise ValueError("Manifest path resolution mismatch")
    train_indices = subset_indices(manifest, plan, "train")
    validation_indices = subset_indices(manifest, plan, "validation")
    validation_sample = select_validation_indices(validation_indices, args.validation_limit)
    validation_sample_sha256 = hashlib.sha256(
        ",".join(str(index) for index in validation_sample).encode("ascii")
    ).hexdigest()

    run_dir = Path(args.run_dir).resolve()
    results_root = (REPO_ROOT / "results/paper_runs").resolve()
    if not run_dir.is_relative_to(results_root) or run_dir == results_root:
        raise ValueError("Use a new run directory below results/paper_runs/")
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = VoiceBankDataset(path, split="train", verify_files=False,
                                     require_complete=True)
    validation_dataset = VoiceBankDataset(path, split="train", verify_files=False,
                                          require_complete=True)
    if args.verify_audio:
        print("Verifying training audio hashes once ...", flush=True)
        for index in range(len(train_dataset)):
            train_dataset[index]
        print("Audio hashes verified.", flush=True)

    model = LiSenNetModel().to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=0.0)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=LR, betas=BETAS, weight_decay=0.0)
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=GAMMA)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=GAMMA)

    recipe = {
        "name": "lisennet_paper",
        "paper": "LiSenNet, arXiv:2409.13285",
        "dataset": "VoiceBank+DEMAND 16 kHz",
        "sample_rate": SAMPLE_RATE,
        "cut_len": CUT_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "betas": list(BETAS),
        "weight_decay": 0.0,
        "scheduler": {"type": "StepLR", "step_size": 1, "gamma": GAMMA},
        "gradient_clip_norm": CLIP,
        "loss": G_LOSS_WEIGHTS,
        "discriminator": "PESQ-conditioned, ndf=16",
        "seed": args.seed,
        "split": "repository speaker_sha256_v1 held-out split",
    }
    previous_started_at = None
    if args.resume and (run_dir / "run.json").is_file():
        previous_started_at = json.loads(
            (run_dir / "run.json").read_text(encoding="utf-8")
        ).get("started_at")
    run_spec = {
        "schema_version": 1,
        "scope": "lisennet_paper_training",
        "status": "running",
        "recipe": recipe,
        "manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "split_counts": {"train": len(train_indices), "validation": len(validation_indices)},
        "split_sha256": plan["split_sha256"],
        "git": {"commit": git_commit(), "dirty": git_dirty()},
        "environment": environment_record(device),
        "command": sys.argv,
        "started_at": previous_started_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "validation_sample": {
            "count": len(validation_sample),
            "sha256": validation_sample_sha256,
            "selection": "sha256_ranked_v2",
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_spec, indent=2) + "\n", encoding="utf-8")
    (run_dir / "validation_sample.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection": "sha256_ranked_v2",
                "count": len(validation_sample),
                "sha256": validation_sample_sha256,
                "indices": validation_sample,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoint to resume from")
        payload = load_checkpoint(checkpoints[-1], model, discriminator,
                                  g_optimizer, d_optimizer, g_scheduler, d_scheduler)
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        restore_rng_state(payload.get("rng_state"))
        print(f"Resumed from {checkpoints[-1].name}: next epoch {start_epoch}", flush=True)

    curve_path = run_dir / "learning_curve.csv"
    curve_exists = curve_path.exists()
    curve = curve_path.open("a" if curve_exists else "w", newline="", encoding="utf-8")
    writer = csv.writer(curve)
    if not curve_exists:
        writer.writerow(["epoch", "global_step", "train_generator_loss", "train_complex_loss",
                         "train_magnitude_loss", "train_adversarial_loss",
                         "train_discriminator_loss", "val_pesq", "val_stoi",
                         "val_si_snr", "val_si_snr_improvement", "g_lr", "d_lr",
                         "epoch_seconds"])

    best = None
    stopped_early = False
    selection_path = run_dir / "selection.json"
    if selection_path.exists():
        best = json.loads(selection_path.read_text(encoding="utf-8")).get("best")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        rng = random.Random(f"lisennet_paper:{args.seed}:{epoch}")
        order = list(train_indices)
        rng.shuffle(order)
        totals = {name: 0.0 for name in (
            "generator_loss", "complex_loss", "magnitude_loss",
            "adversarial_loss", "discriminator_loss")}
        steps = 0
        model.train()
        discriminator.train()
        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]
            noisy, clean = make_batch(train_dataset, batch_indices, CUT_LEN, rng, device)

            results = model.network(noisy, clean)

            d_optimizer.zero_grad(set_to_none=True)
            pesq_targets = batch_pesq(clean, results["est"])
            d_loss, d_metrics = discriminator_loss(discriminator, results, pesq_targets)
            if not torch.isfinite(d_loss):
                raise ValueError("Non-finite discriminator loss")
            d_loss.backward()
            clip_grad_norm_(discriminator.parameters(), CLIP)
            d_optimizer.step()

            g_optimizer.zero_grad(set_to_none=True)
            g_loss, g_metrics = generator_loss(discriminator, results)
            if not torch.isfinite(g_loss):
                raise ValueError("Non-finite generator loss")
            g_loss.backward()
            clip_grad_norm_(model.parameters(), CLIP)
            g_optimizer.step()

            global_step += 1
            steps += 1
            for name in totals:
                totals[name] += (g_metrics.get(name) or d_metrics.get(name) or 0.0)
            if steps % args.log_every == 0:
                print(f"epoch {epoch:03d} step {steps:04d}/{(len(order) + BATCH_SIZE - 1)//BATCH_SIZE} "
                      f"g={g_metrics['generator_loss']:.5f} "
                      f"d={d_metrics['discriminator_loss']:.5f} "
                      f"lr={g_optimizer.param_groups[0]['lr']:.2e}", flush=True)
            del results, noisy, clean, d_loss, g_loss
            if args.max_steps and global_step >= args.max_steps:
                stopped_early = True
                break

        used_g_lr = g_optimizer.param_groups[0]["lr"]
        used_d_lr = d_optimizer.param_groups[0]["lr"]
        g_scheduler.step()
        d_scheduler.step()
        val = validate(model, validation_dataset, validation_indices, device,
                       limit=args.validation_limit)
        elapsed = time.perf_counter() - epoch_start
        row = [epoch, global_step,
               totals["generator_loss"] / max(steps, 1),
               totals["complex_loss"] / max(steps, 1),
               totals["magnitude_loss"] / max(steps, 1),
               totals["adversarial_loss"] / max(steps, 1),
               totals["discriminator_loss"] / max(steps, 1),
               val["pesq"], val["stoi"], val["si_snr"], val["si_snr_improvement"],
               used_g_lr, used_d_lr,
               elapsed]
        writer.writerow(row)
        curve.flush()
        print(f"epoch {epoch:03d} done in {elapsed:.1f}s: val_pesq={val['pesq']} "
              f"val_stoi={val['stoi']} steps={steps}", flush=True)

        payload = checkpoint_payload(model, discriminator, g_optimizer, d_optimizer,
                                     g_scheduler, d_scheduler, epoch, global_step,
                                     args.seed, recipe, capture_rng_state(device))
        checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        save_checkpoint(checkpoint, payload)
        digest = sha256_file(checkpoint)
        validation_record = {
            "schema_version": 1, "scope": "lisennet_paper_validation",
            "epoch": epoch, "checkpoint_sha256": digest,
            "count": val["count"], "metrics": val,
            "validation_sample_sha256": validation_sample_sha256,
            "train_steps": steps, "global_step": global_step,
        }
        (run_dir / "validation").mkdir(exist_ok=True)
        (run_dir / "validation" / f"epoch_{epoch:04d}.json").write_text(
            json.dumps(validation_record, indent=2) + "\n", encoding="utf-8")

        if val["pesq"] is not None and (best is None or val["pesq"] > best["pesq"]):
            best = {"epoch": epoch, "pesq": val["pesq"], "stoi": val["stoi"],
                    "si_snr": val["si_snr"], "checkpoint": checkpoint.relative_to(run_dir).as_posix(),
                    "checkpoint_sha256": digest}
            (run_dir / "selection.json").write_text(
                json.dumps({"schema_version": 1, "best": best}, indent=2) + "\n",
                encoding="utf-8")

        if args.max_steps and global_step >= args.max_steps:
            stopped_early = True
            print("Stopping after requested max steps", flush=True)
            break

    curve.close()
    run_spec["status"] = "stopped_early" if stopped_early else "complete"
    run_spec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_spec["best"] = best
    (run_dir / "run.json").write_text(json.dumps(run_spec, indent=2) + "\n", encoding="utf-8")

    if args.test_manifest and best and not stopped_early:
        test_path = Path(args.test_manifest).resolve(strict=True)
        test_dataset = VoiceBankDataset(test_path, split="test", verify_files=True,
                                        require_complete=True)
        checkpoint = run_dir / best["checkpoint"]
        load_checkpoint(checkpoint, model, discriminator, g_optimizer, d_optimizer,
                        g_scheduler, d_scheduler)
        model.eval()
        print(f"Evaluating epoch {best['epoch']} on {len(test_dataset)} test utterances ...",
              flush=True)
        report = evaluate(model, test_dataset, metrics=("pesq", "stoi", "si_snr",
                                                        "si_snr_improvement"))
        report.update({
            "schema_version": 1,
            "scope": "lisennet_paper_test",
            "selected_epoch": best["epoch"],
            "checkpoint_sha256": best["checkpoint_sha256"],
            "recipe": recipe,
            "run_dir": str(run_dir),
            "test_manifest": str(test_path),
        })
        (run_dir / "test824.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
        print("Test PESQ:", report["summary"]["enhanced"]["pesq"], flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Complete prepared train_manifest.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-audio", action="store_true",
                        help="Hash every training WAV once before training")
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Smoke-test stop after this many optimizer steps")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--test-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.log_every < 1:
        parser.error("epochs and log-every must be positive")
    if args.validation_limit is not None and args.validation_limit < 1:
        parser.error("validation-limit must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
