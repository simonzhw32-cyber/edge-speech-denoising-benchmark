"""Re-select a LiSenNet paper checkpoint with full validation.

The 100-epoch run selects checkpoints on a fixed 200-utterance subset so that
each epoch stays inside the interactive time budget.  That subset is a
deterministic SHA-256-ranked sample and is intentionally cheap, not a paper
protocol.  Once the
run is complete, this script re-evaluates the top-k subset candidates on all
794 held-out utterances, writes ``selection_full.json`` and leaves the original
``selection.json`` untouched.  It can optionally evaluate the chosen epoch on
the fixed 824-utterance test manifest.

Example::

    python -m scripts.reselect_lisennet_paper \
      --run-dir results/paper_runs/lisennet_paper_seed42_100ep \
      --manifest data/voicebank-demand-16k/train_manifest.json \
      --top-k 5 --device cuda \
      --test-manifest data/voicebank-demand-16k/test_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from speech_denoising.datasets.voicebank import VoiceBankDataset
from speech_denoising.metrics.audio_metrics import evaluate
from speech_denoising.models.lisennet.discriminator import Discriminator
from speech_denoising.models.lisennet.model import LiSenNetModel
from speech_denoising.training.data import load_training_inputs, subset_indices
from speech_denoising.training.paper_lisennet import (
    BETAS, GAMMA, LR, load_checkpoint, sha256_file, validate,
)


def _read_candidates(run_dir: Path, top_k: int):
    curve = run_dir / "learning_curve.csv"
    if not curve.is_file():
        raise FileNotFoundError(curve)
    rows = []
    with curve.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                epoch = int(row["epoch"])
                pesq = float(row["val_pesq"])
            except (KeyError, TypeError, ValueError):
                continue
            checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
            if checkpoint.is_file():
                rows.append((epoch, pesq, checkpoint))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:top_k]


def _build_modules(device):
    model = LiSenNetModel().to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=LR, betas=BETAS)
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=GAMMA)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=GAMMA)
    return model, discriminator, g_optimizer, d_optimizer, g_scheduler, d_scheduler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k must be positive")

    run_dir = args.run_dir.resolve()
    manifest_path, _, manifest, plan = load_training_inputs(
        args.manifest.resolve(), "lisennet")
    validation_indices = subset_indices(manifest, plan, "validation")
    validation_dataset = VoiceBankDataset(
        manifest_path, split="train", verify_files=False, require_complete=True)

    candidates = _read_candidates(run_dir, args.top_k)
    if not candidates:
        raise RuntimeError("No checkpoint with a finite validation PESQ")

    device = torch.device(args.device)
    model, discriminator, g_optimizer, d_optimizer, g_scheduler, d_scheduler = (
        _build_modules(device))

    records = []
    for epoch, subset_pesq, checkpoint in candidates:
        print(f"Full validation epoch {epoch} (subset PESQ {subset_pesq:.6f}) ...",
              flush=True)
        load_checkpoint(checkpoint, model, discriminator, g_optimizer,
                        d_optimizer, g_scheduler, d_scheduler)
        model.eval()
        metrics = validate(model, validation_dataset, validation_indices, device)
        records.append({
            "epoch": epoch,
            "subset_pesq": subset_pesq,
            "checkpoint": str(checkpoint.relative_to(run_dir)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "metrics": metrics,
        })
        print(f"  full PESQ {metrics['pesq']} STOI {metrics['stoi']}", flush=True)

    records.sort(key=lambda item: item["metrics"]["pesq"], reverse=True)
    best = records[0]
    report = {
        "schema_version": 1,
        "scope": "lisennet_paper_full_reselection",
        "candidates": records,
        "best": {
            "epoch": best["epoch"],
            "pesq": best["metrics"]["pesq"],
            "stoi": best["metrics"]["stoi"],
            "si_snr": best["metrics"]["si_snr"],
            "checkpoint": best["checkpoint"],
            "checkpoint_sha256": best["checkpoint_sha256"],
        },
        "note": (
            "Re-evaluates the top-k subset candidates on all held-out "
            "utterances; does not overwrite selection.json or run.json."
        ),
    }
    (run_dir / "validation_full_topk.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_dir / "selection_full.json").write_text(
        json.dumps({"schema_version": 1, "best": report["best"]}, indent=2) + "\n",
        encoding="utf-8")
    print("Selected epoch", best["epoch"], "full PESQ", best["metrics"]["pesq"],
          flush=True)

    if args.test_manifest:
        test_path = args.test_manifest.resolve()
        test_dataset = VoiceBankDataset(
            test_path, split="test", verify_files=True, require_complete=True)
        load_checkpoint(run_dir / best["checkpoint"], model, discriminator,
                        g_optimizer, d_optimizer, g_scheduler, d_scheduler)
        model.eval()
        print(f"Test evaluation ({len(test_dataset)} utterances) ...", flush=True)
        test_report = evaluate(
            model, test_dataset,
            metrics=("pesq", "stoi", "si_snr", "si_snr_improvement"),
            limit=args.test_limit)
        test_report.update({
            "schema_version": 1,
            "scope": "lisennet_paper_test_full_reselection",
            "selected_epoch": best["epoch"],
            "checkpoint_sha256": best["checkpoint_sha256"],
            "run_dir": str(run_dir),
            "test_manifest": str(test_path),
        })
        (run_dir / "test824_full_reselection.json").write_text(
            json.dumps(test_report, indent=2) + "\n", encoding="utf-8")
        print("Test PESQ:", test_report["summary"]["enhanced"]["pesq"], flush=True)


if __name__ == "__main__":
    main()
