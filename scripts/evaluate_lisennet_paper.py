"""Full validation and test evaluation for a completed LiSenNet paper run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from speech_denoising.datasets.voicebank import VoiceBankDataset
from speech_denoising.metrics.audio_metrics import evaluate
from speech_denoising.models.lisennet.discriminator import Discriminator
from speech_denoising.models.lisennet.model import LiSenNetModel
from speech_denoising.training.data import load_training_inputs, subset_indices
from speech_denoising.training.paper_lisennet import (
    BETAS, LR, load_checkpoint, sha256_file, validate,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Prepared JSON manifest used for training")
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    run_dir = args.run_dir.resolve()
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    best = selection["best"]
    checkpoint = run_dir / best["checkpoint"]
    digest = sha256_file(checkpoint)
    if digest != best["checkpoint_sha256"]:
        raise ValueError(
            f"Checkpoint hash mismatch: selection.json={best['checkpoint_sha256']} "
            f"actual={digest}"
        )

    manifest_path, _, manifest, plan = load_training_inputs(
        args.manifest.resolve(), "lisennet")
    run_spec = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    plan_digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if plan_digest != run_spec.get("plan_sha256"):
        raise ValueError("Manifest split plan does not match run.json")
    validation_indices = subset_indices(manifest, plan, "validation")
    validation_dataset = VoiceBankDataset(
        manifest_path, split="train", verify_files=False, require_complete=True)

    model = LiSenNetModel().to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=LR, betas=BETAS)
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=0.98)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=0.98)
    load_checkpoint(checkpoint, model, discriminator, g_optimizer, d_optimizer,
                    g_scheduler, d_scheduler)
    model.eval()

    print(f"Full validation of epoch {best['epoch']} "
          f"({len(validation_indices)} utterances) ...", flush=True)
    validation = validate(model, validation_dataset, validation_indices, device)
    record = {
        "schema_version": 1,
        "scope": "lisennet_paper_full_validation",
        "selected_epoch": best["epoch"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": best["checkpoint_sha256"],
        "count": validation["count"],
        "metrics": validation,
    }
    (run_dir / "validation_full.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Full validation:", validation, flush=True)

    if args.test_manifest:
        test_path = args.test_manifest.resolve()
        test_dataset = VoiceBankDataset(test_path, split="test", verify_files=True,
                                        require_complete=True)
        print(f"Test evaluation ({len(test_dataset)} utterances) ...", flush=True)
        report = evaluate(model, test_dataset, metrics=("pesq", "stoi", "si_snr",
                                                        "si_snr_improvement"),
                          limit=args.test_limit)
        report.update({
            "schema_version": 1,
            "scope": "lisennet_paper_test",
            "selected_epoch": best["epoch"],
            "checkpoint_sha256": best["checkpoint_sha256"],
            "run_dir": str(run_dir),
            "test_manifest": str(test_path),
            "test_limit": args.test_limit,
        })
        name = "test824_full.json" if args.test_limit is None else f"test_limit_{args.test_limit}.json"
        (run_dir / name).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Test PESQ:", report["summary"]["enhanced"]["pesq"], flush=True)


if __name__ == "__main__":
    main()
