"""Evaluate audited pretrained profiles with the common waveform/metric runner."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from speech_denoising.datasets import VoiceBankDataset
from speech_denoising.models import build_model
from speech_denoising.models.gtcrn.checkpoint import load_gtcrn_checkpoint
from speech_denoising.metrics.audio_metrics import QUALITY_METRICS, evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
DNS_PROFILE = "dns_ins20_epoch33"
GTCRN_PROFILE = "vctk_demand"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gtcrn", choices=["gtcrn", "lisennet", "tfgridnet"])
    parser.add_argument("--profile", choices=[GTCRN_PROFILE, DNS_PROFILE])
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/voicebank-demand-16k/test_manifest.json")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the selected pretrained profile's checkpoint")
    parser.add_argument("--output", type=Path, help="Defaults to a model-specific report under results/")
    parser.add_argument("--metrics", nargs="+", choices=QUALITY_METRICS, default=list(QUALITY_METRICS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.model == "lisennet":
        parser.error("LiSenNet has no accepted pretrained baseline; see docs/lisennet_checkpoint_audit.md")
    if args.model == "tfgridnet":
        if args.profile != DNS_PROFILE:
            parser.error("TF-GridNet evaluation requires --profile dns_ins20_epoch33")
        checkpoint_name, output_name = "tfgridnet_dns_epoch33.pth", "tfgridnet_dns_voicebank.json"
    else:
        if args.profile not in (None, GTCRN_PROFILE):
            parser.error("GTCRN only accepts --profile vctk_demand")
        args.profile = GTCRN_PROFILE
        checkpoint_name, output_name = "gtcrn_vctk.tar", "gtcrn_voicebank.json"
    if args.checkpoint is None:
        args.checkpoint = REPO_ROOT / "checkpoints" / checkpoint_name
    if args.output is None:
        args.output = REPO_ROOT / "results" / output_name
    if args.threads < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("Require threads >= 1, repeats >= 1 and warmup >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if len(set(args.metrics)) != len(args.metrics):
        parser.error("--metrics must not contain duplicates")
    if args.output.resolve() in {args.manifest.resolve(), args.checkpoint.resolve()}:
        parser.error("Output must not overwrite the manifest or checkpoint")
    return args


def load_pretrained(args):
    """Return a strictly loaded model; never evaluate a random-weight fallback."""
    if args.model == "gtcrn":
        model = build_model("gtcrn")
        metadata = load_gtcrn_checkpoint(model, args.checkpoint)
    elif args.model == "tfgridnet" and args.profile == DNS_PROFILE:
        from speech_denoising.models.tfgridnet.profiles import build_tfgridnet_dns_model
        from speech_denoising.models.tfgridnet.checkpoint import load_tfgridnet_dns_checkpoint

        model = build_tfgridnet_dns_model()
        metadata = load_tfgridnet_dns_checkpoint(model, args.checkpoint)
    else:
        raise ValueError("Unsupported pretrained model/profile")
    return model, metadata


def implementation_record(model_name):
    """Capture the Git context and the exact sources used by this evaluation."""
    names = ["scripts/evaluate.py", "speech_denoising/metrics/audio_metrics.py",
             "speech_denoising/datasets/voicebank.py", "configs/voicebank.json",
             "speech_denoising/models/base.py"]
    suffixes = ["network.py", "model.py", "checkpoint.py"]
    if model_name == "lisennet":
        suffixes = ["network.py", "model.py", "dpr_layer.py"]
    if model_name == "tfgridnet":
        suffixes += ["spectral.py", "profiles.py", "assets.py"]
    names += [f"speech_denoising/models/{model_name}/{suffix}" for suffix in suffixes]
    fingerprints = {name: hashlib.sha256((REPO_ROOT / name).read_text(encoding="utf-8").encode()).hexdigest()
                    for name in names}
    record = {"source_fingerprints": fingerprints,
              "sha256_basis": "UTF-8 text with normalized line endings",
              "git_commit": None, "git_worktree_dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        record.update(git_commit=commit.stdout.strip(), git_worktree_dirty=bool(status.stdout.strip()))
    return record


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    dataset = VoiceBankDataset(args.manifest, split="test")
    model, metadata = load_pretrained(args)
    report = evaluate(model, dataset, metrics=args.metrics, limit=args.limit, warmup=args.warmup, repeats=args.repeats)
    comparison_type = ("external_pretrained; not controlled unified training" if args.model == "tfgridnet"
                       else "upstream_pretrained; not unified training")
    report.update(model=args.model, profile=args.profile, checkpoint=metadata,
                  comparison_type=comparison_type, implementation=implementation_record(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Model/profile:", args.model, "/", args.profile)
    print("Scope:", report["scope"])
    print("Evaluated:", report["evaluated_count"], "/", report["manifest_count"])
    print("Enhanced metrics:", report["summary"]["enhanced"])
    print("Not computed:", report["not_computed"])
    print("Metric failure count:", len(report["failed_utterances"]))
    print("Parameters (all / trainable):", report["parameter_count"], report["trainable_parameter_count"])
    print("RTF:", report["rtf"])
    print("Report:", args.output)
    if report["failed_utterances"]:
        raise SystemExit("Report contains metric failures; inspect coverage before using scores")


if __name__ == "__main__":
    main()
