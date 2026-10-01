"""Evaluate the fixed manifest through one common waveform/metric runner."""

import argparse
import json
from pathlib import Path

import torch

from speech_denoising.datasets import VoiceBankDataset
from speech_denoising.models import build_model
from speech_denoising.models.gtcrn.checkpoint import load_gtcrn_checkpoint
from speech_denoising.metrics.audio_metrics import QUALITY_METRICS, evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gtcrn", choices=["gtcrn", "lisennet", "tfgridnet"])
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/voicebank-demand-16k/test_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/gtcrn_vctk.tar")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/gtcrn_voicebank.json")
    parser.add_argument("--metrics", nargs="+", choices=QUALITY_METRICS, default=list(QUALITY_METRICS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.resolve() in {args.manifest.resolve(), args.checkpoint.resolve()}:
        parser.error("Output must not overwrite input")
    if args.model != "gtcrn":
        parser.error("Only GTCRN has a pretrained adapter yet; other models will use this runner later")
    torch.set_num_threads(args.threads)
    dataset = VoiceBankDataset(args.manifest, split="test")
    model = build_model(args.model)
    metadata = load_gtcrn_checkpoint(model, args.checkpoint)
    report = evaluate(model, dataset, metrics=args.metrics, limit=args.limit, warmup=args.warmup, repeats=args.repeats)
    report.update({"model": args.model, "checkpoint": metadata,
                   "comparison_type": "upstream_pretrained; not unified training"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Scope:", report["scope"])
    print("Evaluated:", report["evaluated_count"], "/", report["manifest_count"])
    print("Enhanced metrics:", report["summary"]["enhanced"])
    print("Not computed:", report["not_computed"])
    print("Metric failure count:", len(report["failed_utterances"]))
    print("RTF:", report["rtf"])
    print("Report:", args.output)
    if report["failed_utterances"]:
        raise SystemExit("Report contains metric failures; inspect coverage before using scores")


if __name__ == "__main__":
    main()
