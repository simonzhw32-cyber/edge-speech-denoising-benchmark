"""Run one reproducible TF-GridNet compute-policy quality proxy."""

import argparse
from pathlib import Path

from speech_denoising.training.tfgridnet_policy import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/voicebank-demand-16k/train_manifest.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, choices=(128, 256), default=128)
    parser.add_argument("--amp", choices=("none", "fp16"), default="none")
    parser.add_argument("--checkpoint-blocks", default="0",
                        help="integer count of first blocks, or comma-separated indices")
    parser.add_argument("--subblock-intra", action="store_true")
    parser.add_argument("--subblock-inter", action="store_true")
    parser.add_argument("--subblock-attention", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-pool", type=int, default=400)
    parser.add_argument("--validation-size", type=int, default=80)
    parser.add_argument("--validation-limit", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--skip-audio-verify", action="store_true",
                        help="proxy runs may skip per-file rehashing after a verified download")
    args = parser.parse_args()
    raw = str(args.checkpoint_blocks).strip()
    args.checkpoint_blocks = (
        [int(part) for part in raw.split(",") if part.strip()]
        if "," in raw else int(raw)
    )
    args.segment_samples = int(round(args.segment_seconds * 16000))
    args.segment_seconds = args.segment_samples / 16000.0
    train(args)


if __name__ == "__main__":
    main()
