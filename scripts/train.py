"""Describe the training draft or execute one explicitly synthetic fixture epoch."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--fixture", action="store_true", help="CPU random-GTCRN synthetic updates only")
    parser.add_argument("--model", choices=("gtcrn", "lisennet", "tfgridnet"), default="gtcrn")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.describe:
            if args.run_dir is not None or args.epoch is not None:
                parser.error("--describe does not create or update a run")
            from scripts.plan_training import model_settings
            from speech_denoising.training.protocol import validate_protocol
            from speech_denoising.utils.utils import load_config
            protocol = load_config("configs/training_protocol.yaml")
            validate_protocol(protocol)
            print(json.dumps({"status": "draft", "execution_ready": False, "model": args.model,
                              "model_settings": model_settings(protocol, args.model),
                              "execution_scope": "synthetic_fixture_only"}, indent=2, allow_nan=False))
            return
        if args.model != "gtcrn" or args.run_dir is None or args.epoch is None:
            parser.error("--fixture requires --run-dir and --epoch; only GTCRN is used by this fixture")
        from speech_denoising.training.runner import train_fixture_epoch
        result = train_fixture_epoch(args.run_dir, args.epoch, args.seed)
        print(f"Saved synthetic GTCRN epoch {result['epoch']}; optimizer steps: {result['global_step']}")
        print("Run scripts.validate separately before the next epoch. No real-data training or quality scores.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
