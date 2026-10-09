"""Describe training or execute one synthetic or verified VoiceBank GTCRN/LiSenNet/TF-GridNet epoch."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--fixture", action="store_true", help="CPU random-GTCRN synthetic updates only")
    mode.add_argument("--real", action="store_true", help="Verified VoiceBank GTCRN/LiSenNet/TF-GridNet training")
    parser.add_argument("--model", choices=("gtcrn", "lisennet", "tfgridnet"), default="gtcrn")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--initialization", choices=("random", "pretrained"))
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--deterministic-algorithms", action="store_true")
    args = parser.parse_args()
    try:
        if args.describe:
            if any(value is not None for value in (
                    args.run_dir, args.epoch, args.manifest, args.preflight, args.device,
                    args.initialization, args.initial_checkpoint)) or args.deterministic_algorithms:
                parser.error("--describe does not create or update a run")
            from scripts.plan_training import model_settings
            from speech_denoising.training.protocol import validate_protocol
            from speech_denoising.utils.utils import load_config
            protocol = load_config("configs/training_protocol.yaml")
            validate_protocol(protocol)
            print(json.dumps({"status": "draft", "execution_ready": False, "model": args.model,
                              "model_settings": model_settings(protocol, args.model),
                              "execution_scope": ("verified_real_training_available"
                                                  if args.model in ("gtcrn", "lisennet", "tfgridnet")
                                                  else "synthetic_fixture_only")},
                             indent=2, allow_nan=False))
            return
        if args.fixture:
            if args.model != "gtcrn" or args.run_dir is None or args.epoch is None or any(
                    value is not None for value in (
                        args.manifest, args.preflight, args.device, args.initialization,
                        args.initial_checkpoint)) or args.deterministic_algorithms:
                parser.error("--fixture accepts --run-dir, --epoch and --seed; it uses GTCRN on CPU")
            from speech_denoising.training.runner import train_fixture_epoch
            result = train_fixture_epoch(args.run_dir, args.epoch, args.seed)
            print(f"Saved synthetic GTCRN epoch {result['epoch']}; optimizer steps: {result['global_step']}")
            print("Run scripts.validate separately before the next epoch. No real-data quality scores.")
            return
        required = (args.run_dir, args.epoch, args.manifest, args.preflight,
                    args.device, args.initialization)
        if args.model not in ("gtcrn", "lisennet", "tfgridnet") or any(value is None for value in required):
            parser.error("--real requires GTCRN, LiSenNet or TF-GridNet, --manifest, --preflight, --run-dir, "
                         "--epoch, --device and --initialization")
        if args.model in ("lisennet", "tfgridnet") and args.initialization != "random":
            parser.error("LiSenNet and TF-GridNet support --initialization random only")
        if (args.initialization == "pretrained") != (args.initial_checkpoint is not None):
            parser.error("--pretrained requires --initial-checkpoint; --random forbids it")
        from speech_denoising.training.real_runner import train_real_epoch
        result = train_real_epoch(
            args.manifest, args.preflight, args.run_dir, args.epoch, seed=args.seed,
            device=args.device, initialization=args.initialization,
            initial_checkpoint=args.initial_checkpoint,
            deterministic_algorithms=args.deterministic_algorithms, model=args.model)
        print(f"Saved real {args.model} epoch {result['epoch']}; optimizer steps: {result['global_step']}")
        print("Run scripts.validate --real for this epoch before starting the next epoch.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
