"""Independently validate a recorded synthetic or real GTCRN/LiSenNet checkpoint."""

import argparse
from pathlib import Path


def validate_fixture(directory, epoch):
    from speech_denoising.training.fixture import fixture_audio, fixture_context, fixture_spec
    from speech_denoising.training.run_store import (audit_run, checkpoint_path, persist_validation,
                                                    read_json, run_path)
    from speech_denoising.training.state import load_training_state
    from speech_denoising.training.validation import run_validation
    run = run_path(directory)
    stored = read_json(run / "run.json")
    spec = fixture_spec(stored["identity"]["seed"])
    records, _ = audit_run(run, spec)
    if type(epoch) is not int or not 1 <= epoch <= len(records):
        raise ValueError("Validate an existing complete training epoch")
    model, optimizer = fixture_context(spec)
    progress = load_training_state(checkpoint_path(run, epoch), model, optimizer, spec["identity"])
    if progress["completed_epoch"] != epoch or progress["global_step"] != records[epoch - 1]["global_step"] or (
            progress["checkpoint_sha256"] != records[epoch - 1]["checkpoint_sha256"]):
        raise ValueError("Checkpoint progress/hash differs from the recorded epoch")
    report = run_validation(model, fixture_audio(spec, "validation"), spec["identity"], spec["validation_ids"],
                            epoch=epoch, checkpoint_sha256=progress["checkpoint_sha256"], synthetic=True)
    selection = persist_validation(run, spec, report)
    return report, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true", help="Synthetic fixture scope only")
    mode.add_argument("--real", action="store_true", help="Verified VoiceBank GTCRN/LiSenNet validation")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    try:
        if args.fixture:
            if any(value is not None for value in (args.manifest, args.preflight, args.device)):
                parser.error("--fixture does not accept real-data inputs")
            report, selection = validate_fixture(args.run_dir, args.epoch)
            print(f"Saved synthetic validation epoch {report['epoch']}; complete utterances: {report['count']}")
            print(f"Selected fixture epoch: {selection['best']['epoch']}. No VoiceBank quality scores.")
            return
        if any(value is None for value in (args.manifest, args.preflight, args.device)):
            parser.error("--real requires --manifest, --preflight and --device")
        from speech_denoising.training.real_runner import validate_real_epoch
        report, selection = validate_real_epoch(
            args.manifest, args.preflight, args.run_dir, args.epoch, device=args.device)
        print(f"Saved real validation epoch {report['epoch']}; complete utterances: {report['count']}")
        print(f"Selected epoch: {selection['best']['epoch']} by held-out train loss.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
