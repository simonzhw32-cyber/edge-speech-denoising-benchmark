"""Verify an existing complete train manifest and audio; never fetch or train."""

import argparse
import hashlib
from pathlib import Path

from speech_denoising.training.data import load_training_inputs
from speech_denoising.training.preflight import report_path, verify_training_audio, write_preflight_report
from speech_denoising.utils.utils import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["gtcrn", "lisennet", "tfgridnet"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        # Refuse a protected/existing output before doing an expensive audio scan.
        output = report_path(args.output, protected=(args.manifest,))
        path, data, manifest, plan = load_training_inputs(args.manifest, args.model)
        paths = ["configs/voicebank.json", "configs/training_protocol.yaml",
                 plan["model_settings"]["config"], "configs/default.yaml",
                 "scripts/plan_training.py", "scripts/preflight_training.py",
                 "speech_denoising/training/protocol.py", "speech_denoising/training/data.py",
                 "speech_denoising/training/preflight.py", "speech_denoising/datasets/voicebank.py",
                 "speech_denoising/utils/utils.py"]
        def source_hashes():
            return {name: hashlib.sha256((REPO_ROOT / name).read_text(encoding="utf-8").encode()).hexdigest()
                    for name in paths}
        initial_hashes = source_hashes()
        def progress(count, total):
            if count % 500 == 0 or count == total:
                print(f"Verified {count}/{total} train-source audio pairs", flush=True)
        report = verify_training_audio(path, manifest, plan, plan["source"], plan["protocol"],
                                       plan["model_settings"], progress=progress)
        if path.read_bytes() != data:
            raise ValueError("Manifest changed since planning")
        _, current_data, _, current_plan = load_training_inputs(path, args.model)
        if current_data != data or current_plan != plan or source_hashes() != initial_hashes:
            raise ValueError("Source, configuration or input changed during verification")
        report["source_fingerprints"] = initial_hashes
        saved = write_preflight_report(output, report, protected=(path,))
        print("PASS: complete local train audio and speaker-disjoint draft plan")
        print("Counts:", plan["counts"])
        print("Validation speakers:", plan["split"]["validation_speakers"])
        print("Split SHA-256:", plan["split_sha256"])
        print("Report:", saved)
        print("Execution ready: false; device, crop and hardware feasibility still unverified.")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
