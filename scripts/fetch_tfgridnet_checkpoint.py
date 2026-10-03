"""Download only the pinned DNS checkpoint, with size and SHA-256 checks."""

import argparse
from pathlib import Path

from speech_denoising.models.tfgridnet.assets import DEFAULT_CHECKPOINT, fetch_checkpoint, verify_checkpoint_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    path = fetch_checkpoint(args.output)
    print("PASS: pinned TF-GridNet DNS checkpoint integrity")
    print("Checkpoint:", path)
    print("SHA-256:", verify_checkpoint_file(path))
    print("No model loading, inference, training or dataset evaluation performed.")


if __name__ == "__main__":
    main()
