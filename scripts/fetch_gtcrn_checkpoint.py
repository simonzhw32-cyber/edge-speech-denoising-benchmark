"""Re-fetch pinned official weights and one demo clip; no repository cloning."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

from speech_denoising.models.gtcrn.checkpoint import (
    CHECKPOINT_URL, CHECKPOINT_SHA256, DEMO_URL, DEMO_SHA256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def fetch(url, digest, path):
    path = Path(path)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        print("Already verified:", path)
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read(2 * 1024 * 1024 + 1)
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Downloaded content failed SHA-256 verification.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    temporary.write_bytes(content)
    temporary.replace(path)
    print("Downloaded and verified:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/gtcrn_vctk.tar")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    fetch(CHECKPOINT_URL, CHECKPOINT_SHA256, args.checkpoint)
    if args.demo:
        fetch(DEMO_URL, DEMO_SHA256, REPO_ROOT / "results/gtcrn_demo/noisy.wav")


if __name__ == "__main__":
    main()
