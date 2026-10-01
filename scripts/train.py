"""Separate train entry point; Phase 1 does not execute this pipeline."""

from speech_denoising.utils.utils import phase1_entrypoint


def main():
    phase1_entrypoint("train")


if __name__ == "__main__":
    main()
