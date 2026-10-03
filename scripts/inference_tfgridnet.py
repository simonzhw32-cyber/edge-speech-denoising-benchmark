"""Single-WAV DNS pretrained inference through BaseEnhancementModel.forward."""

import argparse
import json
from pathlib import Path
import platform
import time

import torch

from scripts.inference import read_audio, write_audio
from speech_denoising.models.tfgridnet.assets import DEFAULT_CHECKPOINT, file_sha256
from speech_denoising.models.tfgridnet.checkpoint import load_tfgridnet_dns_checkpoint
from speech_denoising.models.tfgridnet.profiles import build_tfgridnet_dns_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, required=True, help="Mono 16k PCM16 WAV")
    parser.add_argument("--output", type=Path, required=True, help="Enhanced .wav path")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".json")
    if args.threads < 1 or args.output.suffix.lower() != ".wav":
        parser.error("Use positive --threads and a .wav output")
    protected = {args.input.resolve(), args.checkpoint.resolve()}
    if args.output.resolve() in protected or metadata_path.resolve() in protected:
        parser.error("Output paths must differ from input and checkpoint")
    torch.set_num_threads(args.threads)
    model = build_tfgridnet_dns_model().eval()
    metadata = load_tfgridnet_dns_checkpoint(model, args.checkpoint)
    noisy = read_audio(args.input)
    with torch.inference_mode():
        started = time.perf_counter()
        enhanced = model(noisy)
        elapsed = time.perf_counter() - started
    if enhanced.shape != noisy.shape or not torch.isfinite(enhanced).all():
        raise RuntimeError("Invalid enhanced waveform")
    clipped = write_audio(args.output, enhanced)
    metadata.update({
        "scope": "single_wav_inference", "model": "tfgridnet",
        "input_sha256": file_sha256(args.input),
        "samples": noisy.shape[-1], "duration_seconds": noisy.shape[-1] / 16000,
        "clipped_output_samples": clipped,
        "torch_version": torch.__version__, "platform": platform.platform(),
        "python_version": platform.python_version(),
        "device": "cpu", "precision": "float32", "batch_size": 1,
        "threads": args.threads, "inference_seconds": elapsed,
        "timing_note": "One cold run; not benchmark RTF.",
        "quality_scores_generated": False, "training_performed": False,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PASS: pretrained TF-GridNet DNS single-WAV inference")
    print("Matched checkpoint entries:", metadata["matched_state_entries"])
    print("Input/output samples:", noisy.shape[-1], enhanced.shape[-1])
    print("Enhanced WAV:", args.output)
    print("Metadata:", metadata_path)
    print("No training or dataset quality evaluation performed.")


if __name__ == "__main__":
    main()
