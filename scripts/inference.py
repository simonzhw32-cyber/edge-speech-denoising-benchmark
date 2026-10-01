"""Single-file pretrained inference through the shared waveform model interface."""

import argparse
import json
from pathlib import Path
import platform
import time
import wave

import numpy as np
import torch

from speech_denoising.models import build_model
from speech_denoising.models.gtcrn.checkpoint import file_sha256, load_gtcrn_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_audio(path):
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("This first inference entry point requires mono 16 kHz PCM16 WAV.")
        frames = handle.getnframes()
        samples = np.frombuffer(handle.readframes(frames), dtype="<i2").astype(np.float32) / 32768.0
    if frames == 0 or samples.size != frames:
        raise ValueError("WAV is empty or truncated.")
    return torch.from_numpy(samples.copy()).unsqueeze(0)


def write_audio(path, audio):
    samples = audio.squeeze(0).detach().cpu().numpy()
    clipped = int(np.count_nonzero((samples < -1.0) | (samples > 32767 / 32768)))
    pcm = np.rint(np.clip(samples, -1.0, 32767 / 32768) * 32768).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm.tobytes())
    return clipped


def main():
    parser = argparse.ArgumentParser(description="Pretrained GTCRN single-WAV inference")
    parser.add_argument("--model", default="gtcrn", choices=["gtcrn"])
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/gtcrn_vctk.tar")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results/gtcrn_demo/noisy.wav")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/gtcrn_demo/enhanced.wav")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    metadata_path = args.output.with_suffix(".json")
    protected = {args.input.resolve(), args.checkpoint.resolve()}
    if args.output.resolve() in protected or metadata_path.resolve() in protected:
        parser.error("Output paths must differ from input and checkpoint")
    torch.set_num_threads(args.threads)
    model = build_model(args.model).eval()
    metadata = load_gtcrn_checkpoint(model, args.checkpoint)
    noisy = read_audio(args.input)
    with torch.inference_mode():
        started = time.perf_counter()
        enhanced = model(noisy)
        elapsed = time.perf_counter() - started
    if enhanced.shape != noisy.shape or not torch.isfinite(enhanced).all():
        raise RuntimeError("Model returned invalid audio")
    clipped = write_audio(args.output, enhanced)
    metadata.update({
        "model": args.model, "input_sha256": file_sha256(args.input),
        "sample_rate": 16000, "samples": noisy.shape[-1],
        "duration_seconds": noisy.shape[-1] / 16000, "clipped_output_samples": clipped,
        "torch_version": torch.__version__, "platform": platform.platform(),
        "device": "cpu", "precision": "float32", "batch_size": 1,
        "threads": args.threads, "inference_seconds": elapsed,
        "timing_note": "Single cold run; frontend included, file I/O excluded. Not benchmark RTF.",
        "evaluation_note": "Demo has no clean reference; no PESQ/STOI/SI-SNR score reported.",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("PASS: pretrained GTCRN inference")
    print("Matched checkpoint entries:", metadata["matched_state_entries"])
    print("Input/output samples:", noisy.shape[-1], enhanced.shape[-1])
    print("Enhanced WAV:", args.output)
    print("Metadata:", metadata_path)
    print("No training or dataset benchmark performed.")


if __name__ == "__main__":
    main()
