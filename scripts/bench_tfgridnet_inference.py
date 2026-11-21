"""Benchmark the pretrained TF-GridNet DNS model on a real waveform.

Offline whole-utterance inference, CPU FP32 or CUDA.  Reports p50/p95 latency,
RTF, peak CUDA memory, and PESQ/STOI against the clean reference.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

from speech_denoising.models.tfgridnet.checkpoint import load_tfgridnet_dns_checkpoint
from speech_denoising.models.tfgridnet.profiles import build_tfgridnet_dns_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tfgridnet_dns_epoch33.pth")
    p.add_argument("--audio", default="data/voicebank-demand-16k/audio/test/noisy/p232_023.wav")
    p.add_argument("--clean", default="data/voicebank-demand-16k/audio/test/clean/p232_023.wav")
    p.add_argument("--output", default="results/deploy/tfgridnet_p232_023_enhanced.wav")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seconds", type=float, default=0.0, help="0 keeps the whole utterance")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    audio, sr = sf.read(args.audio, dtype="float32")
    clean, csr = sf.read(args.clean, dtype="float32")
    if audio.ndim != 1 or clean.ndim != 1 or sr != 16000 or csr != 16000:
        raise SystemExit("expected mono 16 kHz noisy/clean pair")
    if args.seconds > 0:
        n = int(sr * args.seconds)
        audio, clean = audio[:n], clean[:n]

    model = build_tfgridnet_dns_model().eval()
    meta = load_tfgridnet_dns_checkpoint(model, Path(args.checkpoint))
    model = model.to(args.device)
    x = torch.from_numpy(audio).unsqueeze(0).to(args.device)

    with torch.inference_mode():
        model(x)  # warm-up
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = model(x)
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    dur = len(audio) / sr
    p50 = statistics.median(times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    print(f"profile={meta['profile']} params={meta['parameter_count']} device={args.device} threads={args.threads}")
    print(f"audio={args.audio} seconds={dur:.3f}")
    print(f"latency_ms p50={p50*1e3:.2f} p95={p95*1e3:.2f} min={times[0]*1e3:.2f} max={times[-1]*1e3:.2f}")
    print(f"rtf_p50={p50/dur:.4f} realtime_x={dur/p50:.2f}")
    if args.device == "cuda":
        print(f"cuda_peak_mib={torch.cuda.max_memory_allocated()/2**20:.1f}")

    enhanced = y[0, :len(audio)].detach().float().cpu().numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, enhanced, sr)
    print(f"PESQ_noisy={pesq(sr, clean, audio, 'wb'):.4f} PESQ_enh={pesq(sr, clean, enhanced, 'wb'):.4f}")
    print(f"STOI_noisy={stoi(clean, audio, sr, extended=False):.4f} STOI_enh={stoi(clean, enhanced, sr, extended=False):.4f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
