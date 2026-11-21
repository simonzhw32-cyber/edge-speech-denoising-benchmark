"""Run the exported GTCRN spectrum model on real audio with ONNX Runtime.

This is the first on-device deployment check: the graph runs through
``onnxruntime`` on the CPU, while the sqrt-Hann STFT/ISTFT stay in torch.  It
reports p50/p95 latency and real-time factor (RTF) for full utterances and for
4 s blocks, and writes the enhanced wav next to the input.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch


def stft(audio: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(audio).float()
    window = torch.hann_window(512).sqrt()
    spec = torch.stft(
        x, n_fft=512, hop_length=256, win_length=512, window=window,
        center=True, pad_mode="reflect", normalized=False, onesided=True,
        return_complex=True,
    )
    return torch.view_as_real(spec).unsqueeze(0).numpy()


def istft(spec: np.ndarray, length: int) -> np.ndarray:
    s = torch.from_numpy(spec[0])
    window = torch.hann_window(512).sqrt()
    out = torch.istft(
        torch.complex(s[..., 0], s[..., 1]),
        n_fft=512, hop_length=256, win_length=512, window=window,
        center=True, normalized=False, onesided=True, length=length,
    )
    return out.numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="exports/gtcrn_paper_seed42_100ep_spec.onnx")
    p.add_argument("--audio", default="data/voicebank-demand-16k/audio/test/noisy/p232_001.wav")
    p.add_argument("--output", default="results/deploy/gtcrn_p232_001_enhanced.wav")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim != 1 or sr != 16000:
        raise SystemExit(f"expected mono 16 kHz, got shape={audio.shape} sr={sr}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.onnx, sess_options=opts, providers=["CPUExecutionProvider"])
    spec = stft(audio)
    print(f"audio={args.audio} samples={len(audio)} seconds={len(audio)/sr:.3f}")
    print(f"onnx={args.onnx} input={spec.shape} threads={args.threads}")

    # Warm-up then timed runs.
    out = sess.run(["enhanced_spec"], {"spec": spec})[0]
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        out = sess.run(["enhanced_spec"], {"spec": spec})[0]
        times.append(time.perf_counter() - t0)
    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    rtf = p50 / (len(audio) / sr)
    print(f"latency_ms p50={p50*1e3:.2f} p95={p95*1e3:.2f} min={times[0]*1e3:.2f} max={times[-1]*1e3:.2f}")
    print(f"rtf_p50={rtf:.4f} realtime_x={1.0/rtf:.1f}")

    enhanced = istft(out, len(audio))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, enhanced, sr)
    print(f"wrote={args.output}")

    # 4 s block benchmark, matching the training/eval segment.
    block = 64000
    if len(audio) >= block:
        chunks = [audio[i:i+block] for i in range(0, len(audio)-block+1, block)]
        specs = [stft(c) for c in chunks]
        for s in specs:
            sess.run(["enhanced_spec"], {"spec": s})
        ts = []
        for s in specs:
            t0 = time.perf_counter()
            sess.run(["enhanced_spec"], {"spec": s})
            ts.append(time.perf_counter() - t0)
        print(f"blocks={len(ts)} block_p50_ms={statistics.median(ts)*1e3:.2f} block_rtf={statistics.median(ts)/4.0:.4f}")


if __name__ == "__main__":
    main()
