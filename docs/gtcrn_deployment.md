# GTCRN ONNX deployment check

First on-device check for the GTCRN paper-track checkpoint. The model is
exported as a spectrum-domain ONNX graph and executed by ONNX Runtime on the
host CPU; the sqrt-Hann STFT/ISTFT stay around the graph because ONNX Runtime
has no portable complex-STFT operator.

## Artifacts

| Item | Value |
|---|---|
| Checkpoint | `results/paper_runs/gtcrn_paper_seed42_100ep/checkpoints/epoch_0100.pt` |
| Checkpoint SHA-256 | `ba09a13bb8176ba4adf3743209ac98b886192eca1921adeaa2df6b24ee1b38e5` |
| ONNX graph | `exports/gtcrn_paper_seed42_100ep_spec.onnx` |
| ONNX bytes | 407,966 |
| ONNX SHA-256 | `d7d691b1c8c129ee156a72400f5226bd9bd92af936c846a178825f0d4a0e88d0` |
| Graph interface | `spec [B, 257, T, 2] -> enhanced_spec [B, 257, T, 2]`, dynamic `B`/`T` |
| PyTorch/ONNX parity on a 4 s input | max abs **3.64e-6** |

Export:

```sh
python -m scripts.export_gtcrn_onnx
```

## Host benchmark

Host: 12th Gen Intel Core i5-12600KF (10 cores / 16 threads), Windows,
ONNX Runtime 1.32.0, CPUExecutionProvider. Latency is the ONNX graph only;
STFT/ISTFT are outside the timed region but are negligible relative to the
network at these lengths. RTF = graph latency / audio duration.

| Input | Threads | p50 latency | p95 latency | RTF | Real-time factor |
|---|---:|---:|---:|---:|---:|
| 1.74 s test utterance | 4 | 14.49 ms | 16.76 ms | 0.0083 | 120x |
| 9.77 s test utterance | 4 | 71.32 ms | 83.52 ms | 0.0073 | 137x |
| 9.77 s test utterance | 1 | 97.34 ms | 107.95 ms | 0.0100 | 100x |
| 4 s block | 4 | 30.44 ms | — | 0.0076 | 132x |
| 4 s block | 1 | 40.21 ms | — | 0.0101 | 99x |

Single-utterance sanity check (`p232_023`, 9.77 s): noisy PESQ 1.3079 /
STOI 0.9373; ONNX-enhanced PESQ **2.2028** / STOI **0.9633**.

Run:

```sh
python -m scripts.deploy_gtcrn_onnx \
  --audio data/voicebank-demand-16k/audio/test/noisy/p232_023.wav \
  --output results/deploy/gtcrn_p232_023_enhanced.wav
```

Single-thread mode is the conservative edge proxy:

```sh
python -m scripts.deploy_gtcrn_onnx --threads 1 --repeats 20 \
  --audio data/voicebank-demand-16k/audio/test/noisy/p232_023.wav \
  --output results/deploy/gtcrn_p232_023_enhanced_1t.wav
```

## What this does and does not establish

- Establishes: the 100-epoch GTCRN checkpoint exports cleanly, runs through
  ONNX Runtime on a real CPU, is bit-close to PyTorch, and has large real-time
  headroom even on one thread.
- Does not establish phone/SoC performance, INT8 quality, or streaming latency.
  The graph is whole-utterance/spectrum-domain; live capture still needs a
  causal streaming wrapper and an overlap-add policy.
- No phone or development board was connected (`adb` absent; only a COM1 port
  and the Arctis Nova 3 headset/mic are present). The next physical target is an
  Android phone with ONNX Runtime Mobile, or a Raspberry Pi with onnxruntime.

## Next steps

1. Build an ONNX Runtime Mobile APK and run the same graph on a phone; record
   p50/p95, RTF, thermal throttling and memory.
2. Add dynamic INT8 quantization and compare PESQ/STOI on the fixed 824 test.
3. Add a streaming wrapper (causal STFT state + overlapping windows) before
   claiming live-conversation latency.
