# TF-GridNet deployment check

The deployment target is the **pretrained DNS TF-GridNet**, not the six-layer
random-weight local profile. It is the audited four-layer
`dns_ins20_epoch33` checkpoint (2,552,790 parameters), offline
whole-utterance inference.

## Artifacts

| Item | Value |
|---|---|
| Checkpoint | `checkpoints/tfgridnet_dns_epoch33.pth` |
| SHA-256 | `613db9fb4dafa7860d9e1390ec8f6bb61bba62cc0a67482bebcfbd0221d865c3` |
| Parameters | 2,552,790 |
| Profile | `dns_ins20_epoch33`, 4 layers, 16 kHz |
| Runtime | PyTorch 2.11.0+cu128, FP32 |
| Mode | offline whole-utterance; no streaming state |

Fetch and verify:

```sh
python -m scripts.fetch_tfgridnet_checkpoint
```

## Host benchmark

Host: 12th Gen Intel Core i5-12600KF (10C/16T), RTX 5060 Ti 16 GB, Windows.
Input `p232_023` (9.77 s) and its first 4 s. Latency includes the full model,
including STFT/ISTFT; no warm-up is included in the timed repeats.

| Device | Threads | Input | p50 latency | p95 latency | RTF | Real-time factor |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 1 | 4.0 s | 2,534 ms | 2,601 ms | 0.634 | 1.58x |
| CPU | 4 | 4.0 s | 1,197 ms | 1,207 ms | 0.299 | 3.34x |
| CPU | 4 | 9.77 s | 3,421 ms | 3,477 ms | 0.350 | 2.86x |
| CUDA | 4 | 4.0 s | 83.8 ms | 85.8 ms | 0.021 | 47.7x |

CUDA peak allocated memory: **344.9 MiB** for the 4 s run.

Quality on the same 4 s segment:

| Signal | PESQ-WB | STOI |
|---|---:|---:|
| Noisy | 1.3359 | 0.9046 |
| CPU FP32 enhanced | **2.8297** | **0.9605** |
| CUDA FP32 enhanced | **2.8385** | **0.9607** |

Full 9.77 s CPU: PESQ 1.3079 → **2.8261**, STOI 0.9373 → **0.9792**, RTF 0.350.

Run:

```sh
python -m scripts.bench_tfgridnet_inference --device cpu --threads 4 --seconds 4
python -m scripts.bench_tfgridnet_inference --device cuda --seconds 4
```

## Interpretation

- TF-GridNet is **not** a real-time edge model in this form. One CPU thread
  reaches only 1.58x realtime on a 4 s segment; four threads reach 3.34x, and
  the GPU reaches 47.7x. The 2.55M-parameter DNS profile is much heavier than
  GTCRN (48K) and needs a GPU or a stronger edge SoC for low-latency use.
- The model is offline and bidirectional; `stride=256` is the local 16 ms
  profile. This benchmark does not claim the paper's 8 ms hop.
- No phone or development board was connected (`adb` absent). The next physical
  target is an Android device with a native PyTorch/ONNX port, or a Jetson/ARM
  board with enough CPU/GPU headroom.
- The six-layer memory-policy study is a **training** result; this deployment
  check uses the separate four-layer DNS checkpoint and does not reuse the
  six-layer policy checkpoints.

## Next steps

1. Export a deployment graph (TorchScript or ONNX with the STFT outside the
   graph) and measure the same segments on the phone/board.
2. Try dynamic INT8 / FP16 and quantify PESQ–latency trade-offs.
3. Add a streaming wrapper only if the target application needs live audio;
   the current model is whole-utterance.
