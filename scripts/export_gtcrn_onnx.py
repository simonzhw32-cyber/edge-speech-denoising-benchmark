"""Export the paper-track GTCRN checkpoint to ONNX.

ONNX Runtime has no portable complex STFT op, so the exported graph is the
spectrum-domain network only: ``spec [B, 257, T, 2] -> enhanced [B, 257, T, 2]``
with dynamic batch and frame count.  The deployment script performs the
sqrt-Hann STFT/ISTFT around it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from speech_denoising.training.paper_gtcrn import GTCRNPaper


def load_checkpoint(path: Path) -> GTCRNPaper:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = GTCRNPaper().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    return model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/paper_runs/gtcrn_paper_seed42_100ep/checkpoints/epoch_0100.pt")
    parser.add_argument("--output", default="exports/gtcrn_paper_seed42_100ep_spec.onnx")
    parser.add_argument("--frames", type=int, default=251)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(ckpt)
    net = model.network.eval()
    dummy = torch.randn(1, 257, args.frames, 2)
    with torch.inference_mode():
        ref = net(dummy).numpy()

    torch.onnx.export(
        net,
        (dummy,),
        str(out),
        input_names=["spec"],
        output_names=["enhanced_spec"],
        dynamic_axes={"spec": {0: "batch", 2: "frames"}, "enhanced_spec": {0: "batch", 2: "frames"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(out))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(["enhanced_spec"], {"spec": dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(got - ref)))
    print(f"checkpoint={ckpt} sha256={sha256(ckpt)}")
    print(f"onnx={out} bytes={out.stat().st_size} sha256={sha256(out)}")
    print(f"input={tuple(dummy.shape)} output={tuple(got.shape)} max_abs_parity={max_abs:.3e}")
    if max_abs > 1e-4:
        raise SystemExit(f"parity too large: {max_abs}")


if __name__ == "__main__":
    main()
