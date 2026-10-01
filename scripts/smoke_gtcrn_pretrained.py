"""Verify pinned weights, strict loading and upstream waveform equivalence."""

import argparse
from pathlib import Path
import tempfile

import torch

from speech_denoising.models import build_model
from speech_denoising.models.gtcrn.checkpoint import load_gtcrn_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/gtcrn_vctk.tar")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = build_model("gtcrn").eval()
    metadata = load_gtcrn_checkpoint(model, args.checkpoint)
    with torch.inference_mode():
        x = torch.randn(2, 16037) * 0.05
        y = model(x)
        spectrum = torch.view_as_real(torch.stft(x, 512, 256, 512, model.window, return_complex=True))
        reference_spectrum = model.network(spectrum)
        reference = torch.istft(torch.view_as_complex(reference_spectrum.contiguous()),
                                512, 256, 512, model.window, length=x.shape[-1])
        assert y.shape == x.shape and torch.isfinite(y).all()
        torch.testing.assert_close(y, reference)
        for length in [1, 257, 16000]:
            z = model(torch.zeros(1, length))
            assert z.shape == (1, length) and torch.isfinite(z).all()
    with tempfile.TemporaryDirectory() as directory:
        invalid = Path(directory) / "bad.tar"
        invalid.write_bytes(b"invalid checkpoint")
        try:
            load_gtcrn_checkpoint(model, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt checkpoint was accepted")
    print("PASS: pretrained GTCRN checksum, strict loading and reference equivalence")
    print("Matched checkpoint entries:", metadata["matched_state_entries"])
    print("Parameters (all / trainable):", metadata["parameter_count"], metadata["trainable_parameter_count"])
    print("No training or quality scores generated.")


if __name__ == "__main__":
    main()
