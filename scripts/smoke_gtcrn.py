"""GTCRN integration checks with random weights; no training or checkpoints."""

import torch

from speech_denoising.models import build_model
from speech_denoising.models.gtcrn.model import GTCRNModel


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = build_model("gtcrn").eval()
    assert isinstance(model, GTCRNModel)
    with torch.inference_mode():
        for batch, length in [(1, 1), (1, 256), (1, 257), (1, 16000), (2, 16037)]:
            x = torch.randn(batch, length)
            y = model(x)
            assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
            assert torch.isfinite(y).all()
        x = torch.randn(2, 16000)
        y = model(x)
        individual = torch.cat([model(x[i:i+1]) for i in range(2)])
        torch.testing.assert_close(y, individual, atol=1e-5, rtol=1e-5)
        # Reproduce the pinned upstream frontend with its legacy STFT layout.
        legacy_spec = torch.stft(x, 512, 256, 512, model.window, return_complex=False)
        legacy_out = model.network(legacy_spec)
        reference = torch.istft(torch.view_as_complex(legacy_out.contiguous()),
                                512, 256, 512, model.window, length=x.shape[-1])
        torch.testing.assert_close(y, reference)
        assert torch.isfinite(model(torch.zeros(1, 16000))).all()
    for bad in [torch.zeros(16000), torch.zeros(1, 0), torch.zeros(1, 512, dtype=torch.int64)]:
        try:
            model(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    # One differentiation check, no optimizer or parameter update.
    x = torch.randn(1, 1024, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    model.zero_grad(set_to_none=True)
    print("PASS: GTCRN forward, alignment, reference equivalence and gradients")
    print("Input: [1, 16000]; output: [1, 16000]")
    print("Random weights only; no checkpoint, training, or quality evaluation.")


if __name__ == "__main__":
    main()
