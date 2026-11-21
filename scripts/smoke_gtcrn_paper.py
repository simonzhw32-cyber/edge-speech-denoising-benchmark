"""CPU-only synthetic verification for the GTCRN paper track.

Runs the fifteen checks of §4.7 in
``docs/gtcrn_paper_track_plan_2026-11-09.md`` with no data, no GPU and no writes.
Every expected value was measured on this machine; tolerances are tight enough
to catch a porting error and loose enough for float32 reassociation.

    CUDA_VISIBLE_DEVICES=  PYTHONDONTWRITEBYTECODE=1  python scripts/smoke_gtcrn_paper.py

Coverage: 1-15 (test 7 is the degenerate all-zero case; test 13 checks the
guards; test 15 checks aligned cropping and wrap padding). Nothing is marked
TODO because every check below is implemented.
"""

from __future__ import annotations

import sys
import traceback

import torch

from speech_denoising.models.gtcrn.model import GTCRNModel
from speech_denoising.training.paper_gtcrn import GTCRNPaper, HybridLoss, _crop_pair

CHECKS: list[str] = []


def check(name):
    def decorate(function):
        CHECKS.append(name)
        return function

    return decorate


def _pair(batch=2, samples=16000, seed=0):
    generator = torch.Generator().manual_seed(seed)
    audio = torch.randn(batch, samples, generator=generator)
    return audio


@check("1 frontend_shapes")
def test_frontend_shapes():
    model = GTCRNPaper()
    for samples in (64000, 16000):
        spec = model.stft(_pair(2, samples))
        expected_frames = 1 + samples // 256
        assert spec.shape == (2, 257, expected_frames, 2), spec.shape
        assert spec.dtype == torch.float32
        assert spec.shape[2] == 1 + samples // model.hop_length


@check("2 istft_lengths")
def test_istft_lengths():
    model = GTCRNPaper()
    spec = model.stft(_pair(2, 16000))
    truncated = model.istft(spec)
    assert truncated.shape == (2, 15872), truncated.shape
    exact = model.istft(spec, length=16000)
    assert exact.shape == (2, 16000), exact.shape


@check("3 roundtrip")
def test_roundtrip():
    model = GTCRNPaper()
    audio = _pair(2, 32000, seed=3)
    rebuilt = model.istft(model.stft(audio), length=audio.shape[-1])
    assert torch.max(torch.abs(rebuilt - audio)) < 1e-5


@check("4 parity_with_adapter")
def test_parity_with_adapter():
    adapter = GTCRNModel().eval()
    paper = GTCRNPaper().eval()
    assert set(adapter.state_dict()) == set(paper.state_dict())
    paper.load_state_dict(adapter.state_dict())
    audio = _pair(1, 16000, seed=4)
    with torch.no_grad():
        assert torch.max(torch.abs(paper(audio) - adapter(audio))) < 1e-6


@check("5 loss_identity")
def test_loss_identity():
    loss_fn = HybridLoss()
    ref = torch.zeros(2, 257, 63, 2)
    ref[..., 0] = torch.randn(2, 257, 63, generator=torch.Generator().manual_seed(5))
    total, parts = loss_fn(ref, ref)
    assert parts["ri"] == 0.0 and parts["mag"] == 0.0, parts
    y = loss_fn._istft(ref)
    expected = (-torch.log10(torch.sum(y.square(), -1) / 1e-8 + 1e-8)).mean()
    assert abs(parts["sisnr"] - float(expected)) < 1e-4, (parts["sisnr"], float(expected))
    assert abs(parts["total"] - 0.01 * float(expected)) < 1e-5


@check("6 loss_scale_equivalence")
def test_loss_scale_equivalence():
    generator = torch.Generator().manual_seed(6)
    est = torch.randn(2, 257, 63, 2, generator=generator)
    ref = est + 0.3 * torch.randn(2, 257, 63, 2, generator=generator)
    paper_total, _ = HybridLoss(scale="paper")(est, ref)
    official_total, _ = HybridLoss(scale="official_x100")(est, ref)
    assert abs(float(official_total) - 100.0 * float(paper_total)) <= 1e-4 * max(1.0, float(official_total))


@check("7 loss_degenerate")
def test_loss_degenerate():
    zeros = torch.zeros(2, 257, 63, 2, requires_grad=True)
    total, parts = HybridLoss()(zeros, torch.zeros_like(zeros))
    assert abs(float(total.detach()) - 0.08) < 1e-6, float(total.detach())
    total.backward()
    assert torch.isfinite(zeros.grad).all()


@check("8 ri_is_per_signal")
def test_ri_is_per_signal():
    # Five identical frames: ISTFT needs >1 frame, and the mse normalizes over
    # frames too, so scaling back by the frequency count recovers the bin value.
    est = torch.zeros(1, 257, 5, 2)
    ref = torch.zeros(1, 257, 5, 2)
    ref[0, 0, :, 0] = 1.0
    est[0, 0, :, 0] = 0.01
    _, parts = HybridLoss()(est, ref)
    # mse divides by the whole element count (1*257*1); scale back to the bin.
    closed_form = (1.0 - 0.01 ** 0.3) ** 2
    clean_only = (1.0 - 0.01) ** 2
    measured = parts["ri"] * 257.0
    assert abs(measured - closed_form) < 1e-5, (measured, closed_form)
    assert abs(measured - clean_only) > 1e-1, (measured, clean_only)


@check("9 sisnr_is_log10")
def test_sisnr_is_log10():
    generator = torch.Generator().manual_seed(9)
    est = torch.randn(2, 257, 63, 2, generator=generator)
    ref = torch.randn(2, 257, 63, 2, generator=generator)
    _, parts = HybridLoss()(est, ref)
    loss_fn = HybridLoss()
    y_pred, y_true = loss_fn._istft(est), loss_fn._istft(ref)
    projection = (
        torch.sum(y_true * y_pred, -1, keepdim=True)
        * y_true
        / (torch.sum(y_true.square(), -1, keepdim=True) + 1e-8)
    )
    ratio = torch.norm(projection, dim=-1, keepdim=True).square() / (
        torch.norm(y_pred - projection, dim=-1, keepdim=True).square() + 1e-8
    )
    expected = -torch.log10(ratio + 1e-8).mean()
    assert abs(parts["sisnr"] - float(expected)) < 1e-6
    decibels = -10.0 * torch.log10(ratio + 1e-8).mean()
    assert abs(float(decibels) - float(expected)) > 1.0


@check("10 gradients")
def test_gradients():
    generator = torch.Generator().manual_seed(10)
    est = torch.randn(2, 257, 63, 2, generator=generator).requires_grad_(True)
    ref = torch.randn(2, 257, 63, 2, generator=generator)
    total, _ = HybridLoss()(est, ref)
    total.backward()
    assert torch.isfinite(est.grad).all()
    assert float(est.grad.abs().max()) > 0


@check("11 scheduler_trace")
def test_scheduler_trace():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    rates = []
    for _ in range(13):
        scheduler.step(1.0)
        rates.append(optimizer.param_groups[0]["lr"])
    expected = [1e-3] * 6 + [5e-4] * 6 + [2.5e-4]
    assert all(abs(a - b) < 1e-12 for a, b in zip(rates, expected)), rates
    try:
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", verbose=True)
    except TypeError:
        pass
    else:
        raise AssertionError("this torch build accepts verbose=; docstring expectation is stale")


@check("12 frozen_erb")
def test_frozen_erb():
    model = GTCRNPaper()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    assert (total, trainable, tensors) == (48245, 23669, 203), (total, trainable, tensors)
    frozen = model.network.erb.erb_fc.weight.detach().clone()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    noisy, clean = _pair(2, 16000, seed=12), _pair(2, 16000, seed=13)
    est = model.enhance_spec(model.stft(noisy))
    loss, _ = HybridLoss()(est, model.stft(clean))
    loss.backward()
    optimizer.step()
    assert torch.equal(frozen, model.network.erb.erb_fc.weight)


@check("13 guards")
def test_guards():
    model = GTCRNPaper()
    loss_fn = HybridLoss()
    for callable_ in (
        lambda: model.stft(torch.randn(2, 100, 3)),
        lambda: model.stft(torch.randn(2, 1000, dtype=torch.complex64)),
        lambda: model.istft(torch.randn(2, 128, 63, 2)),
        lambda: loss_fn(torch.full((2, 257, 63, 2), float("nan")), torch.zeros(2, 257, 63, 2)),
        lambda: loss_fn(torch.zeros(0, 257, 63, 2), torch.zeros(0, 257, 63, 2)),
    ):
        try:
            callable_()
        except (ValueError, TypeError):
            continue
        raise AssertionError("guard did not fire")


@check("14 oracle_equivalence")
def test_oracle_equivalence():
    from speech_denoising.losses.gtcrn import GTCRNHybridLoss

    model = GTCRNPaper()
    generator = torch.Generator().manual_seed(14)
    clean = torch.randn(2, 16000, generator=generator)
    noisy = clean + 0.1 * torch.randn(2, 16000, generator=generator)
    oracle = GTCRNHybridLoss()(noisy, clean)
    mine, _ = HybridLoss(scale="official_x100")(model.stft(noisy), model.stft(clean))
    assert abs(float(mine) - float(oracle)) < 1e-3, (float(mine), float(oracle))


@check("15 aligned_crop")
def test_aligned_crop():
    noisy = torch.arange(100_000, dtype=torch.float32)
    clean = noisy + 1_000_000.0
    for _ in range(8):
        noisy_crop, clean_crop, wrapped = _crop_pair(noisy, clean, 64_000)
        assert not wrapped
        offset = int(noisy_crop[0].item())
        assert int(clean_crop[0].item()) - 1_000_000 == offset
        assert torch.equal(noisy_crop, noisy[offset:offset + 64_000])
        assert torch.equal(clean_crop, clean[offset:offset + 64_000])
    short = torch.arange(1_000, dtype=torch.float32)
    short_crop, short_clean, wrapped = _crop_pair(short, short + 1.0, 4_000)
    assert wrapped and short_crop.shape == (4_000,)
    assert torch.equal(short_crop[:1_000], short)
    assert torch.equal(short_crop[1_000:2_000], short)
    assert torch.equal(short_clean, short_crop + 1.0)
    try:
        _crop_pair(short, short[:-1], 4_000)
    except ValueError:
        pass
    else:
        raise AssertionError("unaligned pair guard did not fire")


def main() -> int:
    torch.manual_seed(0)
    failures = 0
    for name in CHECKS:
        function = globals()["test_" + name.split(" ", 1)[1]]
        try:
            function()
        except Exception:  # noqa: BLE001 - a smoke test reports, it does not raise
            failures += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
        else:
            print(f"[ok] {name}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
