"""Synthetic checks for waveform loss/crops/batches; backward only, no updates."""

import math

import numpy as np
import torch

from speech_denoising.losses.loss import EnhancementLoss
from speech_denoising.metrics.audio_metrics import si_snr
from speech_denoising.training.batching import collate_audio_pairs, crop_offset, paired_crop


def expect_rejection(operation):
    try:
        operation()
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid training-component input was accepted")


def check_loss():
    generator = torch.Generator().manual_seed(817)
    clean = torch.randn(2, 13, dtype=torch.float64, generator=generator)
    estimate = (0.7 * clean + 0.2 * torch.randn(2, 13, dtype=torch.float64, generator=generator))
    lengths = torch.tensor([7, 13])
    expected = torch.tensor([-si_snr(estimate[i, :n], clean[i, :n])
                             for i, n in enumerate(lengths.tolist())], dtype=torch.float64)
    per_item = EnhancementLoss(reduction="none")(estimate, clean, lengths)
    torch.testing.assert_close(per_item, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(EnhancementLoss()(estimate, clean, lengths), expected.mean())
    full = EnhancementLoss()(estimate, clean)
    torch.testing.assert_close(full, torch.tensor(np.mean([
        -si_snr(a, b) for a, b in zip(estimate, clean)
    ]), dtype=torch.float64))
    shifted = EnhancementLoss(reduction="none")(estimate * 2.7 + 5.0, clean - 3.0, lengths)
    torch.testing.assert_close(shifted, per_item, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(EnhancementLoss()(estimate.float(), clean.float(), lengths),
                               expected.float().mean(), atol=2e-5, rtol=2e-5)
    padded = torch.cat([estimate, torch.full((2, 6), float("nan"), dtype=torch.float64)], dim=1)
    target = torch.cat([clean, torch.full((2, 6), float("inf"), dtype=torch.float64)], dim=1)
    padded[0, 7:] = float("nan")
    target[0, 7:] = float("inf")
    padded.requires_grad_()
    value = EnhancementLoss()(padded, target, lengths)
    torch.testing.assert_close(value, expected.mean())
    value.backward()
    assert torch.isfinite(padded.grad).all()
    assert torch.count_nonzero(padded.grad[0, 7:]) == 0
    assert torch.count_nonzero(padded.grad[1, 13:]) == 0
    # Each valid slice must have exactly the same gradient as its unpadded counterpart.
    for i, n in enumerate(lengths.tolist()):
        item = estimate[i:i + 1, :n].clone().requires_grad_()
        EnhancementLoss()(item, clean[i:i + 1, :n]).backward()
        torch.testing.assert_close(padded.grad[i, :n], item.grad[0] / 2)
    small = torch.randn(2, 6, dtype=torch.float64, generator=generator, requires_grad=True)
    reference = torch.randn(2, 6, dtype=torch.float64, generator=generator)
    valid = torch.tensor([4, 6])
    assert torch.autograd.gradcheck(lambda x: EnhancementLoss()(x, reference, valid),
                                   (small,), eps=1e-6, atol=1e-5, rtol=1e-4)
    expect_rejection(lambda: EnhancementLoss()(estimate, torch.ones_like(clean), lengths))
    expect_rejection(lambda: EnhancementLoss()(estimate, torch.zeros_like(clean), lengths))
    expect_rejection(lambda: EnhancementLoss()(estimate, clean, torch.tensor([0, 13])))
    expect_rejection(lambda: EnhancementLoss()(estimate, clean, torch.tensor([7, 14])))
    expect_rejection(lambda: EnhancementLoss()(estimate, clean, lengths.float()))
    expect_rejection(lambda: EnhancementLoss()(estimate, clean[:, :-1]))
    expect_rejection(lambda: EnhancementLoss()(estimate.float(), clean))
    expect_rejection(lambda: EnhancementLoss()(estimate.half(), clean.half()))
    invalid = estimate.clone()
    invalid[0, 0] = float("nan")
    expect_rejection(lambda: EnhancementLoss()(invalid, clean, lengths))
    expect_rejection(lambda: EnhancementLoss(eps=float("nan")))
    expect_rejection(lambda: EnhancementLoss(reduction="sum"))


def pair(identifier, count):
    noisy = torch.arange(count, dtype=torch.float64) / 10
    return {"utterance_id": identifier, "sample_rate": 16000,
            "noisy_audio": noisy, "clean_audio": noisy + 2}


def check_batching():
    raw = pair("p226_001", 19)
    before = raw["noisy_audio"].clone()
    cropped = paired_crop(raw, segment_samples=8, seed=42, epoch=3)
    repeated = paired_crop(raw, segment_samples=8, seed=42, epoch=3)
    assert cropped["crop_start"] == repeated["crop_start"]
    torch.testing.assert_close(cropped["noisy_audio"], repeated["noisy_audio"])
    start = cropped["crop_start"]
    torch.testing.assert_close(cropped["noisy_audio"], raw["noisy_audio"][start:start + 8])
    torch.testing.assert_close(cropped["clean_audio"], raw["clean_audio"][start:start + 8])
    torch.manual_seed(999)
    torch.randn(100)
    assert crop_offset("p226_001", 19, 8, 42, 3) == start
    offsets = {crop_offset("p226_001", 19, 8, 42, epoch) for epoch in range(20)}
    assert len(offsets) > 1 and all(0 <= x <= 11 for x in offsets)
    small = paired_crop(pair("p227_001", 3), segment_samples=8, seed=42, epoch=3)
    assert small["valid_samples"] == 3 and small["crop_start"] == 0
    assert torch.count_nonzero(small["noisy_audio"][3:]) == 0
    assert torch.count_nonzero(small["clean_audio"][3:]) == 0
    batch = collate_audio_pairs([cropped, small])
    assert batch["noisy_audio"].shape == batch["clean_audio"].shape == (2, 8)
    assert batch["lengths"].tolist() == [8, 3] and batch["lengths"].dtype == torch.int64
    assert batch["utterance_ids"] == ["p226_001", "p227_001"]
    assert batch["crop_starts"] == [start, 0]
    assert torch.count_nonzero(batch["clean_audio"][1, 3:]) == 0
    reversed_batch = collate_audio_pairs([small, cropped])
    torch.testing.assert_close(reversed_batch["noisy_audio"].flip(0), batch["noisy_audio"])
    mixed = collate_audio_pairs([pair("p228_001", 5), pair("p229_001", 11)])
    assert mixed["lengths"].tolist() == [5, 11] and mixed["clean_audio"].shape == (2, 11)
    assert torch.count_nonzero(mixed["noisy_audio"][0, 5:]) == 0
    # Crop output owns its storage and cannot change the source pair.
    cropped["noisy_audio"][0] += 100
    torch.testing.assert_close(raw["noisy_audio"], before)
    expect_rejection(lambda: collate_audio_pairs([]))
    expect_rejection(lambda: paired_crop(small, segment_samples=8, seed=42, epoch=3))
    expect_rejection(lambda: crop_offset("p226_001", 19, 8, True, 3))
    expect_rejection(lambda: crop_offset("p226_001", 19, 1, 42, 3))
    wrong_rate = dict(raw, sample_rate=8000)
    expect_rejection(lambda: paired_crop(wrong_rate, segment_samples=8, seed=42, epoch=3))
    wrong_length = dict(raw, clean_audio=raw["clean_audio"][:-1])
    expect_rejection(lambda: collate_audio_pairs([wrong_length]))
    wrong_dtype = dict(raw, clean_audio=raw["clean_audio"].float())
    expect_rejection(lambda: collate_audio_pairs([wrong_dtype]))
    too_long = dict(small, valid_samples=9)
    expect_rejection(lambda: collate_audio_pairs([too_long]))
    expect_rejection(lambda: collate_audio_pairs([
        raw, {key: value.float() if isinstance(value, torch.Tensor) else value
              for key, value in pair("p230_001", 5).items()}
    ]))


def check_model_backward():
    from speech_denoising.models import build_model

    generator = torch.Generator().manual_seed(38)
    clean = torch.randn(2300, generator=generator) * 0.1
    sample = {"utterance_id": "fixture_001", "sample_rate": 16000, "clean_audio": clean,
              "noisy_audio": clean + torch.randn(2300, generator=generator) * 0.03}
    batch = collate_audio_pairs([paired_crop(sample, segment_samples=2048, seed=42, epoch=0)])
    for name in ("gtcrn", "lisennet", "tfgridnet"):
        torch.manual_seed(47)
        model = build_model(name).eval()
        parameters = [(p, p.detach().clone()) for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        enhanced = model(batch["noisy_audio"])
        value = EnhancementLoss()(enhanced, batch["clean_audio"], batch["lengths"])
        assert math.isfinite(value.item())
        value.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) > 0 for g in gradients)
        assert all(torch.equal(p.detach(), before) for p, before in parameters)
        del model, parameters, gradients, enhanced, value


def main():
    torch.set_num_threads(1)
    check_loss()
    print("PASS: SI-SNR reference values, valid-length masking, finite gradients and gradcheck")
    check_batching()
    print("PASS: paired crops, deterministic offsets, padding, batch alignment and rejection checks")
    check_model_backward()
    print("PASS: all three random-weight models backpropagate the shared loss; parameters unchanged")
    print("Synthetic fixtures only; no optimizer steps, assets, training run or quality benchmark.")


if __name__ == "__main__":
    main()
