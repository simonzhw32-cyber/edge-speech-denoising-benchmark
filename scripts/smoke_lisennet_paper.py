"""CPU-only regression checks for the LiSenNet paper track.

No dataset, GPU, or checkpoint is required. These checks pin the corrections
found by the 2026-11-09 audit: crop-then-normalize order, tail-batch retention,
spectral-normalized discriminator heads, the official learnable-sigmoid shape,
deterministic validation sampling, and the 0.9/0.1/0.05 generator weighting.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from speech_denoising.models.lisennet.discriminator import Discriminator
from speech_denoising.models.lisennet.model import LiSenNetModel
from speech_denoising.training import paper_lisennet as paper


def main() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"[{'ok' if condition else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    rng = random.Random(42)
    noise_rng = np.random.RandomState(42)
    noisy = (np.sin(np.arange(48_000) / 37.0) * 3.0 +
             noise_rng.normal(0.0, 0.2, 48_000)).astype(np.float32)
    clean = (noisy * 0.4).astype(np.float32)
    crop_noisy, crop_clean = paper.random_crop_pair(noisy, clean, paper.CUT_LEN, rng)
    check("crop_length", crop_noisy.shape == (paper.CUT_LEN,))
    check("crop_std_is_one", abs(float(np.std(crop_noisy)) - 1.0) < 1e-5,
          f"std={float(np.std(crop_noisy)):.8f}")
    check("crop_clean_scaled_same", np.allclose(crop_clean, crop_noisy * 0.4, atol=1e-6))
    full_std = float(np.std(noisy))
    check("not_full_utterance_normalization", abs(full_std - 1.0) > 1e-3,
          f"full std={full_std:.4f}")

    indices = list(range(794))
    sample_a = paper.select_validation_indices(indices, 200)
    sample_b = paper.select_validation_indices(indices, 200)
    check("validation_sample_count", len(sample_a) == 200)
    check("validation_sample_unique", len(set(sample_a)) == 200)
    check("validation_sample_deterministic", sample_a == sample_b)
    check("validation_sample_covers_tail", max(sample_a) > 700,
          f"max={max(sample_a)}")
    check("validation_sample_not_stride", sample_a != indices[:200])

    check("tail_batch_retained", 10_778 % paper.BATCH_SIZE == 2)
    check("steps_include_tail",
          math.ceil(10_778 / paper.BATCH_SIZE) == 2_695)

    generator = LiSenNetModel()
    discriminator = Discriminator()
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    check("generator_parameters", gen_params == 36_783, str(gen_params))
    check("discriminator_parameters", disc_params == 181_650, str(disc_params))
    check("slope_shape", tuple(discriminator.layers[-1].slope.shape) == (1,))

    linear_params = [m for m in discriminator.modules()
                     if isinstance(m, torch.nn.Linear)]
    sn_linears = sum(1 for m in linear_params
                     if hasattr(m, "weight_orig") or
                     hasattr(m, "parametrizations"))
    check("both_linear_heads_spectral_norm", sn_linears == 2, f"{sn_linears}/2")

    with torch.no_grad():
        audio = torch.randn(2, paper.CUT_LEN)
        results = generator.network(audio, audio)
        outputs = discriminator(results["tgt_mag"], results["est_mag"])
        check("discriminator_output_count", len(outputs) == 19, str(len(outputs)))
        check("discriminator_scalar_shape", tuple(outputs[-1].shape) == (2, 1))
        check("discriminator_range",
              bool((outputs[-1] >= 0).all() and (outputs[-1] <= 1).all()))
        loss, parts = paper.generator_loss(discriminator, results)
        manual = (0.1 * parts["complex_loss"] +
                  0.9 * parts["magnitude_loss"] +
                  0.05 * parts["adversarial_loss"])
        check("generator_loss_weights", abs(float(loss) - manual) < 1e-6,
              f"{float(loss):.6f} vs {manual:.6f}")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("PASS: 19 LiSenNet paper-track checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
