"""Real DNS weights versus a pinned ESPnet 202308 release reference.

Synthetic waveform checks only; no downloads, training or VoiceBank scoring.
The training run's reported Git commit is unavailable; see PRETRAINED.md.
"""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import platform
import tempfile

import torch
from torch.nn import functional as F

from scripts._tfgridnet_dns_reference import REFERENCE_REVISION, reference_class
from speech_denoising.models.tfgridnet.assets import DEFAULT_CHECKPOINT, CHECKPOINT_SHA256
from speech_denoising.models.tfgridnet.checkpoint import _prepare_dns_state_dict, load_tfgridnet_dns_checkpoint
from speech_denoising.models.tfgridnet.model import TFGridNetModel
from speech_denoising.models.tfgridnet.profiles import build_tfgridnet_dns_model, DNS_MODEL_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[1]


def expect_rejection(call, label):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid input accepted: " + label)


def reference_output(reference, audio):
    # Encoder/decoder 'same' are identities in the published config. Its
    # non-segmented SeparateSpeech path performs separator -> same decoder,
    # and normalize_output_wav=False leaves the waveform scale unchanged.
    lengths = torch.full((audio.shape[0],), audio.shape[1], dtype=torch.long)
    return reference(audio, lengths)[0][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/tfgridnet_dns_pretrained_check.json")
    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve() or args.output.suffix.lower() != ".json":
        parser.error("Use a separate .json output path")
    torch.manual_seed(29)
    torch.set_num_threads(1)
    model = build_tfgridnet_dns_model().eval()
    metadata = load_tfgridnet_dns_checkpoint(model, args.checkpoint)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    Reference = reference_class()
    # Preserve the source's default legacy complex path, not our builtin path.
    reference = Reference(input_dim=0, n_srcs=1, n_imics=1, window="hann",
                          use_builtin_complex=False, **dict(DNS_MODEL_PARAMS)).eval()
    state = OrderedDict((key.removeprefix("separator."), value) for key, value in raw.items())
    reference.load_state_dict(state, strict=True)
    if len(reference.state_dict()) != 362:
        raise AssertionError("Unexpected reference state inventory")

    cases = []
    with torch.inference_mode():
        for batch, length in ((1, 768), (1, 1025), (1, 16000), (1, 16001), (2, 1025)):
            audio = torch.randn(batch, length) * 0.1
            actual, expected = model(audio), reference_output(reference, audio)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            if actual.shape != audio.shape or not torch.isfinite(actual).all():
                raise AssertionError("Invalid pretrained output")
            cases.append({"batch": batch, "samples": length,
                          "max_abs_difference": float((actual - expected).abs().max())})
        audio = torch.randn(2, 1025) * 0.1
        expected = torch.cat([model(row[None]) for row in audio])
        torch.testing.assert_close(model(audio), expected, rtol=2e-4, atol=2e-6)
        # A gain test catches accidental extra input or peak normalization.
        scaled = audio * 0.25
        torch.testing.assert_close(model(scaled), model(audio) * 0.25, rtol=1e-5, atol=1e-6)
        for length in (1, 2, 255, 256, 257, 767):
            audio = torch.randn(1, length) * 0.1
            padded = F.pad(audio, (0, model.minimum_samples - length))
            expected = reference_output(reference, padded)[..., :length]
            actual = model(audio)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            if actual.shape != audio.shape:
                raise AssertionError("Short audio alignment failed")
        for length in (1, 768, 16001):
            actual = model(torch.zeros(1, length))
            if not torch.equal(actual, torch.zeros_like(actual)):
                raise AssertionError("Silence must map to exact silence")
        if not torch.isfinite(model(torch.full((1, 1025), 0.1))).all():
            raise AssertionError("Constant-input guard failed")

    # Real-weight rejection checks: no copying on hash/structure failures.
    probe_key = next(iter(model.network.state_dict()))
    before = model.network.state_dict()[probe_key].clone()
    expect_rejection(lambda: load_tfgridnet_dns_checkpoint(TFGridNetModel(), args.checkpoint), "default six-layer profile")
    with tempfile.TemporaryDirectory(prefix="tfgridnet_dns_corrupt_") as temporary:
        corrupt = Path(temporary) / "corrupt.pth"
        damaged = bytearray(args.checkpoint.read_bytes())
        damaged[0] ^= 1
        corrupt.write_bytes(damaged)
        expect_rejection(lambda: load_tfgridnet_dns_checkpoint(model, corrupt), "corrupt hash")
    expected = model.network.state_dict()
    missing = OrderedDict(raw)
    missing.pop(next(iter(missing)))
    expect_rejection(lambda: _prepare_dns_state_dict(missing, expected), "missing tensor")
    unexpected = OrderedDict(raw)
    unexpected["optimizer.extra"] = unexpected.pop(next(iter(unexpected)))
    expect_rejection(lambda: _prepare_dns_state_dict(unexpected, expected), "unexpected namespace")
    tensor_key = next(iter(raw))
    wrong_shape = OrderedDict(raw)
    wrong_shape[tensor_key] = wrong_shape[tensor_key].reshape(-1)
    expect_rejection(lambda: _prepare_dns_state_dict(wrong_shape, expected), "wrong tensor shape")
    wrong_dtype = OrderedDict(raw)
    wrong_dtype[tensor_key] = wrong_dtype[tensor_key].double()
    expect_rejection(lambda: _prepare_dns_state_dict(wrong_dtype, expected), "wrong dtype")
    nonfinite = OrderedDict(raw)
    nonfinite[tensor_key] = nonfinite[tensor_key].clone()
    nonfinite[tensor_key].view(-1)[0] = float("nan")
    expect_rejection(lambda: _prepare_dns_state_dict(nonfinite, expected), "nonfinite tensor")
    torch.testing.assert_close(model.network.state_dict()[probe_key], before, rtol=0, atol=0)
    if metadata["sha256"] != CHECKPOINT_SHA256:
        raise AssertionError("Pinned checkpoint identity differs")

    report = {
        "scope": "pretrained_integration_check_synthetic_waveforms",
        "checkpoint": metadata, "reference_version": "ESPnet v.202308",
        "reference_revision": REFERENCE_REVISION,
        "reference_limit": "Version-matched public release; reported training commit could not be publicly retrieved.",
        "reference_complex_path": "legacy torch_complex; published separator default",
        "equivalence_cases": cases, "rtol": 1e-5, "atol": 1e-6,
        "checks": ["checksum", "strict_state", "finite_tensors", "release_reference_equivalence",
                   "batch_independence", "gain", "short_padding_alignment", "silence", "load_rejections"],
        "device": "cpu", "precision": "float32", "threads": 1,
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "platform": platform.platform(), "training_performed": False,
        "quality_scores_generated": False, "dataset_utterances_evaluated": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PASS: TF-GridNet DNS checksum, strict loading and finite tensors")
    print("PASS: pinned ESPnet 202308 release reference equivalence (legacy complex)")
    print("PASS: gain, batch independence, short alignment, silence and load rejection checks")
    print("Matched checkpoint entries:", metadata["matched_state_entries"])
    print("Parameters (all / trainable):", metadata["parameter_count"], metadata["trainable_parameter_count"])
    print("Maximum reference difference:", max(row["max_abs_difference"] for row in cases))
    print("Report:", args.output)
    print("Synthetic fixtures only; no training or dataset quality scores.")


if __name__ == "__main__":
    main()
