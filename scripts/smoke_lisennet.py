"""Offline LiSenNet architecture/adapter checks, using random weights only."""

import base64
import hashlib
import json
import sys
import types
import zlib

import numpy as np
import torch

from speech_denoising.models import build_model
from speech_denoising.models.lisennet.model import LiSenNetModel
from speech_denoising.metrics.audio_metrics import evaluate

# Filled by the installer builder: unmodified, pinned official source snapshots.
REFERENCE_PAYLOAD = "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"
REFERENCE_HASHES = {
    "dpr_layer.py": "2d497284b7d2652cc3cfef5d5b967626739a28d2a6d99c5d754340c26d606c9c",
    "generator.py": "151e91c1ea7af699239f93530a5121b679fe22a88825106b873556f96625553e",
}


def official_reference():
    sources = json.loads(zlib.decompress(base64.b64decode(REFERENCE_PAYLOAD)))
    package_name = "_lisennet_pinned_smoke_reference"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    for filename in ("dpr_layer.py", "generator.py"):
        source = sources[filename]
        assert hashlib.sha256(source.encode()).hexdigest() == REFERENCE_HASHES[filename]
        # mel_scale is unused by the upstream enhancement forward path.
        # Removing only this import avoids requiring the torchaudio binary.
        source = source.replace("from torchaudio.functional import melscale_fbanks", "")
        name = package_name + "." + filename[:-3]
        module = types.ModuleType(name)
        module.__package__ = package_name
        sys.modules[name] = module
        exec(compile(source, "pinned_upstream/" + filename, "exec"), module.__dict__)
    return sys.modules[package_name + ".generator"].LiSenNet()


def must_reject(model, value, error):
    try:
        model(value)
    except error:
        return
    raise AssertionError("Invalid waveform was accepted")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = build_model("lisennet").eval()
    assert isinstance(model, LiSenNetModel)
    reference = official_reference().eval()
    state = model.network.state_dict()
    assert set(state) == set(reference.state_dict())
    reference.load_state_dict(state, strict=True)
    # Verify generator semantics independently of the adapter.
    with torch.inference_mode():
        src = torch.randn(2, 4097)
        torch.testing.assert_close(model.network(src)["est"], reference(src)["est"],
                                   rtol=1e-5, atol=1e-5)
        # Official data_module.py uses numpy.std with population variance.
        raw = torch.randn(2, 16001) * torch.tensor([[0.03], [0.3]])
        scale = np.std(raw.numpy(), axis=-1, keepdims=True) + 1e-8
        normalized = torch.from_numpy(raw.numpy() / scale)
        expected = reference(normalized)["est"] * torch.from_numpy(scale)
        torch.testing.assert_close(model(raw), expected, rtol=2e-5, atol=2e-5)
        for length in (1, 127, 256, 257, 511, 512, 513, 16000, 16001, 156302):
            value = torch.randn(1, length) * 0.1
            output = model(value)
            assert output.shape == value.shape
            assert output.dtype == value.dtype and output.device == value.device
            assert torch.isfinite(output).all()
        silence = torch.zeros(2, 513)
        assert torch.equal(model(silence), silence)
        # Independent utterances must not share hidden state or normalization.
        value = torch.randn(2, 2049)
        expected = torch.cat([model(row[None]) for row in value])
        torch.testing.assert_close(model(value), expected, rtol=2e-5, atol=2e-5)
        double_model = LiSenNetModel().double().eval()
        value = torch.randn(2, 513, dtype=torch.float64)
        output = double_model(value)
        assert output.dtype == torch.float64 and torch.isfinite(output).all()

    must_reject(model, torch.zeros(16000), ValueError)
    must_reject(model, torch.zeros(1, 0), ValueError)
    must_reject(model, torch.zeros(1, 512, dtype=torch.int16), TypeError)
    must_reject(model, torch.zeros(1, 512, dtype=torch.float64), ValueError)
    must_reject(model, torch.full((1, 512), float("nan")), ValueError)

    # Backpropagate without an optimizer step or weight update.
    model.train()
    value = (torch.randn(2, 2049) * 0.1).requires_grad_()
    model(value).square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    model.zero_grad(set_to_none=True)

    # Prove that the existing common evaluator accepts this waveform adapter.
    # These synthetic-fixture scores are deliberately discarded.
    clean = torch.randn(2049) * 0.1
    fixture = [{"utterance_id": "synthetic_only", "sample_rate": 16000,
                "clean_audio": clean, "noisy_audio": clean + torch.randn(2049) * 0.01}]
    report = evaluate(model, fixture, metrics=("si_snr", "si_snr_improvement"),
                      warmup=0, repeats=1)
    assert model.training
    assert report["scope"] == "subset_or_fixture" and not report["failed_utterances"]

    if torch.cuda.is_available():
        gpu_model = LiSenNetModel().cuda().eval()
        with torch.inference_mode():
            output = gpu_model(torch.randn(1, 513, device="cuda"))
        assert output.is_cuda and torch.isfinite(output).all()

    all_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PASS: LiSenNet upstream equivalence, waveform alignment, normalization and gradients")
    print("PASS: shared evaluate(model, dataset) interface; synthetic fixture only")
    print("Parameters (all / trainable):", all_params, trainable)
    print("Random weights only; no checkpoint, training, or dataset quality evaluation.")


if __name__ == "__main__":
    main()
