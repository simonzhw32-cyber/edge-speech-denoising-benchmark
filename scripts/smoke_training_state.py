"""Check state/validation components; toy updates only, never dataset training."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile

from scripts.plan_training import model_settings
from scripts.smoke_training_plan import synthetic_manifest
from speech_denoising.training.protocol import build_training_plan, fingerprint
from speech_denoising.training.selection import experiment_identity, select_checkpoint
from speech_denoising.utils.utils import REPO_ROOT, load_config


def rejected(operation):
    try:
        operation()
    except (ValueError, KeyError, TypeError, RuntimeError, FileExistsError):
        return
    raise AssertionError("Invalid state or validation input was accepted")


def metadata_checks():
    protocol = load_config("configs/training_protocol.yaml")
    source = json.loads((REPO_ROOT / "configs/voicebank.json").read_text())
    manifest = synthetic_manifest(source)
    plans = {name: build_training_plan(manifest, source, protocol, name,
                                      model_settings(protocol, name)) for name in protocol["models"]}
    identities = {name: experiment_identity(plan, 42) for name, plan in plans.items()}
    assert identities["tfgridnet"]["profile"] == "local_6layer"
    assert len({i["split_sha256"] for i in identities.values()}) == 1
    assert identities["gtcrn"] != experiment_identity(plans["gtcrn"], 43)
    for changed in ("protocol_sha256", "split_sha256"):
        bad = deepcopy(plans["gtcrn"])
        bad[changed] = "0" * 64
        rejected(lambda: experiment_identity(bad, 42))
    bad = deepcopy(plans["tfgridnet"])
    bad["model_settings"]["profile"] = "dns_ins20_epoch33"
    rejected(lambda: experiment_identity(bad, 42))
    rejected(lambda: experiment_identity(plans["gtcrn"], 99))
    identity = identities["gtcrn"]
    ids = ["fixture_a", "fixture_b"]
    candidate = {
        "schema_version": 1, "loss": "negative_si_snr", "reduction": "utterance_mean",
        "scope": "synthetic_fixture", "identity": identity, "epoch": 3,
        "checkpoint_sha256": "a" * 64, "validation_ids_sha256": fingerprint(ids),
        "count": 2, "mean_loss": -10.0,
    }
    best = select_checkpoint(None, candidate, identity, ids, synthetic=True)
    best["identity"]["seed"] = 43
    assert candidate["identity"]["seed"] == 42
    best = select_checkpoint(None, candidate, identity, ids, synthetic=True)
    for epoch, loss, expected_epoch in ((4, -11., 4), (4, -9., 3), (2, -10., 2), (5, -10., 3)):
        current = dict(candidate, epoch=epoch, mean_loss=loss)
        assert select_checkpoint(best, current, identity, ids, synthetic=True)["epoch"] == expected_epoch
    for key, value in (("count", 1), ("count", True), ("epoch", 0), ("mean_loss", float("nan")),
                       ("scope", "full_fixed_test"), ("checkpoint_sha256", "invalid"),
                       ("loss", "pesq"), ("validation_ids_sha256", "0" * 64)):
        bad = dict(candidate, **{key: value})
        rejected(lambda: select_checkpoint(best, bad, identity, ids, synthetic=True))
    rejected(lambda: select_checkpoint(best, dict(candidate, mean_loss=-12.), identity, ids, synthetic=True))
    rejected(lambda: select_checkpoint(None, candidate, identity, list(reversed(ids)), synthetic=True))
    rejected(lambda: select_checkpoint(None, candidate, identity, ids))
    rejected(lambda: select_checkpoint(None, candidate, identities["lisennet"], ids, synthetic=True))
    print("PASS: experiment identities, train/DNS distinction and complete-validation selection")
    return identity


def tensor_checks(identity):
    import numpy as np
    import torch
    from torch import nn
    from speech_denoising.losses.loss import EnhancementLoss
    from speech_denoising.training.state import (
        capture_rng, file_sha256, load_training_state, restore_rng, save_training_state,
    )
    from speech_denoising.training.validation import run_validation

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor) and torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.dropout = nn.Dropout(0.2)

        def forward(self, value):
            return self.dropout(self.linear(value))

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        data = torch.randn(2, 4) + random.random() + float(np.random.random())
        loss = (model(data) - 0.25).square().mean()
        loss.backward()
        optimizer.step()

    outer_rng = capture_rng()
    try:
        # CPU dropout and three RNG streams make an omitted RNG restoration visible.
        seed_all(42)
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch_1.pt"
            digest = save_training_state(path, model, optimizer, identity, completed_epoch=1, global_step=1)
            saved_rng = capture_rng()
            rejected(lambda: save_training_state(path, model, optimizer, identity,
                                                 completed_epoch=1, global_step=1))
            assert digest == file_sha256(path)
            step(model, optimizer)
            step(model, optimizer)
            expected_model = deepcopy(model.state_dict())
            expected_optimizer = deepcopy(optimizer.state_dict())
            expected_rng = capture_rng()
            restored = Toy()
            restored.eval()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.7)
            progress = load_training_state(path, restored, restored_optimizer, identity)
            assert progress == {"completed_epoch": 1, "next_epoch": 2, "global_step": 1,
                                "checkpoint_sha256": digest}
            assert restored.training and restored.dropout.training
            equal(capture_rng(), saved_rng)
            assert restored_optimizer.param_groups[0]["lr"] == 0.001
            assert all(p.grad is None for p in restored.parameters())
            step(restored, restored_optimizer)
            step(restored, restored_optimizer)
            equal(restored.state_dict(), expected_model)
            equal(restored_optimizer.state_dict(), expected_optimizer)
            equal(capture_rng(), expected_rng)
            before_model = deepcopy(restored.state_dict())
            before_optimizer = deepcopy(restored_optimizer.state_dict())
            before_rng = capture_rng()
            wrong = dict(identity, seed=43)
            rejected(lambda: load_training_state(path, restored, restored_optimizer, wrong))
            rejected(lambda: load_training_state(path, restored, torch.optim.SGD(restored.parameters(), lr=.1), identity))
            payload = torch.load(path, map_location="cpu", weights_only=True)
            # Force failure after model/optimizer restoration to exercise rollback.
            payload["rng"]["python"] = (0, (), None)
            broken = Path(directory) / "broken.pt"
            torch.save(payload, broken)
            rejected(lambda: load_training_state(broken, restored, restored_optimizer, identity))
            equal(restored.state_dict(), before_model)
            equal(restored_optimizer.state_dict(), before_optimizer)
            equal(capture_rng(), before_rng)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["model"]["linear.weight"][0, 0] = float("nan")
            torch.save(payload, broken)
            rejected(lambda: load_training_state(broken, restored, restored_optimizer, identity))
            equal(restored.state_dict(), before_model)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            first = next(iter(payload["optimizer"]["state"].values()))
            first["exp_avg"] = torch.zeros(1)
            torch.save(payload, broken)
            rejected(lambda: load_training_state(broken, restored, restored_optimizer, identity))
            equal(restored.state_dict(), before_model)
            equal(capture_rng(), before_rng)
            assert file_sha256(path) == digest
        print("PASS: toy AdamW resume matches uninterrupted parameters, moments, progress and RNG")
        print("PASS: checkpoint overwrite/type/identity rejection and failed-load rollback")

        class Waveform(nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = nn.Parameter(torch.tensor(0.8))
                self.dropout = nn.Dropout(.5)
                self.widths = []

            def forward(self, value):
                self.widths.append(tuple(value.shape))
                # Validation must restore RNG even if a forward consumes it.
                random.random(); np.random.random(); torch.rand(1)
                return self.dropout(value * self.gain)

        audio = []
        for identifier, width in (("fixture_a", 17), ("fixture_b", 31)):
            clean = torch.linspace(-1, 1, width)
            noisy = clean + .1 * torch.sin(torch.arange(width, dtype=torch.float32))
            audio.append({"utterance_id": identifier, "sample_rate": 16000,
                          "noisy_audio": noisy, "clean_audio": clean})
        ids = [sample["utterance_id"] for sample in audio]
        waveform = Waveform()
        waveform.train()
        waveform.dropout.eval()  # Preserve mixed module modes, not just root mode.
        waveform.gain.grad = torch.tensor(2.)
        before_rng = capture_rng()
        before_state = deepcopy(waveform.state_dict())
        report = run_validation(waveform, audio, identity, ids, epoch=1,
                                checkpoint_sha256="b" * 64, synthetic=True)
        oracle = EnhancementLoss(reduction="none")
        expected_loss = sum(oracle((item["noisy_audio"] * .8)[None],
                                   item["clean_audio"][None]).item() for item in audio) / 2
        assert abs(report["mean_loss"] - expected_loss) < 1e-6
        assert waveform.widths == [(1, 17), (1, 31)]
        assert waveform.training and not waveform.dropout.training
        equal(waveform.state_dict(), before_state)
        assert waveform.gain.grad.item() == 2. and report["count"] == 2
        equal(capture_rng(), before_rng)
        bad_audio = [audio[0], dict(audio[1], utterance_id="wrong")]
        rejected(lambda: run_validation(waveform, bad_audio, identity, ids, epoch=1,
                                        checkpoint_sha256="b" * 64, synthetic=True))
        equal(capture_rng(), before_rng)
        assert waveform.training and not waveform.dropout.training
        rejected(lambda: run_validation(waveform, audio, identity, ids, epoch=1,
                                        checkpoint_sha256="b" * 64))
        rejected(lambda: run_validation(waveform, audio[:1], identity, ids, epoch=1,
                                        checkpoint_sha256="b" * 64, synthetic=True))
        cropped = [dict(audio[0], crop_start=0), audio[1]]
        rejected(lambda: run_validation(waveform, cropped, identity, ids, epoch=1,
                                        checkpoint_sha256="b" * 64, synthetic=True))
        from speech_denoising.datasets.voicebank import VoiceBankDataset
        from torch.utils.data import Subset
        parent = VoiceBankDataset.__new__(VoiceBankDataset)
        parent.split = "test"
        parent.verify_files = parent.protocol_complete = True
        subset = Subset(parent, [0, 1])
        rejected(lambda: run_validation(waveform, subset, identity, ids, epoch=1,
                                        checkpoint_sha256="b" * 64))
        parent.split = "train"
        parent.metadata = {"split": "test", "source": {}}
        rejected(lambda: run_validation(waveform, subset, identity, ids, epoch=1,
                                        checkpoint_sha256="b" * 64, plan={
                                            "protocol": load_config("configs/training_protocol.yaml"),
                                            "model": "gtcrn", "model_settings": {}}))
        assert select_checkpoint(None, report, identity, ids, synthetic=True) == report
        print("PASS: full-utterance macro loss, batch-one validation, modes/RNG/gradient preservation")
    finally:
        restore_rng(outer_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="Check identities/selection without PyTorch")
    args = parser.parse_args()
    identity = metadata_checks()
    if args.metadata_only:
        print("Metadata checks only; tensor state and validation checks were not run.")
    else:
        tensor_checks(identity)
        print("Synthetic fixtures and toy optimizer updates only; no speech-model training or quality benchmark.")


if __name__ == "__main__":
    main()
