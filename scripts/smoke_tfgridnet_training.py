"""Generated CPU fixtures only: accumulation and six-layer TF-GridNet persistence."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from scripts.smoke_real_training import tiny_spec, rejected
from speech_denoising.training.protocol import fingerprint
from speech_denoising.training.selection import execution_identity
from speech_denoising.training.real_store import validate_spec, audit_run, checkpoint_path
from speech_denoising.utils.utils import REPO_ROOT


def fixture_spec():
    spec = tiny_spec(model="tfgridnet")
    plan = spec["plan"]
    plan["protocol"]["training"].update(batch_size=4, segment_samples=128)
    plan["protocol_sha256"] = fingerprint(plan["protocol"])
    plan["model_settings"]["constructor_kwargs"].update(
        n_fft=32, stride=16, n_layers=6, lstm_hidden_units=4,
        attn_n_head=1, attn_approx_qk_dim=4, emb_dim=4)
    spec["settings"].update(batch_size=4, segment_samples=128,
        micro_batch_size=1, gradient_accumulation_steps=4,
        accumulation_reduction="actual_group_utterance_mean")
    spec["identity"] = execution_identity(plan, 42, "random")
    spec["plan_sha256"] = fingerprint(plan)
    return spec


def metadata_checks():
    spec = fixture_spec()
    validate_spec(spec)
    for key, value in (("micro_batch_size", 2), ("gradient_accumulation_steps", 2),
                       ("accumulation_reduction", "fixed_four")):
        changed = deepcopy(spec)
        changed["settings"][key] = value
        rejected(lambda: validate_spec(changed))
    from speech_denoising.training.real_runner import source_files, build_run_spec
    names = source_files("tfgridnet")
    assert "speech_denoising/training/accumulation.py" in names
    assert all((REPO_ROOT / name).is_file() for name in names)
    rejected(lambda: build_run_spec("unused", "unused", seed=42, device="cpu",
        initialization="pretrained", model="tfgridnet"))
    print("PASS: six-layer random identity and explicit accumulation settings")


def accumulation_check():
    import torch
    from torch import nn
    from speech_denoising.training.runner import train_one_epoch
    from speech_denoising.training.accumulation import train_accumulated_epoch
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        def forward(self, x):
            return self.conv(x[:, None])[:, 0]
    torch.manual_seed(8)
    rows = []
    for i in range(5):
        clean = torch.randn(160)
        rows.append(dict(utterance_id=f"p226_{i:05d}", sample_rate=16000,
                         clean_audio=clean, noisy_audio=clean + .1 * torch.randn(160)))
    settings = fixture_spec()["settings"]
    a, b = Model(), Model()
    b.load_state_dict(a.state_dict())
    oa, ob = torch.optim.AdamW(a.parameters(), lr=.001), torch.optim.AdamW(b.parameters(), lr=.001)
    ids = [r["utterance_id"] for r in rows]
    ra = train_one_epoch(a, oa, rows, ids, seed=42, epoch=1, settings=settings)
    rb = train_accumulated_epoch(b, ob, rows, ids, seed=42, epoch=1, settings=settings)
    assert ra["count"] == rb["count"] == 5 and ra["global_step"] == rb["global_step"] == 2
    assert abs(ra["mean_loss"] - rb["mean_loss"]) < 1e-4
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)
    assert all(p.grad is None for p in b.parameters())
    print("PASS: accumulated updates match batched toy updates, including final one-item group")


def persistence_check(folder):
    import torch
    from speech_denoising.training import real_runner, protocol as protocol_module
    from speech_denoising.datasets.voicebank import VoiceBankDataset
    from speech_denoising.training.data import TrainingSubset
    spec = fixture_spec()
    plan = spec["plan"]
    rows = {}
    for i, identifier in enumerate(plan["split"]["train_ids"] + plan["split"]["validation_ids"]):
        g = torch.Generator().manual_seed(100+i)
        clean = .2 * torch.randn(160+i, generator=g)
        rows[identifier] = dict(utterance_id=identifier, sample_rate=16000,
            clean_audio=clean, noisy_audio=clean + .05*torch.randn(clean.shape, generator=g))
    parent = VoiceBankDataset.__new__(VoiceBankDataset)
    parent.split = "train"
    parent.verify_files = parent.protocol_complete = True
    parent.metadata = {"source": plan["source"]}
    class Dataset(TrainingSubset):
        def __init__(self, path, supplied_plan, split):
            assert supplied_plan == plan
            self.split = split
            self.utterance_ids = tuple(plan["split"][split+"_ids"])
            self.plan_sha256 = fingerprint(plan)
            self._parent = parent
        def __len__(self):
            return len(self.utterance_ids)
        def __getitem__(self, i):
            return rows[self.utterance_ids[i]]
    initial = None
    def context(saved_spec, **kwargs):
        nonlocal initial
        torch.manual_seed(7)
        model = real_runner.build_training_model(saved_spec)
        if initial is None:
            initial = deepcopy(model.state_dict())
        return model, torch.optim.AdamW(model.parameters(), lr=.001)
    run = folder / "tfgridnet"
    with patch.object(real_runner, "build_run_spec", return_value=deepcopy(spec)), \
            patch.object(real_runner, "_verify_current_inputs", return_value=(Path("fixture"), plan)), \
            patch.object(real_runner, "_context", side_effect=context), \
            patch.object(real_runner, "TrainingSubset", Dataset), \
            patch.object(protocol_module, "build_training_plan", return_value=plan):
        def train(epoch):
            return real_runner.train_real_epoch("fixture", "fixture", run, epoch,
                seed=42, device="cpu", initialization="random", model="tfgridnet")
        first = train(1)
        assert first["count"] == 3 and first["global_step"] == 1
        rejected(lambda: train(2))
        state = torch.load(checkpoint_path(run, 1), weights_only=True, map_location="cpu")
        assert any(not torch.equal(v, initial[k]) for k,v in state["model"].items())
        real_runner.validate_real_epoch("fixture", "fixture", run, 1, device="cpu")
        changed = deepcopy(spec)
        changed["git"]["commit"] = "6"*40
        with patch.object(real_runner, "build_run_spec", return_value=changed):
            second = train(2)
        assert second["global_step"] == 2
        real_runner.validate_real_epoch("fixture", "fixture", run, 2, device="cpu")
        stored, records, reports = audit_run(run, require_selection=True)
        assert stored == spec and len(records) == len(reports) == 2
    print("PASS: tiny six-layer TF-GridNet parameter updates, saved optimizer/RNG, resume and validation")
    from scripts import evaluate_trained
    class TestFixture:
        protocol_complete = True
        def __init__(self, *args, **kwargs):
            assert kwargs == {"split": "test", "verify_files": True, "require_complete": True}
        def __len__(self):
            return 824
    output = REPO_ROOT / "results/trained_reports" / (folder.name + "_tfgridnet.json")
    mock_report = dict(scope="synthetic_test_routing_fixture", evaluated_count=824,
        manifest_count=824, summary={"enhanced": {}}, failed_utterances=[])
    try:
        with patch.object(evaluate_trained, "verify_run_sources"), \
                patch.object(evaluate_trained, "VoiceBankDataset", TestFixture), \
                patch.object(evaluate_trained, "evaluate", return_value=mock_report):
            evaluate_trained.main(["--run-dir", str(run), "--manifest", "fixture",
                                  "--output", str(output), "--device", "cpu"])
        import json
        result=json.loads(output.read_text())
        assert result["model"] == "tfgridnet" and result["profile"] == "local_6layer"
    finally:
        output.unlink(missing_ok=True)
    print("PASS: selected TF-GridNet checkpoint loads into shared test routing (metrics mocked)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args=parser.parse_args()
    metadata_checks()
    if args.metadata_only:
        return
    import torch
    torch.set_num_threads(2)
    accumulation_check()
    root=REPO_ROOT/"results/training_runs"
    root.mkdir(parents=True, exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix="smoke_tf_", dir=root))
    try:
        persistence_check(folder)
    finally:
        shutil.rmtree(folder)
    print("Generated CPU audio only; no real-data training or quality benchmark")

if __name__ == "__main__":
    main()
