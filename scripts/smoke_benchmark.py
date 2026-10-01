"""No-download checks for SI-SNR math, manifest integrity and runner contracts."""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch

from speech_denoising.datasets.voicebank import VoiceBankDataset, rows_sha256
from speech_denoising.metrics.audio_metrics import si_snr, compute_audio_metrics, evaluate
from speech_denoising.models.base import BaseEnhancementModel


class Identity(BaseEnhancementModel):
    def forward(self, x):
        return x


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    clean = rng.normal(size=32000).astype(np.float32) * 0.1
    noise = rng.normal(size=32000).astype(np.float32) * 0.03
    noisy = clean + noise
    # Analytical zero-mean orthogonal signals with known target/error energy.
    assert abs(si_snr(np.array([1.5, -0.5, 0.5, -1.5]), np.array([1., -1., 1., -1.])) - 6.0205999133) < 1e-6
    assert abs(si_snr(noisy, clean) - si_snr(noisy * 3, clean)) < 1e-6
    scores = compute_audio_metrics(clean, noisy, noisy, metrics=("si_snr", "si_snr_improvement"))["values"]
    assert abs(scores["si_snr_improvement"]) < 1e-10
    try:
        si_snr(noisy, np.zeros_like(clean))
    except ValueError:
        pass
    else:
        raise AssertionError("Silent reference accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for i in range(2):
            row = {"utterance_id": f"fixture_{i}", "samples": len(clean)}
            for role, data in [("clean", clean), ("noisy", noisy)]:
                name = f"{role}_{i}.wav"
                sf.write(root / name, data, 16000, subtype="FLOAT")
                row[f"{role}_path"] = name
                row[f"{role}_sha256"] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            rows.append(row)
        manifest = root / "test_manifest.json"
        metadata = {"schema_version": 1, "split": "test", "sample_rate": 16000,
                    "protocol_complete": False, "rows_sha256": rows_sha256(rows), "utterances": rows}
        manifest.write_text(json.dumps(metadata))
        dataset = VoiceBankDataset(manifest, require_complete=False)
        model = Identity().train()
        result = evaluate(model, dataset, metrics=("si_snr", "si_snr_improvement"), warmup=0, repeats=1)
        assert model.training and result["scope"] == "subset_or_fixture"
        assert result["parameter_count"] == 0 and result["evaluated_count"] == 2
        assert result["summary"]["enhanced"]["si_snr_improvement"] == 0
        assert result["rtf"] >= 0 and not result["all_quality_metrics_complete"]
        # Corrupt an audio file; the loader must reject it.
        (root / "noisy_0.wav").write_bytes(b"corrupt")
        try:
            dataset[0]
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt audio accepted")
    print("PASS: benchmark metric math, manifest integrity, mode restoration and report scope")
    print("Fixture tests only; no VoiceBank score, training or download performed.")


if __name__ == "__main__":
    main()
