"""No-download checks for pretrained CLI routing, report scope and rejection."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from scripts import evaluate as cli
from speech_denoising.models.base import BaseEnhancementModel


class Identity(BaseEnhancementModel):
    def forward(self, audio):
        return audio


class Fixture:
    """Synthetic data cannot claim the fixed 824-utterance test protocol."""
    split = "test"
    protocol_complete = False
    metadata = {"schema_version": 1, "split": "test", "protocol_complete": False}

    def __init__(self):
        rng = np.random.default_rng(210)
        self.clean = torch.from_numpy(rng.normal(0, 0.1, 16000).astype(np.float32))
        self.noisy = self.clean + torch.from_numpy(rng.normal(0, 0.03, 16000).astype(np.float32))

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"utterance_id": f"fixture_{index}", "sample_rate": 16000,
                "clean_audio": self.clean, "noisy_audio": self.noisy}


def reject(argv):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            cli.parse_args(argv)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("Invalid CLI invocation accepted: " + repr(argv))


def main():
    gt = cli.parse_args([])
    dns = cli.parse_args(["--model", "tfgridnet", "--profile", cli.DNS_PROFILE])
    assert gt.model == "gtcrn" and gt.profile == cli.GTCRN_PROFILE
    assert gt.checkpoint.name == "gtcrn_vctk.tar" and gt.output.name == "gtcrn_voicebank.json"
    assert dns.checkpoint.name == "tfgridnet_dns_epoch33.pth"
    assert dns.output != gt.output and dns.checkpoint != gt.checkpoint
    reject(["--model", "tfgridnet"])
    reject(["--model", "tfgridnet", "--profile", cli.GTCRN_PROFILE])
    reject(["--model", "gtcrn", "--profile", cli.DNS_PROFILE])
    reject(["--model", "lisennet"])
    for flag in ("--threads", "--repeats", "--limit"):
        reject([flag, "0"])
    reject(["--warmup", "-1"])
    reject(["--metrics", "si_snr", "si_snr"])
    reject(["--output", str(gt.manifest)])
    reject(["--output", str(gt.checkpoint)])
    # Both branches must construct the appropriate model and call its strict loader.
    sentinel = object()
    with patch.object(cli, "build_model", return_value=sentinel) as build, \
         patch.object(cli, "load_gtcrn_checkpoint", return_value={"strict": True}) as load:
        assert cli.load_pretrained(gt) == (sentinel, {"strict": True})
        build.assert_called_once_with("gtcrn")
        load.assert_called_once_with(sentinel, gt.checkpoint)
    with patch("speech_denoising.models.tfgridnet.profiles.build_tfgridnet_dns_model", return_value=sentinel) as build, \
         patch("speech_denoising.models.tfgridnet.checkpoint.load_tfgridnet_dns_checkpoint", return_value={"strict": True}) as load:
        assert cli.load_pretrained(dns) == (sentinel, {"strict": True})
        build.assert_called_once_with()
        load.assert_called_once_with(sentinel, dns.checkpoint)
    # Exercise main() and the actual shared metrics runner; replace only data/weights.
    with tempfile.TemporaryDirectory() as temporary:
        for name, profile in [("gtcrn", cli.GTCRN_PROFILE), ("tfgridnet", cli.DNS_PROFILE)]:
            target = Path(temporary) / f"{name}.json"
            with patch.object(cli, "VoiceBankDataset", return_value=Fixture()) as dataset, \
                 patch.object(cli, "load_pretrained", return_value=(Identity(), {"fixture": True})), \
                 contextlib.redirect_stdout(io.StringIO()):
                cli.main(["--model", name, "--profile", profile, "--metrics", "si_snr", "si_snr_improvement",
                          "--limit", "1", "--warmup", "0", "--repeats", "1", "--output", str(target)])
                dataset.assert_called_once_with(gt.manifest, split="test")
            report = json.loads(target.read_text(encoding="utf-8"))
            assert report["model"] == name and report["profile"] == profile
            assert report["scope"] == "subset_or_fixture" and report["evaluated_count"] == 1
            assert report["manifest_count"] == 2 and not report["all_quality_metrics_complete"]
            assert report["not_computed"] == ["pesq", "stoi"] and not report["failed_utterances"]
            assert report["summary"]["enhanced"]["si_snr_improvement"] == 0
            assert "speech_denoising/metrics/audio_metrics.py" in report["implementation"]["source_fingerprints"]
        # A strict loader failure must abort before evaluator entry or report creation.
        target = Path(temporary) / "rejected.json"
        with patch.object(cli, "VoiceBankDataset", return_value=Fixture()), \
             patch.object(cli, "load_pretrained", side_effect=ValueError("rejected checkpoint")), \
             patch.object(cli, "evaluate") as runner:
            try:
                cli.main(["--output", str(target)])
            except ValueError:
                pass
            else:
                raise AssertionError("Checkpoint rejection ignored")
            runner.assert_not_called()
            assert not target.exists()
    print("PASS: GTCRN compatibility, explicit DNS profile and strict-loader routing")
    print("PASS: common metrics/report runner, subset scope, omissions and load rejection")
    print("Synthetic fixtures only; no checkpoint download, training or VoiceBank quality scores.")


if __name__ == "__main__":
    main()
