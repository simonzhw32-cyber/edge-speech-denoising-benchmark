"""Check one configured train batch and one full validation utterance; no updates."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random

from speech_denoising.training.data import TrainingSubset
from speech_denoising.training.probe import (bind_preflight, output_path, probe_model,
                                            probe_selection, publish_report, source_hashes)
from speech_denoising.training.protocol import fingerprint
from speech_denoising.utils.utils import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["gtcrn", "lisennet", "tfgridnet"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        protected = (args.manifest, args.preflight)
        output = output_path(args.output, protected)
        path, data, manifest, plan, preflight_bytes = bind_preflight(
            args.manifest, args.preflight, args.model)
        selected = probe_selection(manifest, plan)
        names = ["configs/default.yaml", "configs/voicebank.json", "configs/training_protocol.yaml",
                 plan["model_settings"]["config"], "scripts/plan_training.py", "scripts/probe_training.py"]
        names += sorted(p.relative_to(REPO_ROOT).as_posix()
                        for p in (REPO_ROOT / "speech_denoising").rglob("*.py"))
        hashes = source_hashes(names)

        import numpy as np
        import torch
        from speech_denoising.models import MODEL_REGISTRY
        from speech_denoising.training.batching import paired_crop, collate_audio_pairs
        from speech_denoising.training.state import capture_rng, restore_rng
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; the probe does not fall back to CPU")
        device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
        training = TrainingSubset(path, plan, "train")
        validation = TrainingSubset(path, plan, "validation")
        crops = [paired_crop(training[index], segment_samples=selected["segment_samples"],
                             seed=selected["seed"], epoch=selected["epoch"])
                 for index in range(len(selected["train_ids"]))]
        train_batch = collate_audio_pairs(crops)
        validation_batch = collate_audio_pairs([
            validation[validation.utterance_ids.index(selected["validation_id"])]])
        if train_batch["utterance_ids"] != selected["train_ids"]:
            raise ValueError("Training batch IDs differ from the bounded selection")
        selected["crop_starts"] = train_batch["crop_starts"]
        selected["valid_lengths"] = train_batch["lengths"].tolist()
        runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": np.__version__, "cuda_build": torch.version.cuda,
                   "device": str(device), "platform": platform.platform(),
                   "cpu_threads": torch.get_num_threads(),
                   "interop_threads": torch.get_num_interop_threads(),
                   "autocast": False, "tensor_dtype": "float32",
                   "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            free, total = torch.cuda.mem_get_info(device)
            runtime.update(gpu=properties.name, compute_capability=list(torch.cuda.get_device_capability(device)),
                           gpu_total_bytes=total, gpu_free_bytes_before_model=free)
        rng = capture_rng()
        try:
            random.seed(selected["seed"])
            np.random.seed(selected["seed"])
            torch.manual_seed(selected["seed"])
            model = MODEL_REGISTRY[args.model](**plan["model_settings"]["constructor_kwargs"])
            parameters = {"all": sum(p.numel() for p in model.parameters()),
                          "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            print(f"Probing {args.model} on {device}: {len(crops)} x {selected['segment_samples']} train samples; "
                  f"validation {selected['validation_id']} ({selected['validation_samples']} samples)", flush=True)
            try:
                model.to(device=device, dtype=torch.float32)
                result = probe_model(model, train_batch, validation_batch, epsilon=plan["protocol"]["loss"]["epsilon"])
            except torch.cuda.OutOfMemoryError as error:
                result = {"status": "out_of_memory", "phase": "model_to_device_or_preservation",
                          "error": str(error), "stages": {}, "optimizer_steps": 0}
        finally:
            restore_rng(rng)
        _, current_data, _, current_plan, current_preflight = bind_preflight(path, args.preflight, args.model)
        if (current_data != data or current_plan != plan or current_preflight != preflight_bytes
                or source_hashes(names) != hashes):
            raise ValueError("Inputs or source changed during the probe; report not published")
        report = {"schema_version": 1, "scope": "bounded_training_resource_probe", "result": result,
                  "model": args.model, "model_settings": plan["model_settings"], "parameters": parameters,
                  "plan_sha256": fingerprint(plan), "protocol_sha256": plan["protocol_sha256"],
                  "split_sha256": plan["split_sha256"], "manifest_sha256": hashlib.sha256(data).hexdigest(),
                  "preflight_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
                  "selection": selected, "runtime": runtime, "source_fingerprints": hashes,
                  "checkpoint_loaded": False, "optimizer_created": False, "training_performed": False,
                  "quality_scores_generated": False, "complete_validation": False, "test_audio_read": False,
                  "execution_ready": False}
        saved = publish_report(output, report, protected)
        for name, stage in result["stages"].items():
            print(name + ":", json.dumps(stage, allow_nan=False))
        print("Report:", saved)
        print("Bounded resource check; no optimizer steps or quality benchmark. Execution ready: false.")
        if result["status"] != "passed":
            parser.exit(1, f"ERROR: probe {result['status']}; inspect the diagnostic report\n")
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
