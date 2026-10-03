"""Describe a training draft or write a metadata-only train/validation plan."""

import argparse
import hashlib
import json
from pathlib import Path

from speech_denoising.training.protocol import build_training_plan, fingerprint, validate_protocol
from speech_denoising.utils.utils import REPO_ROOT, load_config


def model_settings(protocol, model):
    relative = protocol["models"][model]
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError("Model config must be inside the repository")
    config = load_config(path)
    entry = config["model"]
    if isinstance(entry, dict):
        if entry.get("name") != model or entry.get("checkpoint") is not None:
            raise ValueError("Model name mismatch or pretrained initialization requested")
        parameters = entry.get("parameters", {})
        profile = "native_default"
    else:
        if model != "tfgridnet" or entry != model or config.get("profile") is not None:
            raise ValueError("Use the original six-layer TF-GridNet config, not the DNS profile")
        if config.get("sample_rate") != 16000 or config["protocol"].get("weights") != "random":
            raise ValueError("Expected a random-weight 16 kHz architecture profile")
        parameters = config["model_params"]
        if parameters.get("n_layers") != 6:
            raise ValueError("This draft pins the local six-layer TF-GridNet profile")
        profile = "local_6layer"
    if not isinstance(parameters, dict):
        raise ValueError("Model constructor settings must be a mapping")
    return {
        "config": relative, "profile": profile, "initialization": "random",
        "constructor_kwargs": parameters,
        "config_sha256": hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training_protocol.yaml"))
    parser.add_argument("--model", required=True, choices=["gtcrn", "lisennet", "tfgridnet"])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true", help="Print the draft without reading data")
    mode.add_argument("--manifest", type=Path, help="Existing complete train manifest; no download")
    parser.add_argument("--output", type=Path, help="New JSON plan file; existing files are never overwritten")
    args = parser.parse_args()
    try:
        protocol = load_config(args.config)
        validate_protocol(protocol)
        settings = model_settings(protocol, args.model)
        if args.describe:
            if args.output:
                parser.error("--output is only used with --manifest")
            print(json.dumps({"status": "draft", "execution_ready": False,
                              "model": args.model, "model_settings": settings,
                              "protocol": protocol, "protocol_sha256": fingerprint(protocol)},
                             indent=2, allow_nan=False))
            return
        if not args.output:
            parser.error("--manifest requires --output for the metadata plan")
        output = args.output.resolve()
        results = (REPO_ROOT / "results").resolve()
        if not output.is_relative_to(results) or output.suffix.lower() != ".json":
            parser.error("Write a new .json under results/; source and historical reports are protected")
        if output == args.manifest.resolve():
            parser.error("Output must differ from the input manifest")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        source = json.loads((REPO_ROOT / "configs/voicebank.json").read_text(encoding="utf-8"))
        plan = build_training_plan(manifest, source, protocol, args.model, settings)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        print("Draft metadata plan:", output)
        print("Counts:", plan["counts"])
        print("Validation speakers:", plan["split"]["validation_speakers"])
        print("Split SHA-256:", plan["split_sha256"])
        print("Audio unverified; no model execution or training.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
