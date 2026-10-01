"""Download pinned parquet splits and write verified paired audio manifests."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from speech_denoising.datasets.voicebank import rows_sha256, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def download(url, target, expected_hash, expected_size):
    if target.exists() and sha256_file(target) == expected_hash:
        print("Already verified:", target, flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    received = 0
    with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > expected_size:
                raise ValueError("Download exceeds pinned file size")
            out.write(chunk)
            if received % (8 * 1024 * 1024) < 1024 * 1024:
                print(f"Downloaded {received // (1024*1024)} / {expected_size // (1024*1024)} MiB", flush=True)
    if received != expected_size or sha256_file(temp) != expected_hash:
        raise ValueError("Parquet size/SHA-256 mismatch")
    temp.replace(target)
    return target


def prepare(source, split, root, local_parquet=None):
    selected = [x for x in source["files"] if x["path"].startswith(f"data/{split}-")]
    if local_parquet is not None and len(selected) != 1:
        raise ValueError("--parquet currently supports the single test shard only")
    rows = []
    ids = set()
    for shard in selected:
        if local_parquet is None:
            url = f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}/{shard['path']}"
            path = download(url, root / "parquet" / Path(shard["path"]).name, shard["sha256"], shard["size"])
        else:
            path = Path(local_parquet)
            if path.stat().st_size != shard["size"] or sha256_file(path) != shard["sha256"]:
                raise ValueError("Local parquet does not match the pinned source")
        parquet = pq.ParquetFile(path)
        if set(parquet.schema_arrow.names) != {"id", "clean", "noisy"}:
            raise ValueError("Unexpected parquet columns")
        for batch in parquet.iter_batches(batch_size=16):
            for item in batch.to_pylist():
                identifier = item["id"]
                if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in ids:
                    raise ValueError("Invalid or duplicate utterance ID")
                ids.add(identifier)
                row = {"utterance_id": identifier}
                lengths = []
                for role in ("noisy", "clean"):
                    cell = item[role]
                    content = cell["bytes"]
                    if not isinstance(content, bytes) or not content:
                        raise ValueError("Expected embedded audio bytes")
                    if cell.get("path") and Path(cell["path"]).stem != identifier:
                        raise ValueError("Audio filename does not match utterance ID")
                    signal, sr = sf.read(io.BytesIO(content), dtype="float32")
                    if sr != 16000 or signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
                        raise ValueError("Expected finite mono 16 kHz audio")
                    info = sf.info(io.BytesIO(content))
                    if info.format != "WAV":
                        raise ValueError("Pinned source unexpectedly contains non-WAV audio")
                    relative = f"audio/{split}/{role}/{identifier}.wav"
                    output = root / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(content)  # Preserve original encoded bytes; no resampling or quantization.
                    row[f"{role}_path"] = relative
                    row[f"{role}_sha256"] = hashlib.sha256(content).hexdigest()
                    lengths.append(len(signal))
                if lengths[0] != lengths[1]:
                    raise ValueError("Noisy and clean audio lengths differ")
                row["samples"] = lengths[0]
                rows.append(row)
    rows.sort(key=lambda r: r["utterance_id"])
    if len(rows) != source["expected_counts"][split]:
        raise ValueError("Split count differs from pinned protocol")
    metadata = {"schema_version": 1, "split": split, "sample_rate": 16000,
                "protocol_complete": True, "source": source, "source_files": selected,
                "count": len(rows), "rows_sha256": rows_sha256(rows), "utterances": rows}
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{split}_manifest.json"
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"PASS: {len(rows)} paired {split} utterances verified at 16 kHz", flush=True)
    print("Manifest:", output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "configs/voicebank.json")
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data/voicebank-demand-16k")
    parser.add_argument("--parquet", type=Path, help="Use a manually downloaded pinned test parquet")
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    for split in (["train", "test"] if args.split == "all" else [args.split]):
        prepare(source, split, args.root.resolve(), args.parquet)
    if args.split == "all":
        manifests = [json.loads((args.root / f"{s}_manifest.json").read_text()) for s in ["train", "test"]]
        if {r["utterance_id"] for r in manifests[0]["utterances"]} & {r["utterance_id"] for r in manifests[1]["utterances"]}:
            raise ValueError("Train/test IDs overlap")
    print("Test is fixed; no validation subset or training has been performed.")


if __name__ == "__main__":
    main()
