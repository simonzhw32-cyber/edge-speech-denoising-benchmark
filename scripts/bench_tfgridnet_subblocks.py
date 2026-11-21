"""Benchmark sub-block checkpointing policies against whole-block checkpointing.

Each policy runs in a fresh child process so CUDA state cannot leak between
measurements.  This is a measurement harness, not a training entry point.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


POLICIES = [
    ("fp16_none", 0, ()),
    ("fp16_whole2", 2, ()),
    ("fp16_whole4", 4, ()),
    ("fp16_intra", 0, ("intra",)),
    ("fp16_inter", 0, ("inter",)),
    ("fp16_attention", 0, ("attention",)),
    ("fp16_intra_inter", 0, ("intra", "inter")),
    ("fp16_inter_attention", 0, ("inter", "attention")),
    ("fp16_all_subblocks", 0, ("intra", "inter", "attention")),
]


def run_policy(name: str, blocks: int, flags: tuple[str, ...], args) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--stride", "128",
        "--policy", name,
        "--checkpoint-blocks", str(blocks),
        "--amp", "fp16",
        "--steps", str(args.steps),
        "--seed", str(args.seed),
    ]
    for flag in flags:
        command.append(f"--subblock-{flag}")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=environment,
        timeout=args.timeout,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "policy": name,
        "status": "crashed",
        "returncode": completed.returncode,
        "stderr": completed.stderr[-1200:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true",
                        help="internal: run one policy in this process")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--policy", default="fp16_none")
    parser.add_argument("--checkpoint-blocks", default="0")
    parser.add_argument("--amp", choices=("none", "fp16"), default="fp16")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--subblock-intra", action="store_true")
    parser.add_argument("--subblock-inter", action="store_true")
    parser.add_argument("--subblock-attention", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("results/paper_runs/tfgridnet_subblock_bench.json"))
    args = parser.parse_args()
    if args.child:
        raw = str(args.checkpoint_blocks).strip()
        args.checkpoint_blocks = (
            [int(part) for part in raw.split(",") if part.strip()]
            if "," in raw else int(raw)
        )
        # Delegate to the single-policy child implementation.
        from bench_tfgridnet_policies import _run_child  # type: ignore
        _run_child(args)
        return

    rows = []
    for name, blocks, flags in POLICIES:
        row = run_policy(name, blocks, flags, args)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
