"""Enumerate all non-empty 6-block checkpoint subsets at 8 ms / FP16."""

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).resolve()
OUTPUT = ROOT / "results/paper_runs/tfgridnet_subset_sweep.jsonl"


def main():
    rows = []
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for size in range(1, 7):
            for subset in itertools.combinations(range(6), size):
                mask = ",".join(str(index) for index in subset)
                command = [
                    sys.executable, str(CHILD), "--child",
                    "--stride", "128", "--amp", "fp16",
                    "--checkpoint-blocks", mask,
                    "--policy", f"fp16_subset_{''.join(map(str, subset))}",
                    "--steps", "2",
                ]
                completed = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True,
                    env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=300,
                )
                payload = None
                for line in reversed(completed.stdout.splitlines()):
                    try:
                        payload = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                if payload is None:
                    payload = {"status": "crashed", "subset": list(subset),
                               "stderr": completed.stderr[-400:]}
                payload["subset"] = list(subset)
                rows.append(payload)
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(payload, sort_keys=True), flush=True)
    (ROOT / "results/paper_runs/tfgridnet_subset_sweep.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def child():
    from bench_tfgridnet_policies import _run_child
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--policy", default="fp16_subset")
    parser.add_argument("--checkpoint-blocks", default="0")
    parser.add_argument("--amp", choices=("none", "fp16"), default="fp16")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, default=Path("unused.json"))
    args = parser.parse_args()
    raw = str(args.checkpoint_blocks)
    args.checkpoint_blocks = [int(part) for part in raw.split(",") if part]
    _run_child(args)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        main()
