"""Repeat candidate checkpoint subsets to separate signal from timing noise."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ["0", "2", "0,1", "1,3", "0,1,2", "0,1,4", "0,1,2,3",
              "0,1,2,4", "0,1,3,4", "0,2,3,4", "0,1,2,3,4"]
OUTPUT = ROOT / "results/paper_runs/tfgridnet_candidate_repeats.jsonl"


def main():
    rows = []
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for subset in CANDIDATES:
            for repeat in range(3):
                command = [
                    sys.executable, str(ROOT / "scripts/bench_tfgridnet_policies.py"),
                    "--child", "--stride", "128", "--amp", "fp16",
                    "--checkpoint-blocks", subset, "--policy", "candidate",
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
                    payload = {"status": "crashed", "stderr": completed.stderr[-400:]}
                payload["subset"] = [int(x) for x in subset.split(",")]
                payload["repeat"] = repeat
                rows.append(payload)
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(payload, sort_keys=True), flush=True)
    (ROOT / "results/paper_runs/tfgridnet_candidate_repeats.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
