"""Plot PESQ against optimizer step for TF-GridNet policy proxy runs.

Usage:
  python scripts/plot_tfgridnet_quality.py \
    --series "whole 0-2 (200-step)=results/.../p3_whole3_200_paired" \
    --series "attention + whole 0-2 (1,000-step)=results/.../p3_whole3_1000" \
    --out results/paper_runs/tfgridnet_quality.svg
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape


def _points(run_dir: Path) -> list[tuple[int, float]]:
    points = []
    for path in sorted(run_dir.glob("validation_step_*.json")):
        match = re.search(r"validation_step_(\d+)\.json$", path.name)
        if not match:
            continue
        step = int(match.group(1))
        payload = json.loads(path.read_text(encoding="utf-8"))
        pesq = payload.get("metrics", {}).get("pesq")
        if pesq is not None:
            points.append((step, float(pesq)))
    return sorted(points)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", required=True, help="label=run_dir")
    parser.add_argument("--out", default="results/paper_runs/tfgridnet_quality.svg")
    args = parser.parse_args()

    series = []
    for item in args.series:
        label, raw = item.split("=", 1)
        points = _points(Path(raw))
        if points:
            series.append((label, points))
    if not series:
        raise SystemExit("no validation_step_*.json with metrics.pesq found")

    width, height = 760, 460
    left, right, top, bottom = 90, 40, 45, 95
    xmax = max(p[0] for _, points in series for p in points)
    ymin = min(p[1] for _, points in series for p in points)
    ymax = max(p[1] for _, points in series for p in points)
    ymin = max(1.0, ymin - 0.05)
    ymax = ymax + 0.05
    colors = ["#2c6fbb", "#c0392b", "#2e8b57", "#8e44ad"]
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x(value: float) -> float:
        return left + (value / xmax) * plot_w

    def y(value: float) -> float:
        return top + (1.0 - (value - ymin) / (ymax - ymin)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-size="16" font-weight="bold">'
        'TF-GridNet 8 ms policy-proxy PESQ trajectory</text>',
    ]
    for tick in range(int(ymin * 10) + 1, int(ymax * 10) + 1, 2):
        value = tick / 10
        parts.append(f'<line x1="{left}" y1="{y(value):.0f}" x2="{width-right}" y2="{y(value):.0f}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{left-10}" y="{y(value)+4:.0f}" text-anchor="end" font-size="11">{value:.1f}</text>')
    for step in range(0, xmax + 1, max(50, xmax // 5)):
        parts.append(f'<line x1="{x(step):.0f}" y1="{top}" x2="{x(step):.0f}" y2="{height-bottom}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{x(step):.0f}" y="{height-bottom+18}" text-anchor="middle" font-size="11">{step}</text>')
    parts += [
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-30}" text-anchor="middle" font-size="12">optimizer step</text>',
        f'<text x="22" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90 22 {height/2})">validation PESQ</text>',
    ]
    for idx, (label, points) in enumerate(series):
        color = colors[idx % len(colors)]
        coords = " ".join(f"{x(step):.1f},{y(value):.1f}" for step, value in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for step, value in points:
            parts.append(f'<circle cx="{x(step):.1f}" cy="{y(value):.1f}" r="3.5" fill="{color}"/>')
        parts.append(f'<text x="{left+8}" y="{top+18+idx*18}" font-size="11" fill="{color}">{escape(label)}</text>')
    parts.append("</svg>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
