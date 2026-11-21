"""Render the TF-GridNet 8 ms Pareto table as a dependency-free SVG."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/paper_runs/tfgridnet_pareto_8ms_fp16.json"
OUTPUT = ROOT / "results/paper_runs/tfgridnet_pareto_8ms_fp16.svg"


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r.get("plot", True)]
    width, height = 860, 560
    left, right, top, bottom = 95, 40, 45, 120
    xmin, xmax = 2.2, 11.6
    ymin, ymax = 3.4, 7.6
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-size="17" '
        'font-weight="600">TF-GridNet 8 ms, FP16, batch 1: training-memory Pareto</text>',
    ]

    # Axes and grid.
    for value in range(3, 12):
        x = sx(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" '
                     'stroke="#e6e6e6"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" '
                     f'font-size="12" fill="#444">{value}</text>')
    for value in range(4, 8):
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
                     'stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
                     f'font-size="12" fill="#444">{value}</text>')
    parts += [
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" '
        'stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        'stroke="#333" stroke-width="1.5"/>',
        f'<text x="{width/2}" y="{height-66}" text-anchor="middle" font-size="13">'
        'peak allocated memory (GiB)</text>',
        f'<text x="22" y="{height/2}" text-anchor="middle" font-size="13" '
        f'transform="rotate(-90 22 {height/2})">step time (s)</text>',
    ]

    # Non-dominated frontier in memory order.
    ordered = sorted(rows, key=lambda row: (row["peak_alloc_gib"], row["step_seconds"]))
    frontier = []
    best = float("inf")
    for row in ordered:
        if row["step_seconds"] < best:
            frontier.append(row)
            best = row["step_seconds"]
    points = " ".join(f'{sx(r["peak_alloc_gib"]):.1f},{sy(r["step_seconds"]):.1f}'
                      for r in frontier)
    parts.append(f'<polyline points="{points}" fill="none" stroke="#9aa7b4" '
                 'stroke-width="1.5" stroke-dasharray="4 3"/>')

    label_offsets = {
        "none": (8, -10),
        "attention": (8, 16),
        "attn_whole0": (8, 16),
        "attn_whole2": (-10, 14),
        "attn_whole3": (10, -8),
        "all_subblocks": (8, 16),
    }
    for row in rows:
        hybrid = bool(row["subblocks"])
        color = "#c0392b" if hybrid else "#2c6fbb"
        if row["policy"] == "none":
            color = "#111111"
        radius = 5 if hybrid else 4
        x, y = sx(row["peak_alloc_gib"]), sy(row["step_seconds"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
                     'stroke="white" stroke-width="1"/>')
        if row["policy"] in label_offsets:
            dx, dy = label_offsets[row["policy"]]
            label = row["policy"].replace("fp16_", "").replace("_", " ")
            parts.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" font-size="11" '
                         f'fill="{color}">{label}</text>')

    parts += [
        '<circle cx="110" cy="535" r="4" fill="#2c6fbb"/>',
        '<text x="122" y="539" font-size="12">uniform whole-block</text>',
        '<circle cx="300" cy="535" r="4" fill="#c0392b"/>',
        '<text x="312" y="539" font-size="12">cheap-first hybrid</text>',
        '<circle cx="490" cy="535" r="4" fill="#111111"/>',
        '<text x="502" y="539" font-size="12">no checkpoint</text>',
        '</svg>',
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
