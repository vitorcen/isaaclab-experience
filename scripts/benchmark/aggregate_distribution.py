#!/usr/bin/env python3
"""Aggregate a 20-round (or any-round) PickOrange metrics.json into placement
distribution + 5-round sub-sample variance for *strict statistical* leaderboard.

Single 5-round measurement has ±5.5% σ noise (验证于 2026-05-23 ckpt-6000:
14/15 single-run vs 41/60 mean 20-round). Use this script's output as the
authoritative "strict" leaderboard entry that supersedes single 5-round.

Usage:
    python aggregate_distribution.py <metrics.json> [--out <out.md>] [--svg <out.svg>]

Outputs:
  - stdout markdown table {0,1,2,3} histogram + E(oranges) + 5-round sub-sample mean ± std
  - optional --out file (.md) for embedding in README
  - optional --svg file rendering bar chart for HTML docs

Memory: feedback-20round-strict-benchmark
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def load_metrics(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def histogram(per_round: list[dict]) -> dict[int, int]:
    """Count P(placed=k) for k in {0,1,2,3} from per_round metrics."""
    h = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in per_round:
        n = int(r.get("oranges_placed", 0))
        n = max(0, min(3, n))
        h[n] += 1
    return h


def sub_sample_5round_stats(per_round: list[dict]) -> dict:
    """Bin per_round into 5-round windows, compute oranges mean ± std across windows."""
    oranges = [int(r.get("oranges_placed", 0)) for r in per_round]
    n = len(oranges)
    if n < 5:
        return {"n_windows": 0, "mean_oranges_per_5round": 0.0, "std": 0.0, "windows": []}
    # truncate to multiple of 5
    n = (n // 5) * 5
    oranges = oranges[:n]
    windows = []
    for i in range(0, n, 5):
        w = oranges[i:i+5]
        windows.append({"start_ep": i+1, "end_ep": i+5, "oranges": sum(w), "per_ep": w})
    sums = [w["oranges"] for w in windows]
    mean = statistics.mean(sums)
    std = statistics.stdev(sums) if len(sums) >= 2 else 0.0
    return {
        "n_windows": len(windows),
        "mean_oranges_per_5round": mean,
        "std": std,
        "windows": windows,
    }


def env_success_count(per_round: list[dict]) -> tuple[int, int, int]:
    """Returns (env_success, all_3_placed, total). Note: all_3 ≥ env_success because
    env may not fire task_done even when 3/3 placed (model doesn't return to rest pose)."""
    env_ok = sum(1 for r in per_round if r.get("success", False))
    all3 = sum(1 for r in per_round if int(r.get("oranges_placed", 0)) == 3)
    return env_ok, all3, len(per_round)


def format_markdown(meta: dict, hist: dict, stats: dict, env: tuple) -> str:
    label = meta.get("label", "—")
    n = sum(hist.values())
    total_oranges = sum(k * v for k, v in hist.items())
    pct = {k: v / n * 100 if n else 0 for k, v in hist.items()}
    e_oranges = total_oranges / n if n else 0
    env_ok, all3, _ = env

    out = []
    out.append(f"## {label} — strict {n}-round distribution\n")
    out.append("| Placed per episode | Count | P(placed=k) |")
    out.append("|---|---|---|")
    for k in [3, 2, 1, 0]:
        bold = "**" if k == 3 else ""
        out.append(f"| {bold}{k}{bold} | {bold}{hist[k]}{bold} | {bold}{pct[k]:.1f}%{bold} |")
    out.append("")
    out.append(f"- **E(oranges/ep) = {e_oranges:.2f} / 3 = {e_oranges/3*100:.1f}%** "
               f"({total_oranges}/{n*3} oranges placed)")
    out.append(f"- env_success rate (`task_done` fired): **{env_ok}/{n} = {env_ok/n*100:.1f}%**")
    out.append(f"- all-3-placed rate (oranges placed, env may not fire): **{all3}/{n} = {all3/n*100:.1f}%**")
    if all3 > env_ok:
        gap = all3 - env_ok
        out.append(f"  - gap = {gap} ep where 3/3 placed but env didn't fire `task_done` "
                   f"(model didn't return arm to rest pose → wall_cap waste)")
    if stats["n_windows"]:
        out.append(f"- 5-round sub-sample (n={stats['n_windows']} windows): "
                   f"mean = **{stats['mean_oranges_per_5round']:.2f}/15**, "
                   f"σ = **{stats['std']:.2f} ({stats['std']/15*100:.1f}%)**")
        out.append("  - any single 5-round measurement of THIS ckpt expected ±2σ ≈ "
                   f"±{stats['std']*2/15*100:.1f}%")
    out.append("")
    out.append(f"Per-episode raw: `placed_per_ep = {[int(r.get('oranges_placed', 0)) for r in meta.get('per_round', [])]}`")
    return "\n".join(out)


def format_svg(hist: dict, title: str = "") -> str:
    """Simple inline SVG bar chart of P(placed=k)."""
    n = sum(hist.values())
    W, H = 360, 220
    bar_w = 60
    gap = 25
    base_y = 180
    bar_h_max = 140
    bars = []
    for i, k in enumerate([0, 1, 2, 3]):
        v = hist[k]
        h = (v / n) * bar_h_max if n else 0
        x = 40 + i * (bar_w + gap)
        y = base_y - h
        color = ["#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1"][k]
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" stroke="#333"/>'
            f'<text x="{x+bar_w/2}" y="{y-6:.1f}" text-anchor="middle" font-size="13" fill="#1a1a1a">{v} ({v/n*100:.0f}%)</text>'
            f'<text x="{x+bar_w/2}" y="{base_y+18}" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a1a">{k}</text>'
        )
    title_txt = f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a1a">{title}</text>' if title else ""
    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#fafafa;border:1px solid #ddd;border-radius:6px;display:block;margin:8px auto;">'
        f'{title_txt}'
        f'<line x1="40" y1="{base_y}" x2="{W-10}" y2="{base_y}" stroke="#333" stroke-width="1"/>'
        f'<text x="{W/2}" y="{base_y+38}" text-anchor="middle" font-size="13" fill="#555">oranges placed per episode</text>'
        f'{"".join(bars)}'
        f'</svg>'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", help="path to *.metrics.json (from policy_inference.py)")
    ap.add_argument("--out", help="write markdown to this file (default: stdout only)")
    ap.add_argument("--svg", help="write SVG bar chart to this file")
    args = ap.parse_args()

    meta = load_metrics(args.metrics)
    per_round = meta.get("per_round", [])
    if not per_round:
        print(f"ERROR: no per_round in {args.metrics}", file=sys.stderr)
        sys.exit(1)
    if len(per_round) < 5:
        print(f"WARNING: only {len(per_round)} rounds — strict benchmark wants ≥ 20", file=sys.stderr)

    hist = histogram(per_round)
    stats = sub_sample_5round_stats(per_round)
    env = env_success_count(per_round)
    md = format_markdown(meta, hist, stats, env)
    print(md)

    if args.out:
        Path(args.out).write_text(md)
        print(f"\n[write] {args.out}", file=sys.stderr)

    if args.svg:
        label = meta.get("label", "")
        svg = format_svg(hist, title=f"{label} — placements/ep")
        Path(args.svg).write_text(svg)
        print(f"[write] {args.svg}", file=sys.stderr)


if __name__ == "__main__":
    main()
