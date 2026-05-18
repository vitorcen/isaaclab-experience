#!/usr/bin/env python3
"""Aggregate per-baseline JSON + GPU CSV samples into a single markdown table.

Usage:
    python aggregate.py <results_dir> [--out results.md]

Looks for files <slug>.metrics.json and <slug>.gpu.csv under results_dir/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_gpu_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    mem = []
    util = []
    total_mib = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mem.append(int(row["mem_used_mib"]))
                util.append(int(row["util_gpu_pct"]))
                if total_mib is None:
                    total_mib = int(row["mem_total_mib"])
            except (KeyError, ValueError):
                continue
    if not mem:
        return None
    return {
        "peak_mib": max(mem),
        "mean_mib": sum(mem) // len(mem),
        "mean_util_pct": round(sum(util) / len(util), 1),
        "peak_util_pct": max(util),
        "total_mib": total_mib,
        "n_samples": len(mem),
    }


def fmt_mib(m: int | None) -> str:
    if m is None:
        return "—"
    if m >= 1024:
        return f"{m / 1024:.1f} GB"
    return f"{m} MiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None, help="Output markdown path (default stdout).")
    ap.add_argument("--baselines_tsv", default=None,
                    help="Optional ordered baseline list (slug → label).")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    if not rdir.is_dir():
        sys.exit(f"results dir not found: {rdir}")

    # Load ordering hint
    ordered: list[tuple[str, str]] = []
    if args.baselines_tsv and Path(args.baselines_tsv).exists():
        with open(args.baselines_tsv) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 6:
                    ordered.append((parts[0], parts[5]))

    rows = []
    if ordered:
        slugs = [s for s, _ in ordered]
    else:
        slugs = sorted({p.stem.replace(".metrics", "") for p in rdir.glob("*.metrics.json")})

    for slug in slugs:
        mp = rdir / f"{slug}.metrics.json"
        gp = rdir / f"{slug}.gpu.csv"
        if not mp.exists():
            rows.append({"slug": slug, "missing": True})
            continue
        with open(mp) as f:
            m = json.load(f)
        g = load_gpu_stats(gp)
        # Strict success = all 3 sticky True (excludes env.task_done false-positives
        # where an orange knocked off plate edge can still satisfy height_range
        # check + arm rest, but sticky put_orange_to_plate would not have fired
        # because EE-near + gripper-open isn't satisfied at the moment of release).
        strict_success = sum(1 for r in m.get("per_round", []) if all(r.get("placed_flags", [])))
        # Per-round detail using strict success
        detail = " / ".join(
            f"{r['oranges_placed']}🍊@{r['duration_s']:.0f}s"
            + ("✅" if all(r.get("placed_flags", [])) else "")
            + ("⚠️env-only" if r.get("env_success_sticky_mismatch") else "")
            for r in m.get("per_round", [])
        )
        rows.append({
            "slug": slug,
            "label": m.get("label") or slug,
            "rounds_strict": f"{strict_success}/{m['rounds']}",
            "rounds_env": f"{m['rounds_success']}/{m['rounds']}",
            "oranges": f"{m['oranges_placed_total']}/{m['oranges_max_total']}",
            "orange_rate_pct": round(100.0 * m['oranges_placed_total'] / m['oranges_max_total'], 1),
            "avg_round_s": m["avg_round_s"],
            "peak_vram": fmt_mib(g["peak_mib"]) if g else "—",
            "mean_util_pct": (str(g["mean_util_pct"]) + "%") if g else "—",
            "detail": detail or "—",
            "missing": False,
        })

    # Render markdown
    lines = []
    lines.append("| Policy | Rounds ✅ (strict) | Rounds (env) | 🍊 (n/9) | Pick rate | Avg round time | Peak VRAM | Mean GPU util | Per-round detail |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        if r.get("missing"):
            lines.append(f"| `{r['slug']}` | ❌ no metrics | — | — | — | — | — | — | — |")
            continue
        env_note = f"({r['rounds_env']})" if r['rounds_env'] != r['rounds_strict'] else r['rounds_env']
        lines.append(
            f"| {r['label']} | {r['rounds_strict']} | {env_note} | {r['oranges']} | {r['orange_rate_pct']}% | "
            f"{r['avg_round_s']}s | {r['peak_vram']} | {r['mean_util_pct']} | {r['detail']} |"
        )

    lines.append("")
    lines.append("**Success criteria**:")
    lines.append("- **Rounds ✅ (strict)** = all 3 sticky `put_orange_to_plate` fired during round "
                 "(EE-near + gripper-open + xy-in-plate at any point). Lower bound — misses fast "
                 "transients (<33ms).")
    lines.append("- **Rounds (env)** = `task_done` (orange xyz in plate box + arm rest). "
                 "Can false-positive when an orange is knocked off plate edge to nearby table at similar height.")
    lines.append("- **🍊 (n/9)** = sticky-counted oranges across all 3 rounds.")
    lines.append("- ⚠️ `env-only` tag = env said success but sticky undercounted (mismatch case).")

    md = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(md)
        print(f"[aggregate] wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
