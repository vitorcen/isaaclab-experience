#!/usr/bin/env python3
"""Build STRICT_LEADERBOARD.md from all *.metrics.json in results_dir.
Sorts by E(oranges) DESC, ties broken by env_success DESC then σ ASC."""
from __future__ import annotations
import argparse, glob, json, math, statistics
from pathlib import Path


def metrics_stats(metrics_path: str) -> dict | None:
    try:
        m = json.load(open(metrics_path))
    except Exception as e:
        return None
    per = m.get("per_round", [])
    if len(per) < 5:
        return None
    n = len(per)
    placed = [int(r.get("oranges_placed", 0)) for r in per]
    env_ok = sum(1 for r in per if r.get("success", False))
    all3 = sum(1 for p in placed if p == 3)
    e_o = sum(placed) / n
    # 5-round sub-sample σ
    n5 = (n // 5) * 5
    if n5 >= 10:
        sums = [sum(placed[i:i+5]) for i in range(0, n5, 5)]
        sigma_5 = statistics.stdev(sums) if len(sums) >= 2 else 0
    else:
        sigma_5 = 0
    # per-episode distribution P(placed=k) k∈{0,1,2,3}
    h = {0: 0, 1: 0, 2: 0, 3: 0}
    for p in placed:
        h[max(0, min(3, p))] += 1
    p_dist = {k: v / n * 100 for k, v in h.items()}
    p_geq2 = p_dist[2] + p_dist[3]   # at least 2 oranges placed
    # worst-case mean (1σ lower bound on 5-round)
    worst_5round = (sum(placed) / n * 5) - sigma_5   # mean per-5round - 1σ
    return {
        "label": m.get("label", Path(metrics_path).stem),
        "slug": Path(metrics_path).stem.replace(".metrics", ""),
        "rounds": n,
        "oranges_total": sum(placed),
        "oranges_max": n * 3,
        "e_oranges": e_o,
        "pct_oranges": e_o / 3 * 100,
        "env_success": env_ok,
        "env_pct": env_ok / n * 100,
        "all3_count": all3,
        "all3_pct": all3 / n * 100,
        "sigma_5round": sigma_5,
        "sigma_pct": sigma_5 / 15 * 100,
        "p_dist": p_dist,
        "p_geq2": p_geq2,
        "worst_5round": worst_5round,
        "per_round_oranges": placed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/benchmark",
                    help="dir with *.metrics.json (per-baseline outputs)")
    ap.add_argument("--out", default="scripts/benchmark/STRICT_LEADERBOARD.md",
                    help="output leaderboard markdown (default: under scripts/benchmark/)")
    ap.add_argument("--min_rounds", type=int, default=20,
                    help="ignore metrics with fewer rounds (default: 20 — strict standard)")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(f"{args.results_dir}/*.metrics.json")):
        s = metrics_stats(p)
        if s and s["rounds"] >= args.min_rounds:
            rows.append(s)

    if not rows:
        print(f"no metrics.json in {args.results_dir}")
        return

    # Sort: E(oranges)/ep DESC — straight mean, the metric users actually care about.
    # Tie breakers: P(3) DESC (full success), env_success DESC, σ ASC.
    rows.sort(key=lambda r: (-r["e_oranges"], -r["p_dist"][3], -r["env_success"], r["sigma_5round"]))

    md = []
    md.append("# Strict ≥20-round Leaderboard — PickOrange Probability Distribution\n")
    md.append("自动生成。**排名规则**: E(🍊)/ep DESC → P(3) DESC → env_success DESC → σ ASC.\n")
    md.append("E(🍊)/ep = total_oranges / N_episodes，即每 episode 平均放置橙子数（满分 3）。\n")
    md.append("默认最低 20-round 入榜（strict 标准）；σ(5-round) 跨 5-round sub-sample 计算。See `feedback-20round-strict-benchmark` memory.\n")
    md.append("")
    md.append("## 主表 — Main leaderboard")
    md.append("")
    md.append("| Rank | Model | N | **🔑⬇️ E(🍊)/ep** | σ(5-rd) | Worst-case (mean−1σ)/15 | P(3) | P(≥2) | env_success |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else ""))
        sigma_str = f"{r['sigma_5round']:.2f} ({r['sigma_pct']:.1f}%)" if r["rounds"] >= 10 else "—"
        worst_str = f"{r['worst_5round']:.2f}" if r["rounds"] >= 10 else "—"
        md.append(
            f"| {i}{medal} | {r['label']} | {r['rounds']} | "
            f"**{r['pct_oranges']:.1f}%** ({r['e_oranges']:.2f}/3) | "
            f"{sigma_str} | "
            f"{worst_str} | "
            f"{r['p_dist'][3]:.0f}% | "
            f"{r['p_geq2']:.0f}% | "
            f"{r['env_pct']:.1f}% ({r['env_success']}/{r['rounds']}) |"
        )
    md.append("")
    md.append("## 分布表 — P(placed=k) per model")
    md.append("")
    md.append("| Model | N | P(0) | P(1) | P(2) | P(3) |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        d = r["p_dist"]
        md.append(
            f"| {r['label']} | {r['rounds']} | {d[0]:.0f}% | {d[1]:.0f}% | {d[2]:.0f}% | **{d[3]:.0f}%** |"
        )
    md.append("")
    md.append("## Per-episode raw oranges")
    for r in rows:
        md.append(f"- `{r['slug']}` ({r['rounds']} eps): `{r['per_round_oranges']}`")
    md.append("")
    md.append("---")
    md.append("**指标说明**:")
    md.append("- **🔑⬇️ E(🍊)/ep**: 每 episode 期望放置橙子数（满分 3）— **主排序键**")
    md.append("- **σ(5-rd)**: 跨 5-round sub-sample 的标准差（用于估单次 5-round noise）")
    md.append("- **Worst-case (mean−1σ)/15**: 参考指标 — ~68% 概率任意一次 5-round 不低于此值；衡量 reliability")
    md.append("- **P(3)**: 单 episode 全 3 颗成功的概率")
    md.append("- **P(≥2)**: 单 episode 至少 2 颗成功的概率 (useful threshold)")
    md.append("- **env_success**: 环境 `task_done` fire 的 episode 比例（含 arm-rest 要求）— 通常 ≤ P(3) 因为 placement 后 arm 没收回会卡 wall_cap")

    Path(args.out).write_text("\n".join(md))
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
