#!/usr/bin/env python3
"""Merge episodes from an original strict eval + its re-test runs, dropping serve-hang
episodes, and emit a metrics.json containing the first N *valid* episodes with
recomputed aggregates. Used by the strict-eval retest loop so serve hangs (fake zeros)
never enter the final leaderboard score.

Each --pair is `metrics.json:eval_stdout.log`. Episodes are taken in run order
(original first, then retest passes), keeping only non-hang episodes (see
flag_serve_hang.py for the discriminator), until N are collected. Aggregates
(oranges_placed_total, rounds_success, oranges_max_total, avg_round_s) are recomputed
over exactly those N episodes, which are renumbered 1..N.
"""
import argparse, json, sys
from flag_serve_hang import count_noactions_per_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="metrics.json:log pairs in run order")
    ap.add_argument("--target", type=int, required=True, help="N valid episodes wanted")
    ap.add_argument("--threshold", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = None
    valid = []
    for pair in args.pairs:
        mpath, lpath = pair.rsplit(":", 1)
        m = json.load(open(mpath))
        if base is None:
            base = m
        per_ep_max = (m.get("oranges_max_total", 0) // max(len(m.get("per_round", [])), 1)) or 3
        noact = count_noactions_per_episode(lpath)
        for r in m.get("per_round", []):
            ep = r.get("episode")
            na = noact.get(ep, 0)
            if na >= args.threshold and r.get("oranges_placed", 0) == 0:
                continue  # serve hang -> drop
            valid.append((r, per_ep_max))
            if len(valid) >= args.target:
                break
        if len(valid) >= args.target:
            break

    n = len(valid)
    kept = []
    placed_total = succ = max_total = dur_sum = 0
    for i, (r, pem) in enumerate(valid, start=1):
        rr = dict(r); rr["episode"] = i
        kept.append(rr)
        placed_total += rr.get("oranges_placed", 0)
        succ += 1 if rr.get("success") else 0
        max_total += pem
        dur_sum += rr.get("duration_s", 0)

    out = dict(base)
    out["per_round"] = kept
    out["rounds"] = n
    out["rounds_success"] = succ
    out["oranges_placed_total"] = placed_total
    out["oranges_max_total"] = max_total
    out["avg_round_s"] = round(dur_sum / n, 2) if n else 0
    out["retest_merged"] = True
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[merge] {n} valid episodes -> {args.out}: oranges {placed_total}/{max_total}, "
          f"rounds_success {succ}/{n}")
    if n < args.target:
        print(f"[merge] WARNING: only {n}/{args.target} valid episodes collected "
              f"(retests exhausted) — score uses {n}.")


if __name__ == "__main__":
    main()
