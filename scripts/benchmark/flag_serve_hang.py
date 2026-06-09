#!/usr/bin/env python3
"""Flag serve-hang episodes in a strict eval so they can be re-tested instead of
counted as fake zeros.

Two kinds of 0-orange episode look identical in the metrics (skipped/wall_cap,
placed=[F,F,F]):
  - REAL   : the client received actions throughout, the arm moved, but the policy
             failed to place — a genuine sample of the policy's ability. KEEP.
  - HANG   : the policy server dropped / stalled (websocket ConnectionClosedError,
             OOM, teardown hang), so the client got `no actions after N retries;
             reusing last action` for most of the episode → the arm froze →
             fake zero. INVALID — should be re-tested, not scored.

Discriminator (client-side, no cross-log timestamp matching needed): per-episode
count of `no actions` retry lines. Warm-up (n_obs_steps>1 needs a few obs before
the first chunk) is ~5-10; a real hang stalls for tens of seconds → tens-to-hundreds.
Default threshold 30 cleanly separates the two (tune with --threshold).

Usage:
  flag_serve_hang.py --log <eval_stdout.log> --metrics <metrics.json> [--threshold 30]
Prints a per-episode table and a JSON summary line `SERVE_HANG_SUMMARY={...}` with
  n_total, n_hang, n_valid, hang_episodes (1-based), clean_oranges, clean_max.
Exit code = number of hang episodes (0 = all valid).
"""
import argparse, json, re, sys

EP_RE = re.compile(r"Evaluating episode\s+(\d+)")
NOACT_RE = re.compile(r"no actions after .* retries")


def count_noactions_per_episode(log_path):
    """Return {episode_index(1-based): no_actions_count} by splitting the log on
    'Evaluating episode N' markers and counting 'no actions' lines until the next."""
    counts = {}
    cur = None
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                m = EP_RE.search(line)
                if m:
                    cur = int(m.group(1))
                    counts.setdefault(cur, 0)
                elif cur is not None and NOACT_RE.search(line):
                    counts[cur] += 1
    except FileNotFoundError:
        pass
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--threshold", type=int, default=30,
                    help="no-actions count above which an episode is judged a serve hang")
    args = ap.parse_args()

    m = json.load(open(args.metrics))
    per_round = m.get("per_round", [])
    noact = count_noactions_per_episode(args.log)

    rows, hang_eps, clean_oranges = [], [], 0
    per_ep_max = (m.get("oranges_max_total", 0) // max(len(per_round), 1)) or 3
    for i, r in enumerate(per_round, start=1):
        ep = r.get("episode", i)
        na = noact.get(ep, 0)
        placed = r.get("oranges_placed", 0)
        dur = r.get("duration_s", 0)
        # a hang only matters when the episode also failed: a hang that still placed
        # oranges is not corrupting the score downward, so only flag failed+stalled.
        hang = na >= args.threshold and placed == 0
        if hang:
            hang_eps.append(ep)
        else:
            clean_oranges += placed
        rows.append((ep, placed, dur, na, "HANG" if hang else "ok"))

    n_total = len(per_round)
    n_hang = len(hang_eps)
    n_valid = n_total - n_hang
    clean_max = n_valid * per_ep_max

    print(f"{'ep':>3} {'placed':>6} {'dur_s':>7} {'no_act':>7}  verdict")
    for ep, placed, dur, na, v in rows:
        print(f"{ep:>3} {placed:>6} {dur:>7.1f} {na:>7}  {v}")
    summary = dict(n_total=n_total, n_hang=n_hang, n_valid=n_valid,
                   hang_episodes=hang_eps, clean_oranges=clean_oranges, clean_max=clean_max,
                   threshold=args.threshold)
    print("SERVE_HANG_SUMMARY=" + json.dumps(summary))
    if n_hang:
        print(f"\n[!] {n_hang}/{n_total} episode(s) were serve-hangs (fake zeros): "
              f"{hang_eps} — re-test these for a true score.")
    else:
        print(f"\n[ok] all {n_total} episodes valid (no serve hangs); score is real.")
    sys.exit(n_hang)


if __name__ == "__main__":
    main()
