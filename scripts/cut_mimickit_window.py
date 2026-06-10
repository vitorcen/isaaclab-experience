#!/usr/bin/env python3
"""Cut a [start_s, end_s] window out of a MimicKit G1 motion pkl -> a new MimicKit pkl,
for KINEMATIC preview (mocap replay, never falls) of a sub-action before training.

MimicKit pkl = {loop_mode, fps, frames:[(35,) rows]} (frames at `fps`). Frame index = second*fps.

  python scripts/cut_mimickit_window.py --clip dance --start_s 8 --end_s 12 --out_name _lafan_window
  # -> dependencies/MimicKit/data/motions/g1/_lafan_window.pkl  (preview: mimickit_preview.sh view _lafan_window)
"""
import argparse
import os
import joblib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G1DIR = os.path.join(REPO, "dependencies", "MimicKit", "data", "motions", "g1")
CLIP2PKL = {"fight": "lafan_fight1.pkl", "run": "lafan_run1.pkl",
            "dance": "lafan_dance1.pkl", "jumps": "lafan_jumps1.pkl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", choices=list(CLIP2PKL), help="fight|run|dance|jumps")
    ap.add_argument("--input", help="explicit MimicKit pkl path (overrides --clip)")
    ap.add_argument("--start_s", type=float, required=True)
    ap.add_argument("--end_s", type=float, required=True)
    ap.add_argument("--out_name", default="_lafan_window", help="output basename in MimicKit g1 motions dir")
    a = ap.parse_args()

    src = a.input or (os.path.join(G1DIR, CLIP2PKL[a.clip]) if a.clip else None)
    if not src:
        raise SystemExit("pass --clip or --input")
    d = joblib.load(src)
    fps = int(d["fps"])
    frames = d["frames"]
    n = len(frames)
    i0 = max(0, int(round(a.start_s * fps)))
    i1 = min(n, int(round(a.end_s * fps)))
    if i1 <= i0:
        raise SystemExit(f"empty window: [{a.start_s}s,{a.end_s}s] -> frames [{i0},{i1}) of {n}")
    out = {"loop_mode": d.get("loop_mode", 1), "fps": fps, "frames": frames[i0:i1]}
    out_path = os.path.join(G1DIR, f"{a.out_name}.pkl")
    joblib.dump(out, out_path)
    print(f"[cut] {os.path.basename(src)} [{a.start_s}s,{a.end_s}s] = frames [{i0},{i1}) ({(i1-i0)/fps:.2f}s) "
          f"-> {out_path}\n      preview: bash scripts/mimickit_preview.sh view {a.out_name}")


if __name__ == "__main__":
    main()
