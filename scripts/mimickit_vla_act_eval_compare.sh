#!/usr/bin/env bash
# Closed-loop A/B: run the SAME closed-loop eval for the clean-baseline ACT and the DART
# ACT, parse per-episode survival frames from the eval log, and print a go/pivot table.
# This is the gate for the data-side DART experiment: does noise-injected BC actually keep
# the G1 upright longer than pure clean-rollout BC?
#
#   go/pivot:  mean survival 20 -> >=50 frames  => DART helps, proceed to DAgger.
#              <5 frame improvement             => not covariate shift; go architecture-side.
#
# Usage:  scripts/mimickit_vla_act_eval_compare.sh
#   knobs: NUM_EP=5  NACT=1  MAX_STEPS=400
#          BASELINE_CKPT=...  DART_CKPT=...   (default to the two sanity/dart last ckpts)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_EP="${NUM_EP:-5}"
NACT="${NACT:-1}"            # replan every step: most favorable for balance, isolates data effect
MAX_STEPS="${MAX_STEPS:-400}"
BASELINE_CKPT="${BASELINE_CKPT:-$REPO_ROOT/outputs/act_g1_lafan_sanity/checkpoints/last/pretrained_model}"
DART_CKPT="${DART_CKPT:-$REPO_ROOT/outputs/act_g1_lafan_dart/checkpoints/last/pretrained_model}"
LOGDIR="${LOGDIR:-/tmp/dart_eval_compare}"
mkdir -p "$LOGDIR"

run_one () {  # $1=label  $2=ckpt  $3=port
  local label="$1" ckpt="$2" port="$3" log="$LOGDIR/${1}.log"
  echo "================================================"
  echo "[compare] $label  ckpt=$ckpt  NACT=$NACT  NUM_EP=$NUM_EP/motion"
  echo "================================================"
  if [[ ! -d "$ckpt" ]]; then echo "[compare] MISSING ckpt: $ckpt — skip $label"; return 1; fi
  NUM_EP="$NUM_EP" NACT="$NACT" MAX_STEPS="$MAX_STEPS" PORT="$port" \
    bash "$REPO_ROOT/scripts/mimickit_vla_act_closedloop.sh" "$ckpt" 2>&1 | tee "$log"
}

run_one baseline "$BASELINE_CKPT" 5599 || true
run_one dart     "$DART_CKPT"     5601 || true

# ---- parse survival frames + verdict ---------------------------------------
python3 - "$LOGDIR/baseline.log" "$LOGDIR/dart.log" <<'PY'
import re, sys, statistics as st
def parse(path):
    # lines: "[act-eval] ep N: F frames (done=D) -> ..."  — but motions run sequentially;
    # we tag by the ">>> fight" / ">>> dance" banners that precede each motion block.
    motion="?"; out={}
    try: lines=open(path).read().splitlines()
    except FileNotFoundError: return out
    for ln in lines:
        m=re.search(r">>> (\w+)", ln)
        if m: motion=m.group(1); out.setdefault(motion,[])
        m=re.search(r"ep \d+: (\d+) frames \(done=([\-\d\.]+)\)", ln)
        if m: out.setdefault(motion,[]).append(int(m.group(1)))
    return out
b=parse(sys.argv[1]); d=parse(sys.argv[2])
def stats(v): return (round(st.mean(v),1), int(st.median(v)), min(v), max(v)) if v else (0,0,0,0)
print("\n================== DART go/pivot ==================")
print(f"{'motion':8} {'baseline mean/med/min/max':30} {'DART mean/med/min/max':30} {'Δmean':>7}")
motions=sorted(set(list(b)+list(d)))
all_delta=[]
for mo in motions:
    bv=b.get(mo,[]); dv=d.get(mo,[])
    bm,bmd,bmn,bmx=stats(bv); dm,dmd,dmn,dmx=stats(dv)
    delta=round(dm-bm,1); all_delta.append(delta)
    print(f"{mo:8} {f'{bm}/{bmd}/{bmn}/{bmx} (n={len(bv)})':30} {f'{dm}/{dmd}/{dmn}/{dmx} (n={len(dv)})':30} {delta:>7}")
if all_delta:
    avg=round(sum(all_delta)/len(all_delta),1)
    # overall DART mean across motions
    dall=[x for mo in motions for x in d.get(mo,[])]
    dmean=round(st.mean(dall),1) if dall else 0
    print("==================================================")
    print(f"avg Δmean = {avg} frames | DART overall mean = {dmean}")
    if dmean>=50:   print("VERDICT: ✅ GO — DART works (mean≥50). Proceed to DAgger toward 300.")
    elif avg>=5:    print("VERDICT: 🟡 PARTIAL — DART helps but <50. More DART/σ or straight to DAgger.")
    else:           print("VERDICT: 🔴 PIVOT — <5f gain. Not covariate shift; go architecture-side (WBC).")
PY
echo "  logs: $LOGDIR/{baseline,dart}.log"
