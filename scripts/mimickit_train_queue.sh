#!/usr/bin/env bash
# Sequentially train multiple MimicKit DeepMimic G1 policies — one motion at a time.
# Usage:  scripts/mimickit_train_queue.sh [max_iters_per_motion]
#
# Trains all 4 LAFAN 15s slices in series: fight_15s → dance_15s → jumps_15s → run_15s.
# Each motion stops at max_iters (default 1500) and the queue moves on.
#
# Output:
#   dependencies/MimicKit/output/train_<motion>/
#   dependencies/MimicKit/output/queue_summary.log  — one line per finished motion
set -euo pipefail

MAX_ITERS="${1:-1500}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMMARY="$REPO_ROOT/dependencies/MimicKit/output/queue_summary.log"
mkdir -p "$(dirname "$SUMMARY")"

# Order chosen by 难度: fight first (validated) → run (cyclical, easy) → dance (smooth) → jumps (hard, in-air phases)
QUEUE=(
  lafan_fight_15s
  lafan_run_15s
  lafan_dance_15s
  lafan_jumps_15s
)

echo "================================================" | tee -a "$SUMMARY"
echo "[queue] started $(date -Iseconds)  max_iters_per_motion=$MAX_ITERS" | tee -a "$SUMMARY"

for motion in "${QUEUE[@]}"; do
  echo "------------------------------------------------" | tee -a "$SUMMARY"
  echo "[queue] >>> $motion  $(date -Iseconds)" | tee -a "$SUMMARY"
  if bash "$REPO_ROOT/scripts/mimickit_train_one.sh" "$motion" "$MAX_ITERS"; then
    LATEST=$(ls -t "$REPO_ROOT/dependencies/MimicKit/output/train_${motion}/int_models/"model_*.pt 2>/dev/null | head -1 || echo "")
    echo "[queue] <<< $motion done $(date -Iseconds)  latest=$LATEST" | tee -a "$SUMMARY"
  else
    echo "[queue] !!! $motion FAILED — see output/train_${motion}/run.log" | tee -a "$SUMMARY"
  fi
done

echo "================================================" | tee -a "$SUMMARY"
echo "[queue] all done $(date -Iseconds)" | tee -a "$SUMMARY"
