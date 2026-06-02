#!/usr/bin/env bash
# Orchestrator: wait current fight_5s_resume training to reach RESUME_TARGET_ITER
# (or process death), SIGTERM it, then kick the 4 × 15s queue.
#
# Usage:  scripts/mimickit_train_pipeline.sh
#
# Env overrides:
#   RESUME_TARGET_ITER  (default 2500)
#   QUEUE_MAX_ITERS     (default 1500)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_DIR="$REPO_ROOT/dependencies/MimicKit"
RESUME_OUT="$MIMICKIT_DIR/output/train_lafan_fight_5s_resume"
RESUME_TARGET_ITER="${RESUME_TARGET_ITER:-2500}"
QUEUE_MAX_ITERS="${QUEUE_MAX_ITERS:-1500}"
SUMMARY="$MIMICKIT_DIR/output/pipeline_summary.log"

echo "[pipeline] started $(date -Iseconds)  resume_target=$RESUME_TARGET_ITER  queue_max=$QUEUE_MAX_ITERS" | tee -a "$SUMMARY"

# Phase 1: wait for 5s resume to hit target iter or die
while true; do
  sleep 30
  if ! pgrep -f "output/train_lafan_fight_5s_resume" > /dev/null; then
    echo "[pipeline] phase1: 5s resume process gone — proceeding" | tee -a "$SUMMARY"
    break
  fi
  CUR=$(grep -aoE "Iteration \|\s+[0-9]+" "$RESUME_OUT/run.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$" || echo 0)
  if [[ "${CUR:-0}" -ge "$RESUME_TARGET_ITER" ]]; then
    echo "[pipeline] phase1: 5s resume hit iter $CUR >= $RESUME_TARGET_ITER — SIGTERM" | tee -a "$SUMMARY"
    pkill -TERM -f "output/train_lafan_fight_5s_resume" 2>/dev/null || true
    # graceful then force
    for _ in $(seq 1 12); do
      sleep 5
      pgrep -f "output/train_lafan_fight_5s_resume" > /dev/null || break
    done
    pkill -9 -f "output/train_lafan_fight_5s_resume" 2>/dev/null || true
    break
  fi
done

# Brief GPU recovery
sleep 15

# Phase 2: kick 4 × 15s queue
echo "[pipeline] phase2: kicking queue.sh max_iters=$QUEUE_MAX_ITERS" | tee -a "$SUMMARY"
bash "$REPO_ROOT/scripts/mimickit_train_queue.sh" "$QUEUE_MAX_ITERS" 2>&1 | tee -a "$SUMMARY"

echo "[pipeline] all done $(date -Iseconds)" | tee -a "$SUMMARY"
