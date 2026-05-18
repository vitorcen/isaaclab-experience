#!/usr/bin/env bash
# Run all 7 LeIsaac PickOrange baselines back-to-back and aggregate results.
#
# Inventory is in baselines.tsv. Per-baseline output:
#   results/benchmark/<slug>.metrics.json   (per-round + totals)
#   results/benchmark/<slug>.gpu.csv        (1Hz nvidia-smi samples)
#   results/benchmark/<slug>.summary.txt
#   logs/bench-<slug>.log                   (full Isaac Sim eval log)
#
# Final aggregate:
#   results/benchmark/SUMMARY.md
#
# Env overrides:
#   EVAL_ROUNDS=3 EPISODE_LENGTH_S=120 STEP_HZ=30
#   ONLY=slug,slug   (run subset)
#   SKIP=slug,slug   (skip these)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TSV="$ROOT_DIR/scripts/benchmark/baselines.tsv"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/benchmark}"
mkdir -p "$RESULTS_DIR"

ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

want_slug() {
    local s="$1"
    if [[ -n "$ONLY" ]]; then
        [[ ",$ONLY," == *",$s,"* ]]
        return $?
    fi
    if [[ -n "$SKIP" ]]; then
        if [[ ",$SKIP," == *",$s,"* ]]; then return 1; fi
    fi
    return 0
}

declare -a STATUS
echo "[bench-all] inventory: $TSV"
echo "[bench-all] results:   $RESULTS_DIR"
echo "[bench-all] rounds=${EVAL_ROUNDS:-3} ep_len=${EPISODE_LENGTH_S:-120}s step_hz=${STEP_HZ:-30}"

while IFS=$'\t' read -r slug ptype horizon ckpt server_kind label; do
    # skip blank / comment lines
    [[ -z "$slug" ]] && continue
    [[ "$slug" == \#* ]] && continue
    if ! want_slug "$slug"; then
        echo "[bench-all] -- skip $slug --"
        STATUS+=("$slug:skipped")
        continue
    fi
    echo
    echo "[bench-all] >>> $slug ($label)"
    set +e
    bash "$ROOT_DIR/scripts/benchmark/run_one.sh" \
        "$slug" "$ptype" "$horizon" "$ckpt" "$server_kind" "$label"
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
        STATUS+=("$slug:ok")
    else
        STATUS+=("$slug:fail(exit=$rc)")
    fi
    # Cleanup between runs
    pkill -9 -f "scripts/evaluation/policy_inference.py" 2>/dev/null || true
    sleep 3
done < "$TSV"

echo
echo "[bench-all] ============ STATUS ============"
for s in "${STATUS[@]}"; do echo "  $s"; done

echo
echo "[bench-all] aggregating → $RESULTS_DIR/SUMMARY.md"
python3 "$ROOT_DIR/scripts/benchmark/aggregate.py" \
    "$RESULTS_DIR" \
    --baselines_tsv "$TSV" \
    --out "$RESULTS_DIR/SUMMARY.md"

echo
cat "$RESULTS_DIR/SUMMARY.md"
