#!/usr/bin/env bash
# Run STRICT 20-round benchmark on multiple baselines back-to-back.
# Sequential — each takes ~50min (~10min Isaac startup + 20×3 episode min).
# Total for 8 baselines ≈ 7 hours.
#
# Usage:
#   bash run_all_strict.sh                                # all in baselines.tsv
#   ONLY=gr00t-n16-self,gr00t-n16-hispace bash run_all_strict.sh
#   STRICT_ROUNDS=20 bash run_all_strict.sh
#
# Per-slug output:
#   results/benchmark/<slug>.metrics.json + .distribution.md + .distribution.svg
#
# Aggregated leaderboard:
#   scripts/benchmark/STRICT_LEADERBOARD.md (sorted by worst-case mean−1σ DESC)

set -uo pipefail   # NOT -e: continue past per-slug failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TSV="$SCRIPT_DIR/baselines.tsv"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/benchmark}"
mkdir -p "$RESULTS_DIR"

ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
STRICT_ROUNDS="${STRICT_ROUNDS:-20}"

want_slug() {
    local s="$1"
    if [[ -n "$ONLY" ]]; then
        [[ ",$ONLY," == *",$s,"* ]] && return 0 || return 1
    fi
    if [[ -n "$SKIP" ]] && [[ ",$SKIP," == *",$s,"* ]]; then
        return 1
    fi
    return 0
}

declare -a STATUS
START_T=$(date +%s)
echo "[strict-all] inventory: $TSV"
echo "[strict-all] STRICT_ROUNDS=$STRICT_ROUNDS"
echo "[strict-all] estimate: ~50 min/baseline; sequential"
echo "[strict-all] start: $(date)"

while IFS=$'\t' read -r slug ptype horizon ckpt server_kind label extra_env; do
    [[ -z "$slug" ]] && continue
    [[ "$slug" == \#* ]] && continue
    if ! want_slug "$slug"; then
        STATUS+=("$slug:skipped")
        continue
    fi
    echo
    echo "[strict-all] >>> $slug ($label)"
    SLUG_START=$(date +%s)
    if STRICT_ROUNDS="$STRICT_ROUNDS" RESULTS_DIR="$RESULTS_DIR" \
        bash "$SCRIPT_DIR/run_one_strict.sh" "$slug"; then
        STATUS+=("$slug:OK")
    else
        echo "[strict-all] !!! $slug FAILED, continuing"
        STATUS+=("$slug:FAIL")
    fi
    SLUG_END=$(date +%s)
    echo "[strict-all] <<< $slug took $(( (SLUG_END - SLUG_START) / 60 ))min"
done < "$TSV"

END_T=$(date +%s)
TOTAL_MIN=$(( (END_T - START_T) / 60 ))

# Build aggregated leaderboard
echo
echo "[strict-all] === summary ==="
for s in "${STATUS[@]}"; do echo "[strict-all]   $s"; done
echo "[strict-all] total wall: ${TOTAL_MIN}min"
LEADERBOARD_OUT="$SCRIPT_DIR/STRICT_LEADERBOARD.md"
echo "[strict-all] building $LEADERBOARD_OUT"

python3 "$SCRIPT_DIR/aggregate_strict_leaderboard.py" \
    --results_dir "$RESULTS_DIR" \
    --out "$LEADERBOARD_OUT" \
    --min_rounds 20

echo "[strict-all] DONE → $LEADERBOARD_OUT"
