#!/usr/bin/env bash
# Strict statistical benchmark — 20-round eval + auto-aggregate to distribution.
# Reads baselines.tsv for the slug, calls run_one.sh with full args + EVAL_ROUNDS=20,
# then post-processes metrics.json into:
#   <results_dir>/<slug>.distribution.md  — P(placed=k) histogram + E(oranges) + 5-round σ
#   <results_dir>/<slug>.distribution.svg — inline SVG bar chart
#
# Use this for any leaderboard entry that needs ±2% confidence (vs single 5-round ±11%).
# Memory: feedback-20round-strict-benchmark.
#
# Usage:
#   bash run_one_strict.sh <slug>                        # 20-round default
#   STRICT_ROUNDS=30 bash run_one_strict.sh <slug>       # tighter σ
#
# All other env vars pass through (POLICY_HOST/PORT, etc).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TSV="$SCRIPT_DIR/baselines.tsv"

SLUG="${1:?slug required (see baselines.tsv col1)}"
shift || true

STRICT_ROUNDS="${STRICT_ROUNDS:-20}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/benchmark}"
mkdir -p "$RESULTS_DIR"

# Lookup the slug in baselines.tsv (tab-separated)
ROW="$(awk -F'\t' -v s="$SLUG" 'BEGIN{IGNORECASE=0} !/^#/ && $1==s {print; exit}' "$TSV")"
if [[ -z "$ROW" ]]; then
    echo "[strict] ERROR: slug '$SLUG' not in $TSV" >&2
    exit 1
fi
IFS=$'\t' read -r slug ptype horizon ckpt server_kind label extra_env <<< "$ROW"

echo "[strict] === $SLUG ($label) ==="
echo "[strict] policy_type=$ptype horizon=$horizon ckpt=$ckpt server=$server_kind"
[[ -n "${extra_env:-}" ]] && echo "[strict] extra_env: $extra_env"
echo "[strict] EVAL_ROUNDS=$STRICT_ROUNDS (~$((STRICT_ROUNDS * 3))min wall)"

# Run the eval — env per-row + force EVAL_ROUNDS=20
env ${extra_env:-} EVAL_ROUNDS="$STRICT_ROUNDS" RESULTS_DIR="$RESULTS_DIR" \
    bash "$SCRIPT_DIR/run_one.sh" "$slug" "$ptype" "$horizon" "$ckpt" "$server_kind" "$label"

# Post-process
METRICS_JSON="$RESULTS_DIR/${SLUG}.metrics.json"
if [[ ! -f "$METRICS_JSON" ]]; then
    echo "[strict] ERROR: $METRICS_JSON not found — eval failed?" >&2
    exit 1
fi

DIST_MD="$RESULTS_DIR/${SLUG}.distribution.md"
DIST_SVG="$RESULTS_DIR/${SLUG}.distribution.svg"

python3 "$SCRIPT_DIR/aggregate_distribution.py" "$METRICS_JSON" \
    --out "$DIST_MD" \
    --svg "$DIST_SVG"

echo "[strict] $SLUG done"
echo "[strict]   distribution → $DIST_MD"
echo "[strict]   svg          → $DIST_SVG"
