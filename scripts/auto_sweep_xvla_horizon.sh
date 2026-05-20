#!/usr/bin/env bash
# X-VLA action-chunk horizon sweep — closed-loop eval at one fixed ckpt with
# varying n_action_steps.  No retrain needed: server accepts --n-action-steps.
#
# Per horizon:
#   1. Kill old server / eval procs
#   2. Start serve_xvla.sh --detach with N_ACTION_STEPS=h
#   3. Run eval_pi05.sh (3 round × 60s, no key_R; wall_cap 90s/round)
#   4. Parse Final line, append CSV row
#
# Env knobs:
#   CKPT          ckpt dir (default: Path 1 7k, current best baseline)
#   HORIZONS      space-sep list  (default "1 4 8 16 32")
#   EVAL_ROUNDS   rounds per horizon (default 3)
#   EPISODE_S     seconds per round  (default 60)
#   MAX_ROUND_WALL_S  hard cap per round wall (default 90)
#   POLICY_PORT   server port  (default 5558)
#   PROMPT        task prompt
#   RESULTS_CSV   output CSV
#
# Usage:
#   bash scripts/auto_sweep_xvla_horizon.sh
#   HORIZONS="1 8 32" bash scripts/auto_sweep_xvla_horizon.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="${CKPT:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange/checkpoints/007000/pretrained_model}"
HORIZONS="${HORIZONS:-1 4 8 16 32}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_S="${EPISODE_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLICY_PORT="${POLICY_PORT:-5558}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/xvla_horizon_sweep.csv}"

if [[ ! -d "$CKPT" ]]; then
    echo "[hsweep] ERROR: ckpt dir not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "horizon,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

ckpt_label=$(echo "$CKPT" | sed 's|.*/checkpoints/||; s|/pretrained_model$||')
echo "[hsweep] ckpt=$ckpt_label"
echo "[hsweep] horizons=$HORIZONS"
echo "[hsweep] $EVAL_ROUNDS × ${EPISODE_S}s rounds per horizon (wall_cap=${MAX_ROUND_WALL_S}s)"
echo "[hsweep] results → $RESULTS_CSV"
echo

kill_servers() {
    pkill -f "xvla_leisaac.server" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 3
    pkill -9 -f "xvla_leisaac.server" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 2
}

read -r -a HORIZON_ARR <<< "$HORIZONS"
n=${#HORIZON_ARR[@]}

i=0
for h in "${HORIZON_ARR[@]}"; do
    i=$((i + 1))
    echo "===== [$i/$n] horizon=$h ====="

    kill_servers

    # Start server with horizon override (env var picked up by serve_xvla.sh).
    N_ACTION_STEPS="$h" CKPT="$CKPT" PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/serve_xvla.sh" --detach 2>&1 | tail -5
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[hsweep] WARN: server may not be listening on $POLICY_PORT, sleep 10s"
        sleep 10
    fi

    eval_log="$REPO_ROOT/logs/xvla_horizon_eval_h${h}.log"
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=3000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$PROMPT" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" \
        timeout 360 bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$h,$EVAL_ROUNDS,0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        echo "  ⚠️  horizon=$h: no Final line parsed"
        continue
    fi
    success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
    rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
    placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
    avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
    echo "$h,$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$RESULTS_CSV"
    echo "  ➜ horizon=$h  success=$success/$rt  oranges=$placed/$total  avg=${avg}s"
done

kill_servers

echo
echo "[hsweep] DONE."
column -s, -t < "$RESULTS_CSV"
echo
echo "[hsweep] Best by oranges_placed DESC → rounds_success DESC → avg_round_s ASC:"
tail -n +2 "$RESULTS_CSV" | sort -t, -k4,4nr -k3,3nr -k6,6n | head -3 | \
    awk -F, '{printf "  horizon=%s  oranges=%s/%s  rounds=%s/%s  avg=%ss\n", $1, $4, $5, $3, $2, $6}'
