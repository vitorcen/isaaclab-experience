#!/usr/bin/env bash
# X-VLA EMA action-smoothing sweep — closed-loop eval at one fixed ckpt with
# varying ema_alpha.  Server accepts --ema-alpha (cleared on reset).
#
# alpha=1.0 = no smoothing (baseline);  alpha=0.5 = average new + prev;
# alpha=0.3 = strong smoothing (carries 70% of prev step).
#
# Per alpha:
#   1. Kill old server / eval procs
#   2. Start serve_xvla.sh --detach with EMA_ALPHA=α (+ optional N_ACTION_STEPS)
#   3. Run eval_pi05.sh (3 round × 60s, wall_cap 90s)
#   4. Parse Final line, append CSV row
#
# Env knobs:
#   CKPT             ckpt dir (default Path 1 7k)
#   ALPHAS           space-sep list (default "0.3 0.5 0.7 1.0")
#   N_ACTION_STEPS   passed to server (default empty = config default 8)
#   EVAL_ROUNDS / EPISODE_S / MAX_ROUND_WALL_S / POLICY_PORT / PROMPT
#   RESULTS_CSV
#
# Usage:
#   bash scripts/auto_sweep_xvla_ema.sh
#   ALPHAS="0.5 0.7" N_ACTION_STEPS=4 bash scripts/auto_sweep_xvla_ema.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="${CKPT:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange/checkpoints/007000/pretrained_model}"
ALPHAS="${ALPHAS:-0.3 0.5 0.7 1.0}"
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_S="${EPISODE_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLICY_PORT="${POLICY_PORT:-5558}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/xvla_ema_sweep.csv}"

if [[ ! -d "$CKPT" ]]; then
    echo "[esweep] ERROR: ckpt dir not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "alpha,n_action_steps,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

ckpt_label=$(echo "$CKPT" | sed 's|.*/checkpoints/||; s|/pretrained_model$||')
echo "[esweep] ckpt=$ckpt_label  n_action_steps=${N_ACTION_STEPS:-default}"
echo "[esweep] alphas=$ALPHAS"
echo "[esweep] results → $RESULTS_CSV"
echo

kill_servers() {
    pkill -f "xvla_leisaac.server" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 3
    pkill -9 -f "xvla_leisaac.server" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 2
}

read -r -a ALPHA_ARR <<< "$ALPHAS"
n=${#ALPHA_ARR[@]}

i=0
for a in "${ALPHA_ARR[@]}"; do
    i=$((i + 1))
    echo "===== [$i/$n] alpha=$a ====="

    kill_servers

    EMA_ALPHA="$a" N_ACTION_STEPS="$N_ACTION_STEPS" CKPT="$CKPT" PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/serve_xvla.sh" --detach 2>&1 | tail -5
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[esweep] WARN: server may not be listening, sleep 10s"
        sleep 10
    fi

    eval_log="$REPO_ROOT/logs/xvla_ema_eval_a${a}.log"
    # Dynamic timeout: rounds × wall_cap + 180s Isaac Sim load + safety margin.
    tmo=$(( EVAL_ROUNDS * MAX_ROUND_WALL_S + 180 ))
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=3000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$PROMPT" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" \
        timeout $tmo bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$a,${N_ACTION_STEPS:-},${EVAL_ROUNDS},0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        echo "  ⚠️  alpha=$a: no Final line"
        continue
    fi
    success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
    rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
    placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
    avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
    echo "$a,${N_ACTION_STEPS:-},$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$RESULTS_CSV"
    echo "  ➜ alpha=$a  success=$success/$rt  oranges=$placed/$total  avg=${avg}s"
done

kill_servers

echo
echo "[esweep] DONE."
column -s, -t < "$RESULTS_CSV"
echo
echo "[esweep] Best by oranges_placed DESC → rounds_success DESC → avg_round_s ASC:"
tail -n +2 "$RESULTS_CSV" | sort -t, -k5,5nr -k4,4nr -k7,7n | head -3 | \
    awk -F, '{printf "  alpha=%s n_act=%s  oranges=%s/%s  rounds=%s/%s  avg=%ss\n", $1, $2, $5, $6, $4, $3, $7}'
