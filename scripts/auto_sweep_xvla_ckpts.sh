#!/usr/bin/env bash
# Generic X-VLA ckpt sweep at fixed inference config (h, EMA, etc).  Useful
# for finding sweet-spot ckpt within a single training run.
#
# Per ckpt:
#   1. Kill old server / eval procs
#   2. Start serve_xvla.sh --detach with given N_ACTION_STEPS / EMA_ALPHA
#   3. Run eval_pi05.sh (3 round × 60s, wall_cap 90s)
#   4. Parse Final, append CSV
#
# Env knobs:
#   OUTPUT_DIR        training output dir (with checkpoints/ subdir)
#   CKPT_LIST         space-sep step numbers (e.g. "5000 7000 9000")
#   N_ACTION_STEPS    inference horizon (default empty = config default)
#   EMA_ALPHA         EMA smoothing (default empty = off)
#   EVAL_ROUNDS / EPISODE_S / MAX_ROUND_WALL_S / POLICY_PORT / PROMPT
#   RESULTS_CSV
#
# Usage:
#   OUTPUT_DIR=.../velreweight CKPT_LIST="5000 7000 9000" \
#   N_ACTION_STEPS=32 EMA_ALPHA=0.3 \
#   bash scripts/auto_sweep_xvla_ckpts.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange}"
CKPT_LIST="${CKPT_LIST:-5000 6000 7000 8000 9000 10000}"
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
EMA_ALPHA="${EMA_ALPHA:-}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_S="${EPISODE_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLICY_PORT="${POLICY_PORT:-5558}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/xvla_ckpt_sweep.csv}"

if [[ ! -d "$OUTPUT_DIR/checkpoints" ]]; then
    echo "[csweep] ERROR: $OUTPUT_DIR/checkpoints not found" >&2
    exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "step,n_action_steps,ema_alpha,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

echo "[csweep] output_dir=$OUTPUT_DIR"
echo "[csweep] ckpts=$CKPT_LIST  n_act=${N_ACTION_STEPS:-default}  ema=${EMA_ALPHA:-off}"
echo "[csweep] results → $RESULTS_CSV"
echo

kill_servers() {
    pkill -f "xvla_leisaac.server" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 3
    pkill -9 -f "xvla_leisaac.server" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 2
}

read -r -a CKPT_ARR <<< "$CKPT_LIST"
n=${#CKPT_ARR[@]}

i=0
for step in "${CKPT_ARR[@]}"; do
    i=$((i + 1))
    padded=$(printf "%06d" "$step")
    ckpt_dir="$OUTPUT_DIR/checkpoints/$padded/pretrained_model"
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "[csweep] WARN: $ckpt_dir missing, skip"
        continue
    fi
    echo "===== [$i/$n] step=$step ====="

    kill_servers

    N_ACTION_STEPS="$N_ACTION_STEPS" EMA_ALPHA="$EMA_ALPHA" \
        CKPT="$ckpt_dir" PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/serve_xvla.sh" --detach 2>&1 | tail -5
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[csweep] WARN: server may not be listening, sleep 10s"
        sleep 10
    fi

    eval_log="$REPO_ROOT/logs/xvla_ckpt_eval_${step}.log"
    tmo=$(( EVAL_ROUNDS * MAX_ROUND_WALL_S + 180 ))
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=3000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$PROMPT" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" \
        timeout $tmo bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$step,${N_ACTION_STEPS:-},${EMA_ALPHA:-},${EVAL_ROUNDS},0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        echo "  ⚠️  step=$step: no Final line"
        continue
    fi
    success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
    rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
    placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
    avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
    echo "$step,${N_ACTION_STEPS:-},${EMA_ALPHA:-},$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$RESULTS_CSV"
    echo "  ➜ step=$step  success=$success/$rt  oranges=$placed/$total  avg=${avg}s"
done

kill_servers

echo
echo "[csweep] DONE."
column -s, -t < "$RESULTS_CSV"
echo
echo "[csweep] Best by oranges_placed DESC → rounds_success DESC → avg_round_s ASC:"
tail -n +2 "$RESULTS_CSV" | sort -t, -k6,6nr -k5,5nr -k8,8n | head -3 | \
    awk -F, '{printf "  step=%s n_act=%s ema=%s  oranges=%s/%s  rounds=%s/%s  avg=%ss\n", $1, $2, $3, $6, $7, $5, $4, $8}'
