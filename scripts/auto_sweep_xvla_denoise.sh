#!/usr/bin/env bash
# X-VLA num_denoising_steps sweep at fixed ckpt + h=32, no EMA.  Tests RF head
# denoising step count effect on closed-loop performance.
#
# Env knobs:
#   CKPT             ckpt dir
#   STEPS_LIST       space-sep list of denoise steps (default "5 10 20")
#   N_ACTION_STEPS   horizon (default 32)
#   EVAL_ROUNDS / EPISODE_S / MAX_ROUND_WALL_S / POLICY_PORT / PROMPT
#   RESULTS_CSV

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="${CKPT:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange/checkpoints/008000/pretrained_model}"
STEPS_LIST="${STEPS_LIST:-5 10 20}"
N_ACTION_STEPS="${N_ACTION_STEPS:-32}"
EVAL_ROUNDS="${EVAL_ROUNDS:-6}"
EPISODE_S="${EPISODE_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLICY_PORT="${POLICY_PORT:-5558}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/xvla_denoise_sweep.csv}"

if [[ ! -d "$CKPT" ]]; then
    echo "[dsweep] ERROR: ckpt not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "denoise_steps,n_action_steps,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

ckpt_label=$(echo "$CKPT" | sed 's|.*/checkpoints/||; s|/pretrained_model$||')
echo "[dsweep] ckpt=$ckpt_label  n_act=$N_ACTION_STEPS  denoise_steps=$STEPS_LIST  rounds=$EVAL_ROUNDS"
echo "[dsweep] results → $RESULTS_CSV"

kill_servers() {
    pkill -f "xvla_leisaac.server" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 3
    pkill -9 -f "xvla_leisaac.server" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 2
}

read -r -a STEPS_ARR <<< "$STEPS_LIST"
n=${#STEPS_ARR[@]}

i=0
for ds in "${STEPS_ARR[@]}"; do
    i=$((i + 1))
    echo "===== [$i/$n] denoise_steps=$ds ====="

    kill_servers

    NUM_DENOISING_STEPS="$ds" N_ACTION_STEPS="$N_ACTION_STEPS" \
        CKPT="$CKPT" PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/serve_xvla.sh" --detach 2>&1 | tail -3
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[dsweep] WARN: server not listening, sleep 10s"
        sleep 10
    fi

    eval_log="$REPO_ROOT/logs/xvla_denoise_eval_ds${ds}.log"
    tmo=$(( EVAL_ROUNDS * MAX_ROUND_WALL_S + 180 ))
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=5000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$PROMPT" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" \
        timeout $tmo bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$ds,$N_ACTION_STEPS,$EVAL_ROUNDS,0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        echo "  ⚠️  ds=$ds: no Final"
        continue
    fi
    success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
    rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
    placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
    avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
    echo "$ds,$N_ACTION_STEPS,$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$RESULTS_CSV"
    echo "  ➜ ds=$ds  oranges=$placed/$total"
done

kill_servers
echo
column -s, -t < "$RESULTS_CSV"
