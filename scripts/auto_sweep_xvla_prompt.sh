#!/usr/bin/env bash
# X-VLA prompt sweep — vary natural-language instruction at inference at fixed
# ckpt + h=32, no EMA.  Tests how much wording matters for closed-loop.
#
# Prompts are provided as PROMPT_LIST with a '|' delimiter so they may contain
# spaces.  Each prompt is run 6 rounds × 60s.
#
# Env knobs:
#   CKPT          ckpt dir
#   PROMPT_LIST   '|'-separated prompts (default has 3 variants)
#   N_ACTION_STEPS / EVAL_ROUNDS / etc.
#   RESULTS_CSV

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="${CKPT:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange/checkpoints/008000/pretrained_model}"
PROMPT_LIST="${PROMPT_LIST:-Pick up the orange and put it in the plate|Pick up all oranges and place them in the plate|Move the orange from the table to the plate}"
N_ACTION_STEPS="${N_ACTION_STEPS:-32}"
EVAL_ROUNDS="${EVAL_ROUNDS:-6}"
EPISODE_S="${EPISODE_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLICY_PORT="${POLICY_PORT:-5558}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/xvla_prompt_sweep.csv}"

if [[ ! -d "$CKPT" ]]; then
    echo "[psweep] ERROR: ckpt not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "prompt,n_action_steps,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

ckpt_label=$(echo "$CKPT" | sed 's|.*/checkpoints/||; s|/pretrained_model$||')
echo "[psweep] ckpt=$ckpt_label  n_act=$N_ACTION_STEPS  rounds=$EVAL_ROUNDS"
echo "[psweep] results → $RESULTS_CSV"

kill_servers() {
    pkill -f "xvla_leisaac.server" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 3
    pkill -9 -f "xvla_leisaac.server" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 2
}

IFS='|' read -r -a PROMPT_ARR <<< "$PROMPT_LIST"
n=${#PROMPT_ARR[@]}

i=0
for prompt in "${PROMPT_ARR[@]}"; do
    i=$((i + 1))
    safe=$(echo "$prompt" | tr -cd 'A-Za-z0-9 ' | tr ' ' '_' | head -c 40)
    echo "===== [$i/$n] prompt=\"$prompt\" ====="

    kill_servers

    N_ACTION_STEPS="$N_ACTION_STEPS" CKPT="$CKPT" PORT="$POLICY_PORT" PROMPT="$prompt" \
        bash "$REPO_ROOT/server/serve_xvla.sh" --detach 2>&1 | tail -3
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[psweep] WARN: server not listening, sleep 10s"
        sleep 10
    fi

    eval_log="$REPO_ROOT/logs/xvla_prompt_eval_${safe}.log"
    tmo=$(( EVAL_ROUNDS * MAX_ROUND_WALL_S + 180 ))
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=3000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$prompt" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" \
        timeout $tmo bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "\"$prompt\",$N_ACTION_STEPS,$EVAL_ROUNDS,0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        continue
    fi
    success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
    rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
    placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
    avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
    echo "\"$prompt\",$N_ACTION_STEPS,$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$RESULTS_CSV"
    echo "  ➜ \"$prompt\"  oranges=$placed/$total"
done

kill_servers
echo
column -s, -t < "$RESULTS_CSV"
