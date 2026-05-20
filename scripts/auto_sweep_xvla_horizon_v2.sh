#!/usr/bin/env bash
# X-VLA horizon sweep v2 — like horizon sweep but with EVAL_ROUNDS env and
# dynamic timeout for 6-round eval.  Same ckpt, varying n_action_steps.
#
# Per horizon: kill servers → start with N_ACTION_STEPS=h → 6 round eval → log.
#
# Env knobs:
#   CKPT             ckpt dir
#   HORIZONS         space-sep list (default "1 4 8 16 32")
#   EVAL_ROUNDS / EPISODE_S / MAX_ROUND_WALL_S / POLICY_PORT / PROMPT
#   RESULTS_CSV

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="${CKPT:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange/checkpoints/008000/pretrained_model}"
HORIZONS="${HORIZONS:-1 4 8 16 32}"
EVAL_ROUNDS="${EVAL_ROUNDS:-6}"
EPISODE_S="${EPISODE_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLICY_PORT="${POLICY_PORT:-5558}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/xvla_horizon_sweep_v2.csv}"

if [[ ! -d "$CKPT" ]]; then
    echo "[hsweep2] ERROR: ckpt dir not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "horizon,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

ckpt_label=$(echo "$CKPT" | sed 's|.*/checkpoints/||; s|/pretrained_model$||')
echo "[hsweep2] ckpt=$ckpt_label  horizons=$HORIZONS  rounds=$EVAL_ROUNDS"
echo "[hsweep2] results → $RESULTS_CSV"

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

    N_ACTION_STEPS="$h" CKPT="$CKPT" PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/serve_xvla.sh" --detach 2>&1 | tail -3
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[hsweep2] WARN: server not listening, sleep 10s"
        sleep 10
    fi

    eval_log="$REPO_ROOT/logs/xvla_horizon_v2_eval_h${h}.log"
    tmo=$(( EVAL_ROUNDS * MAX_ROUND_WALL_S + 180 ))
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=3000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$PROMPT" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" \
        timeout $tmo bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$h,$EVAL_ROUNDS,0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        echo "  ⚠️  h=$h: no Final"
        continue
    fi
    success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
    rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
    placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
    avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
    echo "$h,$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$RESULTS_CSV"
    echo "  ➜ h=$h  oranges=$placed/$total"
done

kill_servers
echo
column -s, -t < "$RESULTS_CSV"
