#!/usr/bin/env bash
# X-VLA TAE (Temporal Action Ensembling) sweep — ALOHA 2304.13705 §4.3.
# Server bypasses queue: per step → predict full chunk, buffer last K, ensemble
# overlapping predictions with exp(-m·age) weights.
#
# Per (buffer K, decay m):
#   1. Kill old server / eval procs
#   2. Start serve_xvla.sh --detach with TAE_BUFFER=K TAE_M=m
#   3. Run eval_pi05.sh (3 round × 60s, wall_cap 90s)
#   4. Parse Final line, append CSV row
#
# Env knobs:
#   CKPT            ckpt dir (default Path 1 7k)
#   CONFIGS         space-sep "K:m" pairs (default "2:0.1 4:0.1 4:0.3 8:0.1")
#   EVAL_ROUNDS / EPISODE_S / MAX_ROUND_WALL_S / POLICY_PORT / PROMPT
#   RESULTS_CSV
#
# Usage:
#   bash scripts/auto_sweep_xvla_tae.sh
#   CONFIGS="4:0.1 8:0.1" bash scripts/auto_sweep_xvla_tae.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="${CKPT:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange/checkpoints/007000/pretrained_model}"
CONFIGS="${CONFIGS:-2:0.1 4:0.1 4:0.3 8:0.1}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_S="${EPISODE_S:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLICY_PORT="${POLICY_PORT:-5558}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/xvla_tae_sweep.csv}"

if [[ ! -d "$CKPT" ]]; then
    echo "[tsweep] ERROR: ckpt not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "tae_buffer,tae_m,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

ckpt_label=$(echo "$CKPT" | sed 's|.*/checkpoints/||; s|/pretrained_model$||')
echo "[tsweep] ckpt=$ckpt_label"
echo "[tsweep] configs=$CONFIGS"
echo "[tsweep] results → $RESULTS_CSV"
echo

kill_servers() {
    pkill -f "xvla_leisaac.server" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 3
    pkill -9 -f "xvla_leisaac.server" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 2
}

read -r -a CFG_ARR <<< "$CONFIGS"
n=${#CFG_ARR[@]}

i=0
for cfg in "${CFG_ARR[@]}"; do
    i=$((i + 1))
    K="${cfg%%:*}"
    M="${cfg##*:}"
    echo "===== [$i/$n] TAE K=$K m=$M ====="

    kill_servers

    TAE_BUFFER="$K" TAE_M="$M" CKPT="$CKPT" PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/serve_xvla.sh" --detach 2>&1 | tail -5
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[tsweep] WARN: server may not be listening, sleep 10s"
        sleep 10
    fi

    eval_log="$REPO_ROOT/logs/xvla_tae_eval_K${K}_m${M}.log"
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=5000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$PROMPT" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" \
        timeout 420 bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$K,$M,${EVAL_ROUNDS},0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        echo "  ⚠️  K=$K m=$M: no Final line"
        continue
    fi
    success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
    rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
    placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
    avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
    echo "$K,$M,$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$RESULTS_CSV"
    echo "  ➜ K=$K m=$M  success=$success/$rt  oranges=$placed/$total  avg=${avg}s"
done

kill_servers

echo
echo "[tsweep] DONE."
column -s, -t < "$RESULTS_CSV"
echo
echo "[tsweep] Best by oranges_placed DESC → rounds_success DESC → avg_round_s ASC:"
tail -n +2 "$RESULTS_CSV" | sort -t, -k5,5nr -k4,4nr -k7,7n | head -3 | \
    awk -F, '{printf "  K=%s m=%s  oranges=%s/%s  rounds=%s/%s  avg=%ss\n", $1, $2, $5, $6, $4, $3, $7}'
