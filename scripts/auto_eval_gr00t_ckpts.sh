#!/usr/bin/env bash
# Auto-eval-on-save loop for GR00T-N1.6 finetune.
#
# Monitors $OUTPUT_DIR/checkpoint-* (HF Trainer naming), and for every new
# checkpoint:
#   1. (Re)start GR00T inference server via server/start_server.sh --gr00t-only
#      with GR00T_MODEL_PATH=<ckpt> GR00T_EMBODIMENT_TAG=new_embodiment
#   2. Run LeIsaac/server/eval_gr00t.sh ($EVAL_ROUNDS round × $EPISODE_LENGTH s)
#   3. Parse "Final success rate" line, append CSV row
#   4. Kill server, wait for next ckpt
#
# Two modes:
#   POLL=1 (default): infinite-loop poll for new checkpoint-* dirs
#                     (use during live training). Exits when training-done sentinel
#                     file ${SENTINEL} is touched, or on Ctrl-C.
#   CKPT_LIST=...   : one-shot evaluate listed step numbers, then exit.
#
# Env knobs:
#   OUTPUT_DIR      training output dir (default: LeIsaac/outputs/gr00t-n16-leisaac-pick-orange)
#   POLL_S          poll interval seconds (default: 60)
#   EVAL_ROUNDS     default 6
#   EPISODE_LENGTH  default 60
#   MAX_ROUND_WALL_S  default 90
#   ACTION_HORIZON  GR00T n_action_steps override (default 16)
#   POLICY_PORT     GR00T server port (default 5555)
#   PROMPT          locked default = "Pick up the orange and put it in the plate"
#   RESULTS_CSV     output CSV path
#   SENTINEL        file path that, when present, ends the poll loop

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/LeIsaac/outputs/gr00t-n16-leisaac-pick-orange}"
CKPT_LIST="${CKPT_LIST:-}"
POLL_S="${POLL_S:-60}"
EVAL_ROUNDS="${EVAL_ROUNDS:-6}"
EPISODE_LENGTH="${EPISODE_LENGTH:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
POLICY_PORT="${POLICY_PORT:-5555}"
PROMPT="${PROMPT:-Pick up the orange and put it in the plate}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/gr00t_n16_ckpts.csv}"
SENTINEL="${SENTINEL:-$OUTPUT_DIR/.training_done}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "step,n_action_steps,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

echo "[auto-eval] output_dir=$OUTPUT_DIR"
echo "[auto-eval] poll=${POLL_S}s  rounds=$EVAL_ROUNDS  wall_cap=${MAX_ROUND_WALL_S}s  horizon=$ACTION_HORIZON"
echo "[auto-eval] prompt: $PROMPT"
echo "[auto-eval] csv: $RESULTS_CSV"
echo

kill_servers() {
    pkill -f "gr00t/eval/run_gr00t_server.py" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 3
    pkill -9 -f "gr00t/eval/run_gr00t_server.py" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 2
}

eval_one_ckpt() {
    local ckpt_dir="$1"
    local step="$2"
    echo "===== ckpt=$step  dir=$ckpt_dir ====="

    kill_servers

    GR00T_MODEL_PATH="$ckpt_dir" GR00T_EMBODIMENT_TAG="$EMBODIMENT_TAG" \
        GR00T_PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/start_server.sh" --gr00t-only 2>&1 | tail -10

    # start_server.sh waits up to 120s for port — give a small extra buffer
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[auto-eval] WARN: server not yet listening on :$POLICY_PORT, sleep 15s extra"
        sleep 15
    fi

    eval_log="$REPO_ROOT/logs/gr00t_ckpt_eval_${step}.log"
    tmo=$(( EVAL_ROUNDS * MAX_ROUND_WALL_S + 300 ))
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=10000 \
        ACTION_HORIZON="$ACTION_HORIZON" \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_LENGTH" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" PROMPT="$PROMPT" \
        timeout $tmo bash "$REPO_ROOT/LeIsaac/server/eval_gr00t.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$step,$ACTION_HORIZON,$EVAL_ROUNDS,0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        echo "  ⚠️  step=$step: no Final line"
    else
        success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
        rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
        placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
        total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
        avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
        echo "$step,$ACTION_HORIZON,$rt,$success,$placed,$total,${avg:-NaN},\"$final_line\"" >> "$RESULTS_CSV"
        echo "  ➜ step=$step  success=$success/$rt  oranges=$placed/$total  avg=${avg}s"
    fi

    kill_servers
}

# One-shot list mode
if [[ -n "$CKPT_LIST" ]]; then
    read -r -a CKPT_ARR <<< "$CKPT_LIST"
    for step in "${CKPT_ARR[@]}"; do
        ckpt_dir="$OUTPUT_DIR/checkpoint-$step"
        if [[ ! -d "$ckpt_dir" ]]; then
            echo "[auto-eval] skip: $ckpt_dir missing"
            continue
        fi
        eval_one_ckpt "$ckpt_dir" "$step"
    done
    column -s, -t < "$RESULTS_CSV" | tail -20
    exit 0
fi

# Poll mode
declare -A SEEN
while true; do
    if [[ -f "$SENTINEL" ]]; then
        echo "[auto-eval] sentinel $SENTINEL detected, exiting"
        break
    fi
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        sleep "$POLL_S"
        continue
    fi
    for ckpt_dir in "$OUTPUT_DIR"/checkpoint-*; do
        [[ -d "$ckpt_dir" ]] || continue
        step="${ckpt_dir##*/checkpoint-}"
        # Skip if already evaluated this run
        [[ -n "${SEEN[$step]:-}" ]] && continue
        # Skip if ckpt still being written (no model file yet)
        if ! ls "$ckpt_dir"/*.safetensors "$ckpt_dir"/pytorch_model*.bin 2>/dev/null | head -1 >/dev/null; then
            continue
        fi
        SEEN[$step]=1
        eval_one_ckpt "$ckpt_dir" "$step"
        echo
    done
    sleep "$POLL_S"
done

echo
echo "[auto-eval] DONE."
column -s, -t < "$RESULTS_CSV"
echo
echo "[auto-eval] Top by oranges DESC → success DESC → time ASC:"
tail -n +2 "$RESULTS_CSV" | sort -t, -k5,5nr -k4,4nr -k7,7n | head -5 | \
    awk -F, '{printf "  step=%s h=%s  oranges=%s/%s  rounds=%s/%s  avg=%ss\n", $1, $2, $5, $6, $4, $3, $7}'
