#!/usr/bin/env bash
# Auto closed-loop sweep over X-VLA checkpoints.  Picks the best ckpt by ACTUAL
# Isaac Sim PickOrange placed-count rather than offline val_loss (which has been
# shown to NOT predict closed-loop performance reliably).
#
# Per ckpt:
#   1. Kill any old server / eval procs
#   2. Start serve_xvla.sh --detach with that ckpt
#   3. Run eval_pi05.sh in foreground (3 round × 60s, no key_R press)
#   4. Parse the "Final success rate" / "oranges:" lines
#   5. Append to logs/closed_loop_sweep.csv
#
# Run unattended; user only needs to NOT press R during the Isaac Sim windows.
#
# Env knobs:
#   OUTPUT_DIR     X-VLA training output dir (default: xvla-leisaac-pick-orange)
#   CKPT_FILTER    grep filter for ckpts (default: matches all 6-digit dirs)
#   EVAL_ROUNDS    rounds per ckpt (default 3)
#   EPISODE_S      seconds per round (default 60)
#   POLICY_PORT    server port (default 5558)
#   PROMPT         task prompt
#   RESULTS_CSV    output CSV
#
# Usage:
#   bash scripts/auto_sweep_xvla_closed_loop.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange}"
CKPT_FILTER="${CKPT_FILTER:-[0-9]+}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_S="${EPISODE_S:-60}"
POLICY_PORT="${POLICY_PORT:-5558}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
RESULTS_CSV="${RESULTS_CSV:-$REPO_ROOT/logs/closed_loop_sweep.csv}"

if [[ ! -d "$OUTPUT_DIR/checkpoints" ]]; then
    echo "[sweep] ERROR: $OUTPUT_DIR/checkpoints not found" >&2
    exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
[[ -f "$RESULTS_CSV" ]] || echo "ckpt_step,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$RESULTS_CSV"

# CKPT_LIST overrides filter+sort: space-separated step numbers in scan order
# (e.g. CKPT_LIST="7000 8000 6000 9000 5000 10000 4000" for middle-out).
if [[ -n "${CKPT_LIST:-}" ]]; then
    ckpts=""
    for step in $CKPT_LIST; do
        padded=$(printf "%06d" "$step")
        if [[ -d "$OUTPUT_DIR/checkpoints/$padded" ]]; then
            ckpts="$ckpts$OUTPUT_DIR/checkpoints/$padded"$'\n'
        else
            echo "[sweep] WARN: step $step ($padded) not found, skipping" >&2
        fi
    done
    ckpts="${ckpts%$'\n'}"
else
    ckpts=$(find "$OUTPUT_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex ".*/$CKPT_FILTER" | sort -V)
fi
if [[ -z "$ckpts" ]]; then
    echo "[sweep] ERROR: no ckpts matched" >&2
    exit 1
fi
n=$(echo "$ckpts" | wc -l)

echo "[sweep] $n ckpts to evaluate under $OUTPUT_DIR/checkpoints/"
echo "[sweep] $EVAL_ROUNDS × ${EPISODE_S}s rounds per ckpt"
echo "[sweep] results → $RESULTS_CSV"
echo

i=0
for ckpt in $ckpts; do
    i=$((i + 1))
    step=$(basename "$ckpt")
    echo "===== [$i/$n] ckpt=$step ====="

    # Kill any leftover server / eval procs.
    pkill -f "xvla_leisaac.server" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 3
    pkill -9 -f "xvla_leisaac.server" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 2

    # Start server (detached, wait for ready marker).
    server_log="$REPO_ROOT/logs/sweep_server_${step}.log"
    CKPT="$ckpt/pretrained_model" PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/serve_xvla.sh" --detach 2>&1 | tail -5
    if ! ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[sweep] WARN: server may not be listening on $POLICY_PORT, retry sleep"
        sleep 10
    fi

    # Run eval (foreground, ~200s for 3 rounds + reset overhead).
    eval_log="$REPO_ROOT/logs/sweep_eval_${step}.log"
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=3000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$PROMPT" \
        timeout 280 bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | tee "$eval_log" | \
        grep -E "Evaluating episode|placed|Final|success rate" | tail -10 || true

    # Parse "Final success rate: X.XXX  [a/b], oranges: c/d, avg_round_s: e"
    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$step,$EVAL_ROUNDS,0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$RESULTS_CSV"
        echo "  ⚠️  no Final line parsed"
        continue
    fi
    rate=$(echo "$final_line" | grep -oE "rate: [0-9.]+" | grep -oE "[0-9.]+")
    success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
    rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
    placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
    total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
    avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
    echo "$step,$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$RESULTS_CSV"
    echo "  ➜ ckpt=$step  success=$success/$rt  oranges=$placed/$total  avg=${avg}s"
done

# Kill final server.
pkill -f "xvla_leisaac.server" 2>/dev/null
sleep 2

echo
echo "[sweep] DONE.  Results:"
column -s, -t < "$RESULTS_CSV" | tail -n +1
echo
echo "[sweep] Best ckpt(s) by rounds_success DESC → oranges_placed DESC → avg_round_s ASC:"
# CSV columns: ckpt_step(1), rounds_total(2), rounds_success(3), oranges_placed(4), oranges_total(5), avg_round_s(6)
tail -n +2 "$RESULTS_CSV" | sort -t, -k3,3nr -k4,4nr -k6,6n | head -3 | awk -F, '{printf "  step %s  rounds=%s/%s  oranges=%s/%s  avg=%ss\n", $1, $3, $2, $4, $5, $6}'
