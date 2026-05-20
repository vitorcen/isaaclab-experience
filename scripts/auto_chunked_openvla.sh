#!/usr/bin/env bash
# OpenVLA chunked train + GUI eval loop.
#
# Each iteration:
#   1. Train next 500-step chunk (resume from previous ckpt, MAX_STEPS=target)
#   2. Kill server / leftover procs
#   3. Start serve_openvla.sh --detach with that ckpt
#   4. Run eval_pi05.sh (3 round × 60s, GUI via DISPLAY=:0 --enable_cameras)
#   5. Parse "oranges: a/b" from log, append to CSV
#   6. Kill server, loop
#
# Env knobs:
#   START_STEP    base step to resume from         (default 5000)
#   END_STEP      stop after reaching this step    (default 10000)
#   CHUNK         step interval per eval           (default 500)
#   EVAL_ROUNDS   rounds per ckpt eval             (default 3)
#   EPISODE_S     seconds per round                (default 60)
#   POLICY_PORT   server port                      (default 5557)
#   PROMPT        task prompt
#   CSV           output CSV path
#
# Usage:
#   bash scripts/auto_chunked_openvla.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/LeIsaac/outputs/openvla-leisaac-pick-orange}"
# Seed LoRA from another checkpoint dir on the FIRST chunk of a fresh OUTPUT_DIR.
# Used to migrate across precision changes (4-bit → 8-bit) without dragging
# stale optimizer.pt — see openvla_crash_diagnosis HTML.  Empty = no seed.
LOAD_LORA_FROM="${LOAD_LORA_FROM:-}"
START_STEP="${START_STEP:-5000}"
END_STEP="${END_STEP:-10000}"
CHUNK="${CHUNK:-500}"
# Cumulative-step display anchor.  When seeding LoRA from an external ckpt that
# was already trained for N steps, set OFFSET=N.  Display labels (cumulative)
# = dir_step + OFFSET.  ROUND_TO snaps eval boundaries to nice multiples of N.
# Eg. OFFSET=6100, ROUND_TO=500: ckpt-900 → display 7000; ckpt-1400 → 7500.
OFFSET="${OFFSET:-0}"
ROUND_TO="${ROUND_TO:-$CHUNK}"
# Skip eval for cumulative steps strictly below this threshold (still trains
# normally).  Use to avoid wasting eval time on under-trained ckpts.
START_EVAL_STEP="${START_EVAL_STEP:-0}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_S="${EPISODE_S:-60}"
# Wall-clock hard cap per round (Isaac Sim load + warmup excluded).  Eval timer
# starts after env.reset() + sim warmup, so episode 1 doesn't pay the load cost.
# 90 = 60s episode_length_s + 30s headroom for sim_step + policy inference.
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
POLICY_PORT="${POLICY_PORT:-5557}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
CSV="${CSV:-$REPO_ROOT/logs/openvla_chunked_500.csv}"

mkdir -p "$(dirname "$CSV")" "$REPO_ROOT/logs"
[[ -f "$CSV" ]] || echo "step,rounds_total,rounds_success,oranges_placed,oranges_total,avg_round_s,raw_line" > "$CSV"

kill_servers() {
    pkill -f "openvla_leisaac.server" 2>/dev/null
    pkill -f "policy_inference.py" 2>/dev/null
    sleep 4
    pkill -9 -f "openvla_leisaac.server" 2>/dev/null
    pkill -9 -f "policy_inference.py" 2>/dev/null
    sleep 3
}

# Auto-detect resume point: latest dir_step ckpt that's complete.  When the
# OUTPUT_DIR is empty AND LOAD_LORA_FROM is set, use 0 as the start (first
# chunk seeds the LoRA from external dir).  Otherwise honor explicit START_STEP.
# NOTE: scan ANY dir_step, not just CHUNK-aligned ones — mid-chunk ckpts (e.g.
# ckpt-500 when targets are ckpt-900/1400/...) are also valid resume points.
latest=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' \
    | sort -n | tail -1)
if [[ -n "$latest" ]]; then
    START_STEP="$latest"
    echo "[chunk] auto-detected resume point: ckpt-$latest (cumulative $((latest + OFFSET)))"
elif [[ -n "$LOAD_LORA_FROM" ]]; then
    START_STEP=0
    echo "[chunk] empty OUTPUT_DIR, will seed LoRA from $LOAD_LORA_FROM on first chunk"
else
    START_STEP=0
    echo "[chunk] empty OUTPUT_DIR + no LOAD_LORA_FROM → from-scratch training (fresh LoRA init)"
fi

# Compute target dir_step that lands cumulative on a ROUND_TO boundary.
# cumulative_target = next multiple of ROUND_TO strictly > (prev_dir_step + OFFSET).
compute_target() {
    local prev_dir=$1
    local prev_cum=$((prev_dir + OFFSET))
    local next_cum=$(( (prev_cum / ROUND_TO + 1) * ROUND_TO ))
    echo $((next_cum - OFFSET))
}

prev=$START_STEP
target=$(compute_target "$prev")
END_TARGET_DIR=$((END_STEP - OFFSET))
while [[ $target -le $END_TARGET_DIR ]]; do
    prev_ckpt="$OUTPUT_DIR/checkpoint-$prev"
    target_ckpt="$OUTPUT_DIR/checkpoint-$target"

    # First chunk with LOAD_LORA_FROM: no prev_ckpt expected (we seed externally).
    if [[ "$prev" != "0" && ! -d "$prev_ckpt" ]]; then
        echo "[chunk] ERROR: prev ckpt $prev_ckpt missing. abort."
        exit 1
    fi

    # If target ckpt already exists, skip BOTH train AND eval.  Eval only the
    # ckpts this run produces — old ckpts have CSV rows already.
    if [[ -d "$target_ckpt" ]]; then
        echo "[chunk] ckpt-$target exists, skipping (eval already in CSV)"
        prev=$target
        target=$(compute_target "$prev")
        continue
    fi

    cum_prev=$((prev + OFFSET))
    cum_target=$((target + OFFSET))
    echo
    echo "============================================================"
    echo "[chunk] TRAIN ckpt-$prev → ckpt-$target  (cumulative $cum_prev → $cum_target)"
    echo "============================================================"

    kill_servers

    # First chunk + LOAD_LORA_FROM → pass --load_lora to train.py for fresh
    # AdamW + LoRA-seeded init.  Subsequent chunks: normal Trainer auto-resume.
    EXTRA_TRAIN_ARGS=()
    if [[ "$prev" == "0" && -n "$LOAD_LORA_FROM" ]]; then
        EXTRA_TRAIN_ARGS+=(--load_lora "$LOAD_LORA_FROM")
        echo "[chunk]   first-chunk LoRA seed: $LOAD_LORA_FROM"
    fi

    # NO explicit RESUME — let train.py auto-resume from the latest complete
    # ckpt in output_dir (for chunks 2+).  watchdog re-runs on crash.
    MAX_STEPS="$target" MAX_RETRIES=20 \
        bash "$REPO_ROOT/LeIsaac/scripts/finetune/openvla/watchdog.sh" \
            --save_steps 100 --save_total_limit 200 \
            "${EXTRA_TRAIN_ARGS[@]}" \
            2>&1 | tail -25
    if [[ ! -d "$target_ckpt" ]]; then
        echo "[chunk] ERROR: ckpt-$target not produced after watchdog retries. abort."
        exit 2
    fi

    # Skip eval for early ckpts (model under-trained) — still trained, just no GUI eval.
    if [[ "$cum_target" -lt "$START_EVAL_STEP" ]]; then
        echo
        echo "[chunk] SKIP EVAL ckpt-$target (cumulative $cum_target < START_EVAL_STEP=$START_EVAL_STEP)"
        prev=$target
        target=$(compute_target "$prev")
        continue
    fi

    echo
    echo "[chunk] EVAL ckpt-$target (cumulative $cum_target)  (GUI, $EVAL_ROUNDS × ${EPISODE_S}s)"

    kill_servers
    ADAPTER="$target_ckpt" PORT="$POLICY_PORT" \
        bash "$REPO_ROOT/server/serve_openvla.sh" --detach 2>&1 | tail -3
    sleep 2

    eval_log="$REPO_ROOT/logs/openvla_chunked_eval_${cum_target}.log"
    POLICY_PORT="$POLICY_PORT" POLICY_TIMEOUT_MS=5000 ACTION_HORIZON=1 \
        EVAL_ROUNDS="$EVAL_ROUNDS" EPISODE_LENGTH="$EPISODE_S" PROMPT="$PROMPT" \
        MAX_ROUND_WALL_S="$MAX_ROUND_WALL_S" \
        timeout 600 bash "$REPO_ROOT/server/eval_pi05.sh" 2>&1 | \
        tee "$eval_log" | grep -E "Evaluating episode|placed|Final|oranges:|round" | tail -15 || true

    final_line=$(grep "Final success rate" "$eval_log" | tail -1 || echo "")
    if [[ -z "$final_line" ]]; then
        echo "$cum_target,$EVAL_ROUNDS,0,0,$((EVAL_ROUNDS * 3)),NaN,no-Final-line" >> "$CSV"
        echo "  ⚠️  ckpt-$target (cum $cum_target): no Final line parsed"
    else
        success=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f1)
        rt=$(echo "$final_line" | grep -oE "\[[0-9]+/[0-9]+\]" | head -1 | tr -d '[]' | cut -d/ -f2)
        placed=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f1)
        total=$(echo "$final_line" | grep -oE "oranges: [0-9]+/[0-9]+" | grep -oE "[0-9]+/[0-9]+" | cut -d/ -f2)
        avg=$(echo "$final_line" | grep -oE "avg_round_s: [0-9.]+" | grep -oE "[0-9.]+")
        echo "$cum_target,$rt,$success,$placed,$total,$avg,\"$final_line\"" >> "$CSV"
        echo "  ➜ ckpt=$target (cum $cum_target)  success=$success/$rt  oranges=$placed/$total  avg=${avg}s"
    fi

    column -s, -t < "$CSV"

    prev=$target
    target=$(compute_target "$prev")
done

kill_servers
echo
echo "[chunk] DONE.  Results:"
column -s, -t < "$CSV"
echo
echo "[chunk] Best ckpt(s) by oranges_placed DESC → rounds_success DESC → avg_round_s ASC:"
tail -n +2 "$CSV" | sort -t, -k4,4nr -k3,3nr -k6,6n | head -3 | \
    awk -F, '{printf "  step %s  oranges=%s/%s  rounds=%s/%s  avg=%ss\n", $1, $4, $5, $3, $2, $6}'
