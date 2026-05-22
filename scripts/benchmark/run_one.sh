#!/usr/bin/env bash
# Run a single LeIsaac PickOrange baseline end-to-end:
#   1. start the required policy server (lerobot async / gr00t-n15 / gr00t-n16)
#   2. start an nvidia-smi sampler in the background
#   3. run policy_inference.py with --metrics_out
#   4. stop the sampler + (for gr00t) stop the server so the next baseline can boot
#
# Inputs (positional):
#   SLUG           short id (used for output filenames)
#   POLICY_TYPE    e.g. lerobot-act / lerobot-diffusion / lerobot-smolvla / gr00tn1.5 / gr00tn1.6
#   HORIZON        --policy_action_horizon
#   CKPT           HF repo_id or local path
#   SERVER_KIND    lerobot | gr00t-n15 | gr00t-n16
#   LABEL          display label for the metrics JSON
#
# Env overrides:
#   EVAL_ROUNDS=3 EPISODE_LENGTH_S=120 STEP_HZ=30 RESULTS_DIR=.../results
#
# Exit non-zero on any setup/eval failure; the orchestrator will tag the slot
# as missing in the aggregate.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEISAAC_DIR="${LEISAAC_DIR:-$ROOT_DIR/LeIsaac}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/benchmark}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
# Make RESULTS_DIR absolute since policy_inference.py runs with cwd=$LEISAAC_DIR
# and would otherwise resolve --metrics_out relative to LeIsaac/.
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

SLUG="${1:?slug required}"
POLICY_TYPE="${2:?policy_type required}"
HORIZON="${3:?horizon required}"
CKPT="${4:?ckpt required}"
SERVER_KIND="${5:?server_kind required}"
LABEL="${6:-$SLUG}"

EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-120}"
STEP_HZ="${STEP_HZ:-30}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-180}"
SIM_WARMUP_STEPS="${SIM_WARMUP_STEPS:-30}"
# ACT/DP can pause/re-plan within a chunk; stuck detector tends to false-trip
# and skip otherwise-recoverable episodes (saw 60k eval ep2 cut at 44.9s).
# Episode_length_s is the natural cap. Override per-call if needed.
case "${POLICY_TYPE}" in
    lerobot-act|lerobot-diffusion)
        STUCK_WINDOW_S="${STUCK_WINDOW_S:-99999}"
        STUCK_EPS_RAD="${STUCK_EPS_RAD:-0}"
        ;;
    *)
        STUCK_WINDOW_S="${STUCK_WINDOW_S:-30}"
        STUCK_EPS_RAD="${STUCK_EPS_RAD:-0.05}"
        ;;
esac
CONDA_ENV="${CONDA_ENV:-isaaclab}"
LEROBOT_HOST="${LEROBOT_HOST:-127.0.0.1}"
LEROBOT_PORT="${LEROBOT_PORT:-8080}"
GR00T_HOST="${GR00T_HOST:-127.0.0.1}"
GR00T_PORT="${GR00T_PORT:-5555}"
PROMPT="${PROMPT:-Pick up the orange and place it on the plate}"

EVAL_LOG="$LOG_DIR/bench-${SLUG}.log"
METRICS_JSON="$RESULTS_DIR/${SLUG}.metrics.json"
GPU_CSV="$RESULTS_DIR/${SLUG}.gpu.csv"
GPU_PIDFILE="$RESULTS_DIR/.${SLUG}.gpu.pid"
SUMMARY="$RESULTS_DIR/${SLUG}.summary.txt"

echo "[bench] === ${LABEL} ==="
echo "[bench] slug=${SLUG} policy_type=${POLICY_TYPE} horizon=${HORIZON} ckpt=${CKPT}"
echo "[bench] server_kind=${SERVER_KIND} rounds=${EVAL_ROUNDS} ep_len=${EPISODE_LENGTH_S}s"
echo "[bench] eval log: ${EVAL_LOG}"
echo "[bench] metrics:  ${METRICS_JSON}"

# --- 1. server start ---------------------------------------------------------
SERVER_HOST="$LEROBOT_HOST"
SERVER_PORT="$LEROBOT_PORT"
case "$SERVER_KIND" in
    lerobot)
        bash "$ROOT_DIR/scripts/policy_server.sh" start lerobot
        ;;
    gr00t-n15)
        # Free :5555 first — N1.5 and N1.6 share the port.
        bash "$ROOT_DIR/scripts/policy_server.sh" stop gr00t-n16 || true
        bash "$ROOT_DIR/scripts/policy_server.sh" stop gr00t-n15 || true
        bash "$ROOT_DIR/scripts/policy_server.sh" start gr00t-n15 "$CKPT"
        SERVER_HOST="$GR00T_HOST"; SERVER_PORT="$GR00T_PORT"
        ;;
    gr00t-n16)
        bash "$ROOT_DIR/scripts/policy_server.sh" stop gr00t-n15 || true
        bash "$ROOT_DIR/scripts/policy_server.sh" stop gr00t-n16 || true
        GR00T_MODEL_PATH="$CKPT" bash "$ROOT_DIR/scripts/policy_server.sh" start gr00t-n16
        SERVER_HOST="$GR00T_HOST"; SERVER_PORT="$GR00T_PORT"
        ;;
    pi05)
        # π0.5 PyTorch server on :5556 (does not conflict with GR00T :5555 / lerobot :8080)
        pkill -9 -f "pi05_leisaac.server" 2>/dev/null || true
        sleep 2
        LORA_NPZ="$CKPT" bash "$ROOT_DIR/server/serve_pi05.sh" --detach
        SERVER_HOST="127.0.0.1"; SERVER_PORT="5556"
        ;;
    *)
        echo "[bench] unknown server_kind: $SERVER_KIND" >&2
        exit 2
        ;;
esac

# --- 2. GPU sampler in background -------------------------------------------
bash "$ROOT_DIR/scripts/benchmark/gpu_sampler.sh" "$GPU_CSV" "$GPU_PIDFILE" 1 &
GPU_BG_PID=$!
trap '
  set +e
  rm -f "$GPU_PIDFILE" 2>/dev/null
  wait "$GPU_BG_PID" 2>/dev/null
  if [[ "$SERVER_KIND" == "gr00t-n15" || "$SERVER_KIND" == "gr00t-n16" ]]; then
      bash "$ROOT_DIR/scripts/policy_server.sh" stop "$SERVER_KIND" || true
  fi
  if [[ "$SERVER_KIND" == "pi05" ]]; then
      pkill -9 -f "pi05_leisaac.server" 2>/dev/null || true
  fi
' EXIT

# --- 3. run eval -------------------------------------------------------------
# For lerobot policies the client loads the ckpt; for gr00t* the server loads it.
EXTRA_ARGS=()
case "$POLICY_TYPE" in
    lerobot-*)
        EXTRA_ARGS+=(--policy_checkpoint_path="$CKPT")
        ;;
esac

cd "$LEISAAC_DIR"
set +e
DISPLAY="${DISPLAY:-:0}" PYTHONUNBUFFERED=1 \
    timeout $((120 + EVAL_ROUNDS * EPISODE_LENGTH_S * 2)) \
    conda run -n "$CONDA_ENV" --no-capture-output \
        python -u scripts/evaluation/policy_inference.py \
            --task=LeIsaac-SO101-PickOrange-v0 \
            --eval_rounds="$EVAL_ROUNDS" \
            --episode_length_s="$EPISODE_LENGTH_S" \
            --step_hz="$STEP_HZ" \
            --policy_type="$POLICY_TYPE" \
            --policy_host="$SERVER_HOST" --policy_port="$SERVER_PORT" \
            --policy_timeout_ms=30000 \
            --policy_action_horizon="$HORIZON" \
            --policy_language_instruction="$PROMPT" \
            --max_round_wall_s="$MAX_ROUND_WALL_S" \
            --sim_warmup_steps="$SIM_WARMUP_STEPS" \
            --stuck_window_s="$STUCK_WINDOW_S" \
            --stuck_eps_rad="$STUCK_EPS_RAD" \
            --metrics_out="$METRICS_JSON" \
            --metrics_label="$LABEL" \
            "${EXTRA_ARGS[@]}" \
            --device=cuda --enable_cameras 2>&1 | tee "$EVAL_LOG"
EVAL_EXIT=${PIPESTATUS[0]}
set -e

# Free Isaac Sim GPU memory before next baseline (Isaac doesn't release on its own).
pkill -9 -f "scripts/evaluation/policy_inference.py" 2>/dev/null || true
sleep 3

# Summary line for orchestrator console
if [[ -f "$METRICS_JSON" ]]; then
    python3 - "$METRICS_JSON" <<'PY' | tee "$SUMMARY"
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  ✓ {m['rounds_success']}/{m['rounds']} rounds, "
      f"{m['oranges_placed_total']}/{m['oranges_max_total']} oranges, "
      f"avg {m['avg_round_s']}s/round")
PY
else
    echo "  ✗ no metrics produced (exit=$EVAL_EXIT)" | tee "$SUMMARY"
fi

exit "$EVAL_EXIT"
