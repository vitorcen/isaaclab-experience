#!/usr/bin/env bash
# Start the X-VLA inference ZMQ server.
#
# Knobs (env vars):
#   CONDA_ENV  conda env (default: lerobot)
#   PORT       listen port (default: 5558)
#   BIND_HOST  listen host (default: 127.0.0.1)
#   CKPT       ckpt dir (default: outputs/xvla-leisaac-pick-orange/checkpoints/last/pretrained_model)
#   PROMPT     fallback prompt
#
# Usage:
#   bash server/serve_xvla.sh [--detach]
#   CKPT=/path/to/pretrained_model bash server/serve_xvla.sh --detach
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot}"
PORT="${PORT:-5558}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
CKPT="${CKPT:-$REPO_ROOT/LeIsaac/outputs/xvla-leisaac-pick-orange/checkpoints/last/pretrained_model}"
PROMPT="${PROMPT:-Pick up the orange and put it in the plate}"  # 50% per-ep on 8k h=32, see vla_improvement_methods_checklist.html §0.1
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
EMA_ALPHA="${EMA_ALPHA:-}"
TAE_BUFFER="${TAE_BUFFER:-}"
TAE_M="${TAE_M:-}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-}"
DETACH=0

for arg in "$@"; do
    [[ "$arg" == "--detach" ]] && DETACH=1
done

export PYTHONPATH="$REPO_ROOT/server:${PYTHONPATH:-}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/xvla_server.log"

CMD=(python -u -m xvla_leisaac.server
     --host "$BIND_HOST" --port "$PORT"
     --ckpt "$CKPT" --prompt "$PROMPT")
[[ -n "$N_ACTION_STEPS" ]] && CMD+=(--n-action-steps "$N_ACTION_STEPS")
[[ -n "$EMA_ALPHA" ]] && CMD+=(--ema-alpha "$EMA_ALPHA")
[[ -n "$TAE_BUFFER" ]] && CMD+=(--tae-buffer "$TAE_BUFFER")
[[ -n "$TAE_M" ]] && CMD+=(--tae-m "$TAE_M")
[[ -n "$NUM_DENOISING_STEPS" ]] && CMD+=(--num-denoising-steps "$NUM_DENOISING_STEPS")

echo "[xvla] launching server: ${CMD[*]}"
echo "[xvla] log: $LOG_FILE"

if (( DETACH )); then
    conda run -n "$CONDA_ENV" --no-capture-output "${CMD[@]}" >"$LOG_FILE" 2>&1 &
    pid=$!
    echo "[xvla] pid=$pid binding to $BIND_HOST:$PORT..."
    # Wait for "ready" marker or pid death (timeout 60s)
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[xvla] ❌ server pid $pid died — see $LOG_FILE"
            tail -20 "$LOG_FILE"
            exit 1
        fi
        if grep -q "listening on tcp" "$LOG_FILE" 2>/dev/null; then
            tail -10 "$LOG_FILE"
            exit 0
        fi
        sleep 1
    done
    echo "[xvla] ⚠️  timed out waiting for ready marker; check $LOG_FILE"
    tail -20 "$LOG_FILE"
else
    exec conda run -n "$CONDA_ENV" --no-capture-output "${CMD[@]}"
fi
