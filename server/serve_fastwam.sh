#!/usr/bin/env bash
# Start the FastWAM (Wan2.2-5B + 1B action expert) demo ZMQ inference server.
#
# Wire-compatible with LeIsaac Pi05ServicePolicyClient (ZMQ REQ/REP + msgpack
# __ndarray__). See server/fastwam_leisaac/README.md for the cosmetic remap.
#
# Usage:
#   bash server/serve_fastwam.sh [--detach] [extra args]
#
# Knobs (env vars):
#   CONDA_ENV          conda env w/ fastwam deps     (default: fastwam)
#   PORT               listen port                    (default: 5559)
#   BIND_HOST          listen host                    (default: 127.0.0.1)
#   CKPT               ckpt name in yuanty/fastwam    (default: libero_uncond_2cam224.pt)
#   PROMPT             fallback prompt                (default: "Pick up the orange...")
#   ARM_DELTA_SCALE    EEF→joint Δ safety factor      (default: 0.05)
#   ACTION_HORIZON     chunk size                      (default: 24)
#   NUM_INFERENCE_STEPS  flow-matching denoise steps  (default: 10)
#   FASTWAM_REPO_ROOT  path to fastwam repo           (default: ~/work/fastwam-repo)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-fastwam}"
PORT="${PORT:-5559}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
CKPT="${CKPT:-libero_uncond_2cam224.pt}"
PROMPT="${PROMPT:-Pick up the orange and place it on the plate}"
ARM_DELTA_SCALE="${ARM_DELTA_SCALE:-0.05}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
DETACH=0

EXTRA=()
for arg in "$@"; do
    if [[ "$arg" == "--detach" ]]; then
        DETACH=1
    else
        EXTRA+=("$arg")
    fi
done

# Lean GPU allocator so Isaac Sim can co-exist on the same 24GB card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FASTWAM_REPO_ROOT="${FASTWAM_REPO_ROOT:-$HOME/work/fastwam-repo}"
export PYTHONPATH="${REPO_ROOT}/server:${PYTHONPATH:-}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/fastwam_server.log"
PID_FILE="${LOG_DIR}/fastwam_server.pid"

CMD=(
    python -u -m fastwam_leisaac.server
    --host "${BIND_HOST}" --port "${PORT}"
    --ckpt "${CKPT}"
    --prompt "${PROMPT}"
    --arm-delta-scale "${ARM_DELTA_SCALE}"
    --action-horizon "${ACTION_HORIZON}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    "${EXTRA[@]}"
)

echo "[fastwam] launching server: ${CMD[*]}"
echo "[fastwam] log: ${LOG_FILE}"

if [[ "${DETACH}" -eq 1 ]]; then
    nohup conda run -n "${CONDA_ENV}" --no-capture-output "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "[fastwam] pid=$(cat "${PID_FILE}") binding to ${BIND_HOST}:${PORT}..."
    until grep -qE "ready|error|Traceback" "${LOG_FILE}" 2>/dev/null; do sleep 2; done
    tail -10 "${LOG_FILE}"
else
    exec conda run -n "${CONDA_ENV}" --no-capture-output "${CMD[@]}"
fi
