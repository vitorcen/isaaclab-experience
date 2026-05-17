#!/usr/bin/env bash
# Start the OpenVLA-7B 4-bit demo ZMQ inference server.
#
# Wire-compatible with LeIsaac Pi05ServicePolicyClient (ZMQ REQ/REP + msgpack
# __ndarray__). See server/openvla_leisaac/README.md for the action-space hack.
#
# Usage:
#   bash server/serve_openvla.sh [--detach] [extra args forwarded to server]
#
# Knobs (env vars):
#   CONDA_ENV          conda env w/ openvla deps      (default: openvla)
#   PORT               listen port                    (default: 5557)
#   BIND_HOST          listen host                    (default: 127.0.0.1)
#   MODEL_NAME         HF repo id                     (default: openvla/openvla-7b)
#   UNNORM_KEY         action unnormalization stats   (default: bridge_orig)
#   PROMPT             fallback prompt                (default: "Pick up the orange...")
#   ARM_DELTA_SCALE    EEF→joint Δ safety factor      (default: 0.05)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-openvla}"
PORT="${PORT:-5557}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-openvla/openvla-7b}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
PROMPT="${PROMPT:-Pick up the orange and place it on the plate}"
ARM_DELTA_SCALE="${ARM_DELTA_SCALE:-0.05}"
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
# Make the package importable when run from a fresh shell (no install).
export PYTHONPATH="${REPO_ROOT}/server:${PYTHONPATH:-}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/openvla_server.log"
PID_FILE="${LOG_DIR}/openvla_server.pid"

CMD=(
    python -u -m openvla_leisaac.server
    --host "${BIND_HOST}" --port "${PORT}"
    --model-name "${MODEL_NAME}"
    --unnorm-key "${UNNORM_KEY}"
    --prompt "${PROMPT}"
    --arm-delta-scale "${ARM_DELTA_SCALE}"
    "${EXTRA[@]}"
)

echo "[openvla] launching server: ${CMD[*]}"
echo "[openvla] log: ${LOG_FILE}"

if [[ "${DETACH}" -eq 1 ]]; then
    nohup conda run -n "${CONDA_ENV}" --no-capture-output "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "[openvla] pid=$(cat "${PID_FILE}") binding to ${BIND_HOST}:${PORT}..."
    until grep -qE "ready|error|Traceback" "${LOG_FILE}" 2>/dev/null; do sleep 2; done
    tail -5 "${LOG_FILE}"
else
    exec conda run -n "${CONDA_ENV}" --no-capture-output "${CMD[@]}"
fi
