#!/usr/bin/env bash
# Start DreamZero policy server (single 4090 24G NF4 + LoRA).
#
# Usage:
#   bash server/dreamzero_leisaac/start.sh --ckpt-path /path/to/checkpoint-1000
#   PORT=5556 bash server/dreamzero_leisaac/start.sh --ckpt-path ...
#
# Phase 1: STUB mode (hold-pose), validates ZMQ + msgpack wire protocol.
# Phase 2: real Wan2.1 forward — see server.py TODOs.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CKPT_PATH=""
ACTION_HORIZON="${ACTION_HORIZON:-24}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5556}"
CONDA_ENV="${CONDA_ENV:-dreamzero}"  # phase-2 real forward requires dreamzero env

# Parse --ckpt-path arg
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-path) CKPT_PATH="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --action-horizon) ACTION_HORIZON="$2"; shift 2;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "${CKPT_PATH}" ]]; then
    # Default to most recent ckpt in standard outputs dir
    CKPT_PATH=$(ls -td "${REPO_ROOT}/LeIsaac/outputs/dreamzero-leisaac-so101-lora-r4"/checkpoint-* 2>/dev/null | head -1)
    if [[ -z "${CKPT_PATH}" ]]; then
        echo "ERROR: no --ckpt-path given and no checkpoints under LeIsaac/outputs/dreamzero-leisaac-so101-lora-r4/"
        exit 1
    fi
    echo "[start] auto-detected --ckpt-path ${CKPT_PATH}"
fi

echo "[start] ckpt:           ${CKPT_PATH}"
echo "[start] listening on:   tcp://${HOST}:${PORT}"
echo "[start] action_horizon: ${ACTION_HORIZON}"

PYTHONUNBUFFERED=1 conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u "${REPO_ROOT}/server/dreamzero_leisaac/server.py" \
        --ckpt-path "${CKPT_PATH}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --action-horizon "${ACTION_HORIZON}"
