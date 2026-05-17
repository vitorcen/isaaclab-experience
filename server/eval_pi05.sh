#!/usr/bin/env bash
# Run a LeIsaac PickOrange eval against a running π0.5 inference server.
#
# Assumes:
#   - server/serve_pi05.sh is already running (default host 127.0.0.1:5556)
#   - the isaaclab conda env has Isaac Sim + LeIsaac sources
#
# Usage:
#   bash server/eval_pi05.sh                          # 3×60s default
#   EVAL_ROUNDS=10 EPISODE_LENGTH=120 bash server/eval_pi05.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5556}"
POLICY_TIMEOUT_MS="${POLICY_TIMEOUT_MS:-10000}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
EPISODE_LENGTH="${EPISODE_LENGTH:-60}"
PROMPT="${PROMPT:-Pick up the orange and place it on the plate}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
LEISAAC_DIR="${LEISAAC_DIR:-${REPO_ROOT}/LeIsaac}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pi05_eval-$(date +%Y%m%d-%H%M%S).log"

echo "[eval] server:  tcp://${POLICY_HOST}:${POLICY_PORT}"
echo "[eval] task:    LeIsaac-SO101-PickOrange-v0 (${EVAL_ROUNDS}×${EPISODE_LENGTH}s)"
echo "[eval] log:     ${LOG_FILE}"
echo "[eval] leisaac: ${LEISAAC_DIR}"

cd "${LEISAAC_DIR}"
DISPLAY="${DISPLAY:-:0}" PYTHONUNBUFFERED=1 \
conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u scripts/evaluation/policy_inference.py \
        --task=LeIsaac-SO101-PickOrange-v0 \
        --eval_rounds="${EVAL_ROUNDS}" \
        --episode_length_s="${EPISODE_LENGTH}" \
        --policy_type=pi05 \
        --policy_host="${POLICY_HOST}" --policy_port="${POLICY_PORT}" \
        --policy_timeout_ms="${POLICY_TIMEOUT_MS}" \
        --policy_action_horizon="${ACTION_HORIZON}" \
        --policy_language_instruction="${PROMPT}" \
        --device=cuda --enable_cameras 2>&1 | tee "${LOG_FILE}"
