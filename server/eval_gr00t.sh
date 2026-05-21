#!/usr/bin/env bash
# Run a LeIsaac PickOrange eval against a running GR00T N1.6 inference server.
# Sibling of server/eval_pi05.sh — same wire-format eval flow, only policy_type
# and default port differ.
#
# Assumes:
#   - server/start_server.sh --gr00t-only is already running (default 127.0.0.1:5555)
#   - the isaaclab conda env has Isaac Sim + LeIsaac sources
#
# Usage:
#   bash server/eval_gr00t.sh                        # 6×60s default
#   EVAL_ROUNDS=3 EPISODE_LENGTH=120 MAX_ROUND_WALL_S=180 bash server/eval_gr00t.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
POLICY_TIMEOUT_MS="${POLICY_TIMEOUT_MS:-10000}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
EVAL_ROUNDS="${EVAL_ROUNDS:-6}"
EPISODE_LENGTH="${EPISODE_LENGTH:-60}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-90}"
PROMPT="${PROMPT:-Pick up the orange and put it in the plate}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
LEISAAC_DIR="${LEISAAC_DIR:-${REPO_ROOT}/LeIsaac}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gr00t_eval-$(date +%Y%m%d-%H%M%S).log"

echo "[eval] server:  tcp://${POLICY_HOST}:${POLICY_PORT}  (gr00tn1.6)"
echo "[eval] task:    LeIsaac-SO101-PickOrange-v0 (${EVAL_ROUNDS}×${EPISODE_LENGTH}s, wall_cap=${MAX_ROUND_WALL_S}s)"
echo "[eval] prompt:  ${PROMPT}"
echo "[eval] log:     ${LOG_FILE}"

cd "${LEISAAC_DIR}"
DISPLAY="${DISPLAY:-:0}" PYTHONUNBUFFERED=1 \
conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u scripts/evaluation/policy_inference.py \
        --task=LeIsaac-SO101-PickOrange-v0 \
        --eval_rounds="${EVAL_ROUNDS}" \
        --episode_length_s="${EPISODE_LENGTH}" \
        --policy_type=gr00tn1.6 \
        --policy_host="${POLICY_HOST}" --policy_port="${POLICY_PORT}" \
        --policy_timeout_ms="${POLICY_TIMEOUT_MS}" \
        --policy_action_horizon="${ACTION_HORIZON}" \
        --policy_language_instruction="${PROMPT}" \
        --max_round_wall_s="${MAX_ROUND_WALL_S}" \
        --device=cuda --enable_cameras 2>&1 | tee "${LOG_FILE}"
