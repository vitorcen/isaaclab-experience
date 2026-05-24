#!/usr/bin/env bash
# Run a LeIsaac PickOrange eval against a running DreamZero (Wan2.1-I2V-14B + LoRA) server.
# Sibling of server/eval_gr00t.sh — same TCP/ZMQ wire flow, dreamzero policy_type.
#
# Assumes:
#   - server/dreamzero_leisaac/server.py is already running (default 127.0.0.1:5556)
#     -> bash server/dreamzero_leisaac/start.sh [--ckpt-path PATH]
#   - the isaaclab conda env has Isaac Sim + LeIsaac sources
#
# Usage:
#   bash server/eval_dreamzero.sh                          # 5×120s default (leaderboard standard)
#   EVAL_ROUNDS=1 EPISODE_LENGTH=60 bash server/eval_dreamzero.sh   # quick smoke
#
# Per [[feedback-dreamzero-eval-stage-criteria]]:
#   - ckpt-1000 (5% of 20k): expect arm to MOVE + tend toward orange. No placed required.
#   - ckpt-5000+: expect occasional placed. Strict success eval only meaningful here.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5556}"
POLICY_TIMEOUT_MS="${POLICY_TIMEOUT_MS:-60000}"   # DreamZero NF4 forward 6-25s/chunk → big timeout
ACTION_HORIZON="${ACTION_HORIZON:-24}"            # DreamZero default action_horizon
EVAL_ROUNDS="${EVAL_ROUNDS:-5}"
EPISODE_LENGTH="${EPISODE_LENGTH:-120}"
MAX_ROUND_WALL_S="${MAX_ROUND_WALL_S:-180}"
PROMPT="${PROMPT:-Pick up the orange and put it in the plate}"
STUCK_WINDOW_S="${STUCK_WINDOW_S:-99999}"         # DreamZero is slow per-chunk; disable stuck-detector
STUCK_EPS_RAD="${STUCK_EPS_RAD:-0.05}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
LEISAAC_DIR="${LEISAAC_DIR:-${REPO_ROOT}/LeIsaac}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/dreamzero_eval-$(date +%Y%m%d-%H%M%S).log"

echo "[eval] server:  tcp://${POLICY_HOST}:${POLICY_PORT}  (dreamzero Wan2.1-14B + LoRA)"
echo "[eval] task:    LeIsaac-SO101-PickOrange-v0 (${EVAL_ROUNDS}×${EPISODE_LENGTH}s, wall_cap=${MAX_ROUND_WALL_S}s)"
echo "[eval] prompt:  ${PROMPT}"
echo "[eval] action_horizon: ${ACTION_HORIZON}"
echo "[eval] log:     ${LOG_FILE}"

cd "${LEISAAC_DIR}"
# Default: disable home/retract detectors for early DreamZero ckpts (action delta too
# small → arm sits at reset pose → immediate retracted_middle / home_return termination).
export LEISAAC_DISABLE_RETRACT_DETECT="${LEISAAC_DISABLE_RETRACT_DETECT:-1}"
DISPLAY="${DISPLAY:-:0}" PYTHONUNBUFFERED=1 \
conda run -n "${CONDA_ENV}" --no-capture-output \
    python -u scripts/evaluation/policy_inference.py \
        --task=LeIsaac-SO101-PickOrange-v0 \
        --eval_rounds="${EVAL_ROUNDS}" \
        --episode_length_s="${EPISODE_LENGTH}" \
        --policy_type=dreamzero \
        --policy_host="${POLICY_HOST}" --policy_port="${POLICY_PORT}" \
        --policy_timeout_ms="${POLICY_TIMEOUT_MS}" \
        --policy_action_horizon="${ACTION_HORIZON}" \
        --policy_language_instruction="${PROMPT}" \
        --max_round_wall_s="${MAX_ROUND_WALL_S}" \
        --stuck_window_s="${STUCK_WINDOW_S}" \
        --stuck_eps_rad="${STUCK_EPS_RAD}" \
        --device=cuda --enable_cameras 2>&1 | tee "${LOG_FILE}"
