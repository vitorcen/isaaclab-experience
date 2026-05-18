#!/usr/bin/env bash
# Launch FastWAM QLoRA inference server (post-finetune ckpt).
# Wire-compatible with eval_pi05.sh / Pi05ServicePolicyClient on PORT 5559.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-fastwam}"
PORT="${PORT:-5559}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
CKPT_DIR="${CKPT_DIR:-/home/david/work/fastwam-repo/runs/train/fastwam_qlora_pickorange_5phase/phase2/checkpoints/state/step_004000}"
PROMPT="${PROMPT:-Grab orange and place into plate}"
ARM_DELTA_SCALE="${ARM_DELTA_SCALE:-0.05}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"  # MUST match training (num_frames-1=32 divisibility)
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
DETACH=0

EXTRA=()
for arg in "$@"; do
    if [[ "$arg" == "--detach" ]]; then DETACH=1
    else EXTRA+=("$arg")
    fi
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FASTWAM_REPO_ROOT="${FASTWAM_REPO_ROOT:-$HOME/work/fastwam-repo}"
export PYTHONPATH="${REPO_ROOT}/server:${REPO_ROOT}/LeIsaac/scripts/finetune:${FASTWAM_REPO_ROOT}/src:${PYTHONPATH:-}"

LOG_DIR="${REPO_ROOT}/logs"; mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/fastwam_qlora_server.log"
PID_FILE="${LOG_DIR}/fastwam_qlora_server.pid"

CMD=(
    python -u -m fastwam_leisaac.server_qlora
    --host "${BIND_HOST}" --port "${PORT}"
    --ckpt-dir "${CKPT_DIR}"
    --prompt "${PROMPT}"
    --arm-delta-scale "${ARM_DELTA_SCALE}"
    --action-horizon "${ACTION_HORIZON}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    "${EXTRA[@]}"
)

echo "[fastwam-qlora] launching: ${CMD[*]}"
echo "[fastwam-qlora] log: ${LOG_FILE}"

if [[ "${DETACH}" -eq 1 ]]; then
    nohup conda run -n "${CONDA_ENV}" --no-capture-output "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "[fastwam-qlora] pid=$(cat "${PID_FILE}") binding to ${BIND_HOST}:${PORT}..."
    until grep -qE "ready, listening|error|Traceback" "${LOG_FILE}" 2>/dev/null; do sleep 2; done
    tail -10 "${LOG_FILE}"
else
    exec conda run -n "${CONDA_ENV}" --no-capture-output "${CMD[@]}"
fi
