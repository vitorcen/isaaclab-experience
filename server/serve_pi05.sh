#!/usr/bin/env bash
# Start the π0.5 PyTorch ZMQ inference server (NVIDIA GPU).
#
# Wire-compatible with the LeIsaac Pi05ServicePolicyClient: ZMQ REQ/REP +
# msgpack with `__ndarray__` ndarray encoding.
#
# Usage:
#   bash server/serve_pi05.sh [--detach] [extra server args...]
#
# Knobs (env vars):
#   CONDA_ENV       conda env w/ lerobot + torch + cuda      (default: lerobot)
#   PORT            listen port                              (default: 5556)
#   BIND_HOST            listen host                              (default: 127.0.0.1)
#   DTYPE           float32 | bfloat16 | float16             (default: bfloat16)
#   LORA_NPZ        LoRA weights path                        (required if not via flag)
#   DATASET_ROOT    Local v3.0 dataset (for stats + features) (default: ~/work/LeIsaac/datasets/raw/leisaac-pick-orange)
#   LEROBOT_SRC     editable lerobot src dir                 (default: ~/work/lerobot-experience/lerobot/src)
#
# All trailing args are forwarded to `python -m pi05_leisaac.server`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot}"
PORT="${PORT:-5556}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
DTYPE="${DTYPE:-bfloat16}"
LORA_NPZ="${LORA_NPZ:-${HOME}/work/isaaclab-experience/LeIsaac/outputs/pi05-leisaac-pt-v3/final_lora.npz}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/work/isaaclab-experience/LeIsaac/datasets/raw/leisaac-pick-orange}"
LEROBOT_SRC="${LEROBOT_SRC:-${HOME}/work/lerobot-experience/lerobot/src}"
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
export LEROBOT_SRC

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pi05_server.log"
PID_FILE="${LOG_DIR}/pi05_server.pid"

if [[ ! -f "${LORA_NPZ}" ]]; then
    echo "[pi05] ERROR: LoRA weights not found: ${LORA_NPZ}" >&2
    echo "[pi05] hint: set LORA_NPZ=... or pass --lora-npz <path>" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "[pi05] ERROR: dataset root not found: ${DATASET_ROOT}" >&2
    exit 1
fi

CMD=(
    python -u -m pi05_leisaac.server
    --host "${BIND_HOST}" --port "${PORT}" --dtype "${DTYPE}"
    --lora-npz "${LORA_NPZ}"
    --dataset-root "${DATASET_ROOT}"
    "${EXTRA[@]}"
)

echo "[pi05] launching server: ${CMD[*]}"
echo "[pi05] log: ${LOG_FILE}"

if [[ "${DETACH}" -eq 1 ]]; then
    nohup conda run -n "${CONDA_ENV}" --no-capture-output "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "[pi05] pid=$(cat "${PID_FILE}") bound to ${BIND_HOST}:${PORT}"
    # tail until "ready" or error
    until grep -qE "ready|error|Traceback" "${LOG_FILE}" 2>/dev/null; do sleep 2; done
    tail -5 "${LOG_FILE}"
else
    exec conda run -n "${CONDA_ENV}" --no-capture-output "${CMD[@]}"
fi
