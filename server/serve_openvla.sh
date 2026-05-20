#!/usr/bin/env bash
# Start the OpenVLA-7B 4-bit ZMQ inference server.
#
# Two modes:
#   - Demo (no ADAPTER):    base openvla/openvla-7b with bridge_orig stats + Δ→joint hack.
#   - Finetuned (ADAPTER):  loads our LoRA on top, uses leisaac stats + canonical prompt
#                           from <ADAPTER>/dataset_statistics.json (falls back to
#                           <ADAPTER>/../dataset_statistics.json for ckpt-N layouts).
#
# Wire-compatible with LeIsaac Pi05ServicePolicyClient (ZMQ REQ/REP + msgpack __ndarray__).
#
# Usage:
#   bash server/serve_openvla.sh [--detach] [extra args forwarded to server]
#   ADAPTER=/path/to/checkpoint-N bash server/serve_openvla.sh --detach
#
# Knobs (env vars):
#   CONDA_ENV          conda env w/ openvla deps      (default: openvla)
#   PORT               listen port                    (default: 5557)
#   BIND_HOST          listen host                    (default: 127.0.0.1)
#   MODEL_NAME         HF base repo id                (default: openvla/openvla-7b)
#   ADAPTER            LoRA adapter dir               (default: empty → demo mode)
#   UNNORM_KEY         override unnorm_key            (default: leisaac if ADAPTER else bridge_orig)
#   PROMPT             override fallback prompt       (default: canonical from stats file / demo string)
#   ARM_DELTA_SCALE    demo-only EEF→joint Δ scale    (default: 0.05)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-openvla}"
PORT="${PORT:-5557}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-openvla/openvla-7b}"
ADAPTER="${ADAPTER:-}"
UNNORM_KEY="${UNNORM_KEY:-}"     # empty → server picks leisaac/bridge_orig based on ADAPTER
PROMPT="${PROMPT:-}"             # empty → server reads canonical_prompt from stats
ARM_DELTA_SCALE="${ARM_DELTA_SCALE:-0.05}"
QUANT="${QUANT:-8bit}"           # 4bit | 8bit | bf16 — must match training
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
    --arm-delta-scale "${ARM_DELTA_SCALE}"
    --quant "${QUANT}"
)
[[ -n "${ADAPTER}"    ]] && CMD+=( --adapter   "${ADAPTER}"    )
[[ -n "${UNNORM_KEY}" ]] && CMD+=( --unnorm-key "${UNNORM_KEY}" )
[[ -n "${PROMPT}"     ]] && CMD+=( --prompt    "${PROMPT}"     )
CMD+=( "${EXTRA[@]}" )

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
