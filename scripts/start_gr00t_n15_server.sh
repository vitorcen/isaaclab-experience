#!/usr/bin/env bash
# Launch the GR00T N1.5 inference server (ZMQ :5555) with the LightwheelAI
# fine-tuned ckpt for LeIsaac SO-101 PickOrange. Idempotent.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(cd "$ROOT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

GR00T_N15_DIR="${GR00T_N15_DIR:-$WORK_DIR/Isaac-GR00T-N1.5}"
GR00T_N15_PYTHON="${GR00T_N15_PYTHON:-$HOME/miniconda3/envs/gr00t-n15/bin/python}"
CKPT_DIR="${LEISAAC_N15_CKPT_DIR:-$HOME/models/leisaac-pick-orange-v0}"
HOST="${GR00T_N15_HOST:-0.0.0.0}"
PORT="${GR00T_N15_PORT:-5555}"

port_listening() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${1}\$"
    else
        netstat -an 2>/dev/null | grep -E "[.:]${1}[[:space:]]+" | grep -qi LISTEN
    fi
}

if port_listening "$PORT"; then
    echo "[INFO] GR00T N1.5 server already listening on :$PORT"
    exit 0
fi

[ -d "$GR00T_N15_DIR/gr00t" ] || { echo "[ERROR] GR00T N1.5 repo not found: $GR00T_N15_DIR" >&2; exit 1; }
[ -x "$GR00T_N15_PYTHON" ]    || { echo "[ERROR] gr00t-n15 python not found: $GR00T_N15_PYTHON" >&2; exit 1; }
[ -f "$CKPT_DIR/config.json" ] || { echo "[ERROR] ckpt missing: $CKPT_DIR (run scripts/download_gr00t_n15_ckpt.sh)" >&2; exit 1; }

DETACH=()
command -v setsid >/dev/null 2>&1 && DETACH=(setsid)

echo "[INFO] launching GR00T N1.5 server: model=$CKPT_DIR port=$PORT"
nohup "${DETACH[@]}" bash -lc "cd '$GR00T_N15_DIR' && PYTHONPATH='$GR00T_N15_DIR' '$GR00T_N15_PYTHON' scripts/inference_service.py --server --model-path '$CKPT_DIR' --embodiment-tag new_embodiment --data-config so100_dualcam --host '$HOST' --port '$PORT'" \
    > "$LOG_DIR/gr00t_n15_server.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/gr00t_n15_server.pid"
echo "[INFO] pid=$(cat "$LOG_DIR/gr00t_n15_server.pid"), bind=$HOST:$PORT"

# Loading the ckpt + Eagle backbone takes ~20-30s.
for i in $(seq 1 60); do
    if port_listening "$PORT"; then
        echo "[INFO] GR00T N1.5 server listening on :$PORT"
        exit 0
    fi
    sleep 1
done

echo "[ERROR] server did not come up within 60s; last log lines:" >&2
tail -n 30 "$LOG_DIR/gr00t_n15_server.log" >&2 || true
exit 1
