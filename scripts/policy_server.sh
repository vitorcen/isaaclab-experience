#!/usr/bin/env bash
# Manage local VLA policy inference servers used by LeIsaac SO-101 PickOrange.
#
# Usage:
#   bash scripts/policy_server.sh start gr00t-n15 [MODEL_PATH]
#   bash scripts/policy_server.sh start gr00t-n16 [MODEL_PATH]
#   bash scripts/policy_server.sh start lerobot
#   bash scripts/policy_server.sh stop  gr00t-n15
#   bash scripts/policy_server.sh stop  gr00t-n16
#   bash scripts/policy_server.sh stop  lerobot
#
# Backends:
#   gr00t-n15  GR00T N1.5 inference_service.py over ZMQ :5555.
#              The server *loads the checkpoint*. MODEL_PATH = HF repo_id
#              (resolved via from_pretrained against the default HF cache) or
#              absolute local directory.
#              Default: LightwheelAI/leisaac-pick-orange-v0
#              Pre-fetch with: bash scripts/download_hf_model.sh LightwheelAI/leisaac-pick-orange-v0
#              Env overrides: GR00T_N15_DIR / GR00T_N15_PYTHON / GR00T_N15_HOST / GR00T_N15_PORT
#
#   gr00t-n16  GR00T N1.6 run_gr00t_server.py over ZMQ :5555 (shares port with
#              N1.5 — only one can listen). embodiment_tag = NEW_EMBODIMENT
#              (uppercase enum, N1.6-specific).
#              Default MODEL_PATH: hi-space/GR00T-N1.6-3B-Pick-Orange
#              Delegates to server/start_server.sh --gr00t-only via env vars.
#
#   lerobot    LeRobot async-inference policy_server :8080.
#              The *client* (policy_inference.py --policy_checkpoint_path=...)
#              selects which model to load, so no MODEL_PATH here.
#              Delegates to server/start_server.sh --lerobot-only.
#
# Idempotent: start is a no-op if the port already listens; stop is a no-op
# if no server is running.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(cd "$ROOT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

ACTION="${1:-}"
BACKEND="${2:-}"
case "$ACTION:$BACKEND" in
    start:gr00t-n15|start:gr00t-n16|start:gr00t-n17|start:lerobot) ;;
    stop:gr00t-n15|stop:gr00t-n16|stop:gr00t-n17|stop:lerobot) ;;
    *)
        echo "usage: $0 {start|stop} {gr00t-n15|gr00t-n16|gr00t-n17|lerobot} [MODEL_PATH]" >&2
        exit 2
        ;;
esac

port_listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"
    else
        netstat -an 2>/dev/null | grep -E "[.:]${port}[[:space:]]+" | grep -qi LISTEN
    fi
}

# -------- GR00T N1.5 ZMQ :5555 --------
n15_dir="${GR00T_N15_DIR:-$ROOT_DIR/dependencies/Isaac-GR00T-N1.5}"
n15_python="${GR00T_N15_PYTHON:-$HOME/miniconda3/envs/gr00t-n15/bin/python}"
n15_host="${GR00T_N15_HOST:-0.0.0.0}"
n15_port="${GR00T_N15_PORT:-5555}"
n15_pidfile="$LOG_DIR/gr00t_n15_server.pid"
n15_logfile="$LOG_DIR/gr00t_n15_server.log"

start_gr00t_n15() {
    local model_path="${1:-${LEISAAC_N15_CKPT:-LightwheelAI/leisaac-pick-orange-v0}}"

    if port_listening "$n15_port"; then
        echo "[INFO] GR00T N1.5 server already listening on :$n15_port"
        return 0
    fi
    [ -d "$n15_dir/gr00t" ] || { echo "[ERROR] GR00T N1.5 repo not found: $n15_dir" >&2; exit 1; }
    [ -x "$n15_python" ]    || { echo "[ERROR] gr00t-n15 python not found: $n15_python" >&2; exit 1; }
    # Don't pre-validate model_path — from_pretrained accepts both repo_id and
    # local path. If it's a repo_id not in cache, inference_service.py will
    # download it (slow first launch); pre-fetch with download_hf_model.sh.

    local detach=()
    command -v setsid >/dev/null 2>&1 && detach=(setsid)

    echo "[INFO] launching GR00T N1.5 server: model=$model_path port=$n15_port"
    nohup "${detach[@]}" bash -lc "cd '$n15_dir' && PYTHONPATH='$n15_dir' '$n15_python' scripts/inference_service.py --server --model-path '$model_path' --embodiment-tag new_embodiment --data-config so100_dualcam --host '$n15_host' --port '$n15_port'" \
        > "$n15_logfile" 2>&1 < /dev/null &
    echo $! > "$n15_pidfile"
    echo "[INFO] pid=$(cat "$n15_pidfile"), bind=$n15_host:$n15_port"

    # Eagle backbone + DiT head: ~20-30s cold load
    for i in $(seq 1 60); do
        if port_listening "$n15_port"; then
            echo "[INFO] GR00T N1.5 server listening on :$n15_port"
            return 0
        fi
        sleep 1
    done
    echo "[ERROR] server did not come up within 60s; last log lines:" >&2
    tail -n 30 "$n15_logfile" >&2 || true
    return 1
}

stop_gr00t_n15() {
    if [ -f "$n15_pidfile" ]; then
        local pid
        pid=$(cat "$n15_pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
            echo "[INFO] stopped GR00T N1.5 server (pid=$pid)"
        fi
        rm -f "$n15_pidfile"
    fi
    pkill -9 -f "inference_service.py.*--port[= ]*$n15_port" 2>/dev/null || true
}

# -------- GR00T N1.6 ZMQ :5555 (uses dependencies/Isaac-GR00T-N1.6 submodule) --------
# N1.6 cached HF code expects transformers 4.51.3 — isolated venv per release tag.
start_gr00t_n16() {
    local model_path="${1:-hi-space/GR00T-N1.6-3B-Pick-Orange}"
    export GR00T_MODEL_PATH="$model_path"
    export GR00T_EMBODIMENT_TAG="NEW_EMBODIMENT"
    export GR00T_DIR="$ROOT_DIR/dependencies/Isaac-GR00T-N1.6"
    exec bash "$ROOT_DIR/server/start_server.sh" --gr00t-only
}
stop_gr00t_n16() {
    exec bash "$ROOT_DIR/server/stop_server.sh" --gr00t-only
}

# -------- GR00T N1.7 ZMQ :5555 (uses dependencies/Isaac-GR00T submodule @ n1.7-release-2) --------
# N1.7 (Cosmos-Reason2 backbone) requires transformers 4.57.3.
start_gr00t_n17() {
    local model_path="${1:-hi-space/GR00T-N1.7-3B-Pick-Orange}"
    export GR00T_MODEL_PATH="$model_path"
    export GR00T_EMBODIMENT_TAG="NEW_EMBODIMENT"
    export GR00T_DIR="$ROOT_DIR/dependencies/Isaac-GR00T"
    exec bash "$ROOT_DIR/server/start_server.sh" --gr00t-only
}
stop_gr00t_n17() {
    exec bash "$ROOT_DIR/server/stop_server.sh" --gr00t-only
}

# -------- LeRobot async :8080 (delegated) --------
start_lerobot() {
    exec bash "$ROOT_DIR/server/start_server.sh" --lerobot-only
}
stop_lerobot() {
    exec bash "$ROOT_DIR/server/stop_server.sh" --lerobot-only
}

case "$ACTION:$BACKEND" in
    start:gr00t-n15) start_gr00t_n15 "${3:-}" ;;
    stop:gr00t-n15)  stop_gr00t_n15 ;;
    start:gr00t-n16) start_gr00t_n16 "${3:-}" ;;
    stop:gr00t-n16)  stop_gr00t_n16 ;;
    start:gr00t-n17) start_gr00t_n17 "${3:-}" ;;
    stop:gr00t-n17)  stop_gr00t_n17 ;;
    start:lerobot)   start_lerobot ;;
    stop:lerobot)    stop_lerobot ;;
esac
