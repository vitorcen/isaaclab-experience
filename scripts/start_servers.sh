#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(cd "$ROOT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# sibling repos (override with env vars if your layout differs)
GR00T_DIR="${GR00T_DIR:-$WORK_DIR/Isaac-GR00T}"
LEROBOT_DIR="${LEROBOT_DIR:-$WORK_DIR/lerobot}"

# resolve lerobot conda env python (override with LEROBOT_PYTHON if needed)
if [ -z "${LEROBOT_PYTHON:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        LEROBOT_PYTHON="$(conda info --base)/envs/lerobot/bin/python"
    fi
fi

if ! command -v ss >/dev/null 2>&1; then
    echo "[ERROR] missing command: ss"
    exit 1
fi

echo "[INFO] log dir: $LOG_DIR"

MODE="both"
case "${1:-}" in
    "" ) MODE="both" ;;
    "--gr00t-only" ) MODE="gr00t" ;;
    "--lerobot-only" ) MODE="lerobot" ;;
    * )
        echo "[ERROR] unknown option: ${1}"
        echo "usage: $0 [--gr00t-only|--lerobot-only]"
        exit 1
        ;;
esac

wait_for_port() {
    local port="$1"
    local retries="${2:-20}"
    local delay_s="${3:-0.5}"
    local i
    for i in $(seq 1 "$retries"); do
        if ss -ltn | awk '{print $4}' | grep -q ":${port}\$"; then
            return 0
        fi
        sleep "$delay_s"
    done
    return 1
}

start_gr00t_server() {
    if ss -ltn | awk '{print $4}' | grep -q ':5555$'; then
        echo "[INFO] GR00T server already listening on :5555"
        return
    fi

    if [ ! -d "$GR00T_DIR" ]; then
        echo "[ERROR] GR00T repo not found: $GR00T_DIR (set GR00T_DIR to override)"
        exit 1
    fi
    nohup setsid bash -lc "cd '$GR00T_DIR' && uv run --extra=gpu python gr00t/eval/run_gr00t_server.py --embodiment-tag GR1 --model-path nvidia/GR00T-N1.6-3B --host 127.0.0.1 --port 5555" > "$LOG_DIR/gr00t_server.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/gr00t_server.pid"
    echo "[INFO] GR00T server launched, pid=$(cat "$LOG_DIR/gr00t_server.pid")"
    if wait_for_port 5555 6 0.5; then
        echo "[INFO] GR00T server is listening on :5555"
    else
        echo "[WARN] GR00T server not ready yet (it may still be building dependencies)."
        echo "[WARN] Check log: $LOG_DIR/gr00t_server.log"
    fi
}

start_lerobot_server() {
    if ss -ltn | awk '{print $4}' | grep -q ':8080$'; then
        echo "[INFO] LeRobot server already listening on :8080"
        return
    fi

    if [ -z "${LEROBOT_PYTHON:-}" ] || [ ! -x "$LEROBOT_PYTHON" ]; then
        echo "[ERROR] lerobot python not found: ${LEROBOT_PYTHON:-<unset>}"
        echo "        set LEROBOT_PYTHON or ensure 'conda' is on PATH with env 'lerobot'"
        exit 1
    fi
    if [ ! -d "$LEROBOT_DIR/src" ]; then
        echo "[ERROR] lerobot src not found: $LEROBOT_DIR/src (set LEROBOT_DIR to override)"
        exit 1
    fi

    nohup setsid bash -lc "PYTHONPATH='$LEROBOT_DIR/src':\${PYTHONPATH:-} '$LEROBOT_PYTHON' -m lerobot.async_inference.policy_server --host 127.0.0.1 --port 8080" > "$LOG_DIR/lerobot_server.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/lerobot_server.pid"
    echo "[INFO] LeRobot server launched, pid=$(cat "$LOG_DIR/lerobot_server.pid")"
    if wait_for_port 8080 20 0.5; then
        echo "[INFO] LeRobot server is listening on :8080"
    else
        echo "[ERROR] LeRobot server failed to start."
        echo "[ERROR] Last log lines:"
        tail -n 40 "$LOG_DIR/lerobot_server.log" || true
        exit 1
    fi
}

if [ "$MODE" = "both" ] || [ "$MODE" = "gr00t" ]; then
    start_gr00t_server
fi
if [ "$MODE" = "both" ] || [ "$MODE" = "lerobot" ]; then
    start_lerobot_server
fi

echo "[INFO] listening ports:"
ss -ltn | grep -E ':5555|:8080' || true
