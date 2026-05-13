#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"

stop_by_pid_file() {
    local name="$1"
    local pid_file="$2"

    if [ -f "$pid_file" ]; then
        local pid
        pid="$(cat "$pid_file")"
        kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
        echo "[INFO] stopped $name (pid=$pid)"
    else
        echo "[INFO] no pid file for $name"
    fi
}

stop_by_pid_file "gr00t_server" "$LOG_DIR/gr00t_server.pid"
stop_by_pid_file "lerobot_server" "$LOG_DIR/lerobot_server.pid"

# Best-effort cleanup for stale background processes without pid files.
pkill -f 'gr00t/eval/run_gr00t_server.py' 2>/dev/null || true
pkill -f 'lerobot.async_inference.policy_server' 2>/dev/null || true

sleep 1
echo "[INFO] remaining ports:"
ss -ltn | grep -E ':5555|:8080' || true
