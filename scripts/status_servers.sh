#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"

show_pid_status() {
    local name="$1"
    local pid_file="$2"
    if [ -f "$pid_file" ]; then
        local pid
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[INFO] $name pid=$pid (running)"
        else
            echo "[WARN] $name pid=$pid (not running)"
        fi
    else
        echo "[INFO] $name pid file not found"
    fi
}

show_pid_status "gr00t_server" "$LOG_DIR/gr00t_server.pid"
show_pid_status "lerobot_server" "$LOG_DIR/lerobot_server.pid"
echo ""

echo "[INFO] server ports:"
ss -ltn | grep -E ':5555|:8080' || true

echo ""
echo "[INFO] gr00t log tail:"
tail -n 30 "$LOG_DIR/gr00t_server.log" 2>/dev/null || echo "no gr00t log"

echo ""
echo "[INFO] lerobot log tail:"
tail -n 30 "$LOG_DIR/lerobot_server.log" 2>/dev/null || echo "no lerobot log"
