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
if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -E ':5555|:8080' || true
elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:5555 -sTCP:LISTEN 2>/dev/null || true
    lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null || true
elif command -v netstat >/dev/null 2>&1; then
    netstat -an 2>/dev/null | grep -E "[.:](5555|8080)[[:space:]]+" | grep -i LISTEN || true
else
    echo "[WARN] no port-check tool found (ss/lsof/netstat)"
fi

echo ""
echo "[INFO] gr00t log tail:"
tail -n 30 "$LOG_DIR/gr00t_server.log" 2>/dev/null || echo "no gr00t log"

echo ""
echo "[INFO] lerobot log tail:"
tail -n 30 "$LOG_DIR/lerobot_server.log" 2>/dev/null || echo "no lerobot log"
