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

MODE="both"
case "${1:-}" in
    "" )               MODE="both" ;;
    "--gr00t-only" )   MODE="gr00t" ;;
    "--lerobot-only" ) MODE="lerobot" ;;
    * ) echo "[ERROR] unknown option: ${1}"; echo "usage: $0 [--gr00t-only|--lerobot-only]"; exit 1 ;;
esac

if [ "$MODE" = "both" ] || [ "$MODE" = "gr00t" ]; then
    stop_by_pid_file "gr00t_server" "$LOG_DIR/gr00t_server.pid"
    pkill -f 'gr00t/eval/run_gr00t_server.py' 2>/dev/null || true
fi
if [ "$MODE" = "both" ] || [ "$MODE" = "lerobot" ]; then
    stop_by_pid_file "lerobot_server" "$LOG_DIR/lerobot_server.pid"
    pkill -f 'lerobot.async_inference.policy_server' 2>/dev/null || true
fi

sleep 1
echo "[INFO] remaining ports:"
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
