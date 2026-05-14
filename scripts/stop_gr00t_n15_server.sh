#!/usr/bin/env bash
# Stop the GR00T N1.5 inference server started by start_gr00t_n15_server.sh.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PIDFILE="$LOG_DIR/gr00t_n15_server.pid"
PORT="${GR00T_N15_PORT:-5555}"

if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        echo "[INFO] stopped GR00T N1.5 server (pid=$pid)"
    fi
    rm -f "$PIDFILE"
fi

# Fallback: kill any inference_service.py on this port.
pkill -9 -f "inference_service.py.*--port[= ]*$PORT" 2>/dev/null || true
