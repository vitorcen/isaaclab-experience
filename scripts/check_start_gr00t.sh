#!/usr/bin/env bash
#
# Ensure GR00T inference server is listening on :5555 with the sim policy
# wrapper enabled (required by robocasa GR1 tabletop benchmark).
#
# Idempotent: if a server is already listening, do nothing.
#
# Usage: bash scripts/check_start_gr00t.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${GR00T_HOST:-127.0.0.1}"
PORT="${GR00T_PORT:-5555}"
GR00T_VENV_PY="${GR00T_DIR:-$ROOT_DIR/dependencies/Isaac-GR00T}/.venv/bin/python"

port_up() { (exec 3<>"/dev/tcp/$HOST/$PORT") >/dev/null 2>&1; }

if port_up; then
    echo "[OK] GR00T server already listening on $HOST:$PORT"
else
    echo "[INFO] starting GR00T server with sim policy wrapper"
    GR00T_SIM_WRAPPER=1 bash "$ROOT_DIR/server/start_server.sh" --gr00t-only

    echo "[INFO] waiting for model load (typically 30-90s)..."
    for i in $(seq 1 180); do
        port_up && break
        sleep 1
    done
    port_up || { echo "[FAIL] timed out; see logs/gr00t_server.log"; exit 1; }
    echo "[OK] :$PORT up"
fi

# ZMQ application-level handshake via GR00T venv's pyzmq (avoid system pyzmq breakage)
if [ -x "$GR00T_VENV_PY" ]; then
    echo -n "[CHECK] ZMQ ping ... "
    "$GR00T_VENV_PY" - "$HOST" "$PORT" <<'PY'
import sys, zmq
host, port = sys.argv[1], int(sys.argv[2])
ctx = zmq.Context.instance()
s = ctx.socket(zmq.REQ)
s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, 5000); s.setsockopt(zmq.SNDTIMEO, 5000)
s.connect(f"tcp://{host}:{port}")
s.send(b"ping")
try:
    r = s.recv()
    print(f"replied {len(r)} bytes — server is processing messages")
except zmq.error.Again:
    print("TIMEOUT — port open but no app-level response"); sys.exit(2)
PY
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo -n "[GPU]  "
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
fi
