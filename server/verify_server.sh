#!/usr/bin/env bash
#
# Verify GR00T / LeRobot policy servers are reachable from this machine.
# Works on macOS and Linux. Uses bash /dev/tcp (no nc required).
# Optional ZMQ ping for the GR00T server when python3 + pyzmq are available.
#
# Usage:
#   server/verify_server.sh                          # localhost, default ports
#   server/verify_server.sh 192.168.1.20             # custom host, default ports
#   server/verify_server.sh 192.168.1.20 5555 8080   # custom host + ports
#
# Env overrides:
#   GR00T_HOST / GR00T_PORT / LEROBOT_HOST / LEROBOT_PORT
#   ZMQ_PING=1   force ZMQ REQ ping to GR00T (default: only TCP probe)
#   TIMEOUT=2    per-probe timeout in seconds

set -uo pipefail

HOST_ARG="${1:-}"
GR00T_HOST="${GR00T_HOST:-${HOST_ARG:-127.0.0.1}}"
LEROBOT_HOST="${LEROBOT_HOST:-${HOST_ARG:-127.0.0.1}}"
GR00T_PORT="${GR00T_PORT:-${2:-5555}}"
LEROBOT_PORT="${LEROBOT_PORT:-${3:-8080}}"
TIMEOUT="${TIMEOUT:-2}"
ZMQ_PING="${ZMQ_PING:-0}"

PASS=0
FAIL=0

probe_tcp() {
    # $1 host  $2 port  -> 0 on success
    local host="$1" port="$2"
    # bash /dev/tcp opens with the shell's connect timeout; wrap with a
    # subshell + background timer for portable per-probe timeout.
    (
        exec 3<>"/dev/tcp/${host}/${port}"
    ) >/dev/null 2>&1 &
    local pid=$!
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$waited" -ge "$TIMEOUT" ]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    wait "$pid"
    return $?
}

check_port() {
    local label="$1" host="$2" port="$3"
    printf "[CHECK] %-7s %s:%s ... " "$label" "$host" "$port"
    if probe_tcp "$host" "$port"; then
        echo "OK (TCP reachable)"
        PASS=$((PASS + 1))
        return 0
    else
        echo "FAIL (TCP unreachable within ${TIMEOUT}s)"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

zmq_ping() {
    # Best-effort ZMQ REQ/REP ping. Many GR00T deployments respond to a
    # 'ping'-style empty/control message; if not, we just report timeout.
    local host="$1" port="$2"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[INFO]  ZMQ ping skipped: python3 not found"
        return
    fi
    python3 - "$host" "$port" "$TIMEOUT" <<'PY' 2>&1
import sys
host, port, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    import zmq
except ImportError:
    print("[INFO]  ZMQ ping skipped: pyzmq not installed (pip install pyzmq)")
    sys.exit(0)

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.setsockopt(zmq.RCVTIMEO, timeout * 1000)
sock.setsockopt(zmq.SNDTIMEO, timeout * 1000)
addr = f"tcp://{host}:{port}"
try:
    sock.connect(addr)
    sock.send(b"ping")
    reply = sock.recv()
    print(f"[OK]    ZMQ replied {len(reply)} bytes from {addr}")
except zmq.error.Again:
    print(f"[WARN]  ZMQ no reply within {timeout}s (TCP up; server may use a different protocol)")
except Exception as e:
    print(f"[WARN]  ZMQ probe error: {e}")
finally:
    sock.close()
    ctx.term()
PY
}

echo "[INFO] verifying servers (timeout=${TIMEOUT}s per probe)"
echo

GR00T_OK=0
check_port "GR00T"   "$GR00T_HOST"   "$GR00T_PORT"   && GR00T_OK=1
check_port "LeRobot" "$LEROBOT_HOST" "$LEROBOT_PORT" || true

if [ "$GR00T_OK" -eq 1 ] && [ "$ZMQ_PING" = "1" ]; then
    echo
    zmq_ping "$GR00T_HOST" "$GR00T_PORT"
fi

echo
echo "[RESULT] pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
