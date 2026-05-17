#!/usr/bin/env bash
# Emergency cleanup for LeIsaac eval — kill stuck Isaac sim / policy_inference / inference servers, free GPU.
# 一键清理：杀 Isaac Sim 窗口 + policy_inference grandchild + LeRobot/GR00T/π0.5 server，释放显存。
#
# Use when:
#   - Isaac Sim window hangs / 窗口卡死
#   - `timeout` failed to kill a policy_inference grandchild
#   - GPU memory is exhausted
#
# Optional env:
#   LEISAAC_ROOT    parent repo root (default: $HOME/work/isaaclab-experience)
#   SKIP_LEROBOT    set to skip stopping LeRobot server
#   SKIP_GR00T      set to skip stopping GR00T servers
#   SKIP_PI05       set to skip stopping π0.5 PT server

set -u  # no -e: we want to keep going even if individual kills fail

ROOT="${LEISAAC_ROOT:-$HOME/work/isaaclab-experience}"

echo "=== 1) kill policy_inference + Isaac sim grandchildren ==="
pgrep -af "policy_inference|isaac-sim|kit\.py" 2>/dev/null \
    | grep -v "pgrep\|jupyter\|ipykernel\|cleanup_eval" \
    | awk '{print $1}' | xargs -r kill -TERM 2>/dev/null
sleep 3
pgrep -af "policy_inference|isaac-sim|kit\.py" 2>/dev/null \
    | grep -v "pgrep\|jupyter\|ipykernel\|cleanup_eval" \
    | awk '{print $1}' | xargs -r kill -KILL 2>/dev/null

if [ -z "${SKIP_LEROBOT:-}" ] || [ -z "${SKIP_GR00T:-}" ]; then
    echo "=== 2) stop LeRobot / GR00T servers (via policy_server.sh) ==="
fi
[ -z "${SKIP_LEROBOT:-}" ] && bash "$ROOT/scripts/policy_server.sh" stop lerobot   2>/dev/null || true
[ -z "${SKIP_GR00T:-}"   ] && bash "$ROOT/scripts/policy_server.sh" stop gr00t-n15 2>/dev/null || true
[ -z "${SKIP_GR00T:-}"   ] && bash "$ROOT/scripts/policy_server.sh" stop gr00t-n16 2>/dev/null || true

if [ -z "${SKIP_PI05:-}" ]; then
    echo "=== 3) stop π0.5 PT server (custom, not via policy_server.sh) ==="
    PID_FILE="$ROOT/logs/pi05_server.pid"
    if [ -f "$PID_FILE" ]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
    pgrep -af "pi05_leisaac.server" 2>/dev/null | awk '{print $1}' | xargs -r kill -KILL 2>/dev/null
fi

sleep 2
echo
echo "=== residual processes (should be empty) ==="
pgrep -af "policy_inference|isaac-sim|kit\.py|policy_server|pi05_leisaac" 2>/dev/null \
    | grep -v "pgrep\|jupyter\|ipykernel\|cleanup_eval" \
    || echo "  (all clean)"

echo
echo "=== listening ports (should be empty for :5555 / :5556 / :8080) ==="
ss -ltn 2>/dev/null | grep -E ":(5555|5556|8080)\b" || echo "  (all ports released)"

echo
echo "=== GPU usage ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
