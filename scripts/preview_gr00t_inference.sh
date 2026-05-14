#!/usr/bin/env bash
#
# Run 1 rollout episode against the GR00T server on a robocasa GR1 tabletop
# task, then open the produced mp4 in totem (full-screen) for live preview.
#
# Server must already be listening on :5555 with --use-sim-policy-wrapper.
# Run scripts/check_start_gr00t.sh first if unsure.
#
# Usage:
#   bash scripts/preview_gr00t_inference.sh                # default task
#   ENV_NAME=<env> bash scripts/preview_gr00t_inference.sh # override task

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_DIR="${GR00T_DIR:-$ROOT_DIR/../Isaac-GR00T}"
ROBOCASA_PY="$GR00T_DIR/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"
ENV_NAME="${ENV_NAME:-gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env}"
HOST="${GR00T_HOST:-127.0.0.1}"
PORT="${GR00T_PORT:-5555}"

if [ ! -x "$ROBOCASA_PY" ]; then
    echo "[ERROR] robocasa venv missing: $ROBOCASA_PY"
    echo "        run: bash $GR00T_DIR/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh"
    exit 1
fi

echo "[INFO] rollout 1 episode on $(basename "$ENV_NAME")"
echo "[INFO] this takes ~15s (504 sim steps, ~13 server inference calls)"

cd "$GR00T_DIR"
"$ROBOCASA_PY" gr00t/eval/rollout_policy.py \
    --n_episodes 1 \
    --n_envs 1 \
    --max_episode_steps 504 \
    --policy_client_host "$HOST" \
    --policy_client_port "$PORT" \
    --env_name "$ENV_NAME" \
    --n_action_steps 8 \
    2>&1 | tee "$ROOT_DIR/logs/rollout.log" | grep --line-buffered -E "Running collecting|Video saved|success rate|Traceback|Error" || true

VIDEO_DIR=$(grep -oP '(?<=Video saved to:  ).*' "$ROOT_DIR/logs/rollout.log" | tail -n1)
[ -n "$VIDEO_DIR" ] || { echo "[ERROR] no video produced; check $ROOT_DIR/logs/rollout.log"; exit 1; }

VIDEO_FILE=$(ls -t "$VIDEO_DIR"/*.mp4 2>/dev/null | head -n1)
[ -n "$VIDEO_FILE" ] || { echo "[ERROR] no mp4 in $VIDEO_DIR"; exit 1; }

echo "[OK] $(du -h "$VIDEO_FILE" | cut -f1) -> $VIDEO_FILE"

if command -v totem >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
    DISPLAY="${DISPLAY:-:0}" nohup totem --fullscreen "$VIDEO_FILE" > /dev/null 2>&1 &
    echo "[INFO] opened in totem (full-screen)"
else
    echo "[INFO] no DISPLAY or totem; open manually: $VIDEO_FILE"
fi
