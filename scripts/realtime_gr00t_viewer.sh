#!/usr/bin/env bash
#
# Launch a live MuJoCo viewer window driven by the GR00T server.
# Unlike preview_gr00t_inference.sh (which records mp4 and plays it back),
# this one renders each sim step on-screen in real time via
# mujoco.viewer.launch_passive — no video file is produced.
#
# Requires:
#   - GR00T server on :5555 with --use-sim-policy-wrapper  (scripts/check_start_gr00t.sh)
#   - DISPLAY available  (X11 / Xwayland)
#
# Usage:
#   bash scripts/realtime_gr00t_viewer.sh
#   ENV_NAME=<env> bash scripts/realtime_gr00t_viewer.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_DIR="${GR00T_DIR:-$ROOT_DIR/dependencies/Isaac-GR00T}"
ROBOCASA_PY="$GR00T_DIR/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"

[ -x "$ROBOCASA_PY" ] || { echo "[ERROR] robocasa venv missing: $ROBOCASA_PY"; exit 1; }
[ -n "${DISPLAY:-}" ] || { echo "[ERROR] no DISPLAY; need a graphical session"; exit 1; }

# robosuite's offscreen camera renderer requires EGL; the GLFW viewer
# window is independent of mjr_render so it coexists with EGL offscreen.
export MUJOCO_GL=egl
unset PYOPENGL_PLATFORM

cd "$GR00T_DIR"
exec "$ROBOCASA_PY" "$ROOT_DIR/scripts/realtime_gr00t_viewer.py" \
    --env_name "${ENV_NAME:-gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env}" \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps 504 \
    --n_episodes "${N_EPISODES:-1}" \
    --n_action_steps 8
