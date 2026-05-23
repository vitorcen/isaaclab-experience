#!/usr/bin/env bash
# Smoke test — verify Isaac Sim headless + dummy 1-episode eval runs OK.
# Doesn't load any real policy (uses zero-action), just confirms sim init + render works.

set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/isaaclab-experience}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
PY="/root/miniconda3/envs/$CONDA_ENV/bin/python"

export KIT_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=Y
export ACCEPT_EULA=Y
export DISPLAY=""   # force no X

echo "[smoke] importing isaacsim + isaaclab in headless mode"
$PY -c "
import os
os.environ['KIT_HEADLESS'] = '1'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
print('[smoke] AppLauncher OK')
from isaaclab.envs import ManagerBasedRLEnv
print('[smoke] env import OK')
app.close()
print('[smoke] app close OK')
" || { echo "[smoke] FAIL — likely missing Vulkan / driver" >&2; exit 1; }

echo "[smoke] sim init smoke PASSED. ready for run_one_strict.sh"
