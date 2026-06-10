#!/usr/bin/env bash
# KINEMATIC preview (mocap replay, NEVER falls) of a [START,END]-second window of a LAFAN clip.
# Uses MimicKit's view_motion_g1_env (motors off, FK-synced) — unlike SONIC WBC tracking, the
# character follows the reference exactly so you can study any sub-action even if a physical
# controller couldn't track it. For hand-picking gentle segments before training.
#
#   CLIP=dance START=8 END=12 bash scripts/mimickit_preview_window.sh
#   CLIP=fight START=22 END=25 bash scripts/mimickit_preview_window.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIP="${CLIP:?set CLIP=fight|run|dance|jumps}"
START="${START:?set START=<seconds>}"
END="${END:?set END=<seconds>}"
NAME="${NAME:-_lafan_window}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"

conda run --no-capture-output -n "$CONDA_ENV" python "$REPO_ROOT/scripts/cut_mimickit_window.py" \
    --clip "$CLIP" --start_s "$START" --end_s "$END" --out_name "$NAME" || exit 1

echo "[preview-window] kinematic replay of $CLIP [$START s, $END s] ..."
exec bash "$REPO_ROOT/scripts/mimickit_preview.sh" view "$NAME"
