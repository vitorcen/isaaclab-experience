#!/usr/bin/env bash
# One-click GEAR-SONIC multi-motion demo: line up N Unitree G1 in the Isaac Sim
# viewer, each tracking a different local SONIC deploy demo motion
# (macarena / dance / kick / lunge / jump / squat / walk-360).
#
# This is Path-A of the architecture-side VLA route: the released SONIC WBC
# (sonic_release/last.pt) is asked to TRACK each reference motion. If it tracks
# the kick/dance without falling, the motion lives in SONIC's token space and a
# VLA can later drive it by emitting the token (see doc/groot_sonic_wbc_route.html).
#
# Motion source: gear_sonic_deploy/reference/example/*  (G1 29-DoF deploy CSVs)
# converted to robot_filtered via convert_soma_csv_to_motion_lib.py --fps 50.
#   NO DDS, NO C++ — single-process Isaac Lab eval, same route as gear_sonic_preview.sh.
#
# Usage:
#   bash scripts/gear_sonic_demo.sh            # all 7 motions, one G1 each (paired line-up)
#   bash scripts/gear_sonic_demo.sh kick       # single motion, one G1
#   MOTION=dance bash scripts/gear_sonic_demo.sh
#   names: dance | lunge | macarena | kick | squat | jump | walk | all
# Viewer keys: F = free camera (stop auto-follow) · R = reset · T = next motion · V = ref skeleton.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
HEADLESS="${HEADLESS:-False}"
PKL="${PKL:-data/demo_robot_filtered.pkl}"   # multi-motion robot_filtered (single file)
STAGE_DIR="${STAGE_DIR:-data/demo_base}"      # symlink staging of chosen demo motions
FPS="${FPS:-50}"                              # eval target_fps=50; converter default is 30
export DISPLAY="${DISPLAY:-:0}"

# Short name -> full robot_filtered motion key (skip _M mirror variants).
declare -A SHORT2KEY=(
  [dance]=dance_in_da_party_001__A464
  [lunge]=forward_lunge_R_001__A359_M
  [macarena]=macarena_001__A545
  [kick]=neutral_kick_R_001__A543
  [squat]=squat_001__A359
  [jump]=tired_one_leg_jumping_R_001__A359
  [walk]=walking_quip_360_R_002__A428
)
# Stable order used for the all-in-one paired line-up and pkl build.
ORDER=(dance lunge macarena kick squat jump walk)
MOTIONS=(); for s in "${ORDER[@]}"; do MOTIONS+=("${SHORT2KEY[$s]}"); done

# Selection: $1 (or $MOTION env), default "all". One of: ${ORDER[*]} | all
SEL="${1:-${MOTION:-all}}"

if [[ ! -d "$WBC_DIR" ]]; then
  echo "[demo] $WBC_DIR missing — run scripts/gear_sonic_setup.sh first."; exit 1
fi
cd "$WBC_DIR"

if [[ ! -e "$CKPT" ]]; then
  echo "[demo] checkpoint $CKPT missing — run scripts/gear_sonic_setup.sh."; exit 1
fi

# 1) Stage chosen motions as symlinks (idempotent) + convert to robot_filtered if stale.
EX="gear_sonic_deploy/reference/example"
if [[ ! -f "$PKL" ]]; then
  echo "[demo] building $PKL from ${#MOTIONS[@]} demo motions ..."
  mkdir -p "$STAGE_DIR"
  for m in "${MOTIONS[@]}"; do
    [[ -d "$EX/$m" ]] || { echo "[demo] WARN missing motion $m"; continue; }
    ln -sfn "$(realpath "$EX/$m")" "$STAGE_DIR/$m"
  done
  conda run --no-capture-output -n "$ENV_NAME" python \
    gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input "$STAGE_DIR" --output "$PKL" --fps "$FPS" || exit 1
fi

# 2) Resolve selection -> filter override + num_envs.
FILTER_OVERRIDE=()
if [[ "$SEL" == "all" ]]; then
  NUM_ENVS="${NUM_ENVS:-${#MOTIONS[@]}}"
  echo "[demo] launching Isaac Sim viewer — $NUM_ENVS G1, one per motion (paired)."
  echo "  motions: ${ORDER[*]}"
else
  KEY="${SHORT2KEY[$SEL]:-}"
  if [[ -z "$KEY" ]]; then
    echo "[demo] unknown motion '$SEL'. choose one of: ${ORDER[*]} | all"; exit 2
  fi
  NUM_ENVS="${NUM_ENVS:-1}"
  # filter_motion_keys keeps only this motion; all envs then track it.
  FILTER_OVERRIDE=("++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$KEY")
  echo "[demo] launching Isaac Sim viewer — $NUM_ENVS G1 · motion=$SEL ($KEY)."
fi
echo "  Isaac Sim takes ~30-60s to boot. Ctrl+C here to stop. (viewer: press F = free camera)"

# use_paired_motions=True -> env i tracks motion i (arange % num_motions); deterministic
# line-up so each demo is clearly visible side by side.
conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
    +checkpoint="$CKPT" \
    +headless="$HEADLESS" \
    ++num_envs="$NUM_ENVS" \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    ++manager_env.commands.motion.use_paired_motions=True \
    "${FILTER_OVERRIDE[@]}" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy"

echo "[demo] viewer closed."
