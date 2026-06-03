#!/usr/bin/env bash
# Real-time preview of NVIDIA GEAR-SONIC driving the Unitree G1 — single-process Isaac
# Lab eval that opens the Isaac Sim viewer and plays the released SONIC policy on the
# sample motion. NO DDS, NO C++ deploy binary.
#
# (The two-process MuJoCo sim2sim route — `deploy.sh` + `run_sim_loop.py` — needs the
#  separate TensorRT C++ deployment build AND `sudo ip link set lo multicast on`; it
#  lives in gear_sonic_preview_sim2sim.sh and is NOT what most people want for a look.)
#
#   docs: https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html  (Isaac Lab Eval)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
HEADLESS="${HEADLESS:-False}"   # False -> on-screen viewer; True -> headless metrics
NUM_ENVS="${NUM_ENVS:-1}"
export DISPLAY="${DISPLAY:-:0}"

if [[ ! -d "$WBC_DIR" ]]; then
  echo "[preview] $WBC_DIR missing — run scripts/gear_sonic_setup.sh first."; exit 1
fi
cd "$WBC_DIR"

if [[ ! -e "$CKPT" ]]; then
  echo "[preview] checkpoint $CKPT missing — run scripts/gear_sonic_setup.sh (downloads sonic_release/last.pt)."
  exit 1
fi

echo "[preview] launching Isaac Sim viewer  (env=$ENV_NAME · num_envs=$NUM_ENVS · headless=$HEADLESS)"
echo "  checkpoint: $CKPT -> $(readlink -f "$CKPT" 2>/dev/null || echo "$CKPT")"
echo "  Isaac Sim takes ~30-60s to boot the first time. Ctrl+C here to stop."

conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
    +checkpoint="$CKPT" \
    +headless="$HEADLESS" \
    ++num_envs="$NUM_ENVS" \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"

echo "[preview] viewer closed."
