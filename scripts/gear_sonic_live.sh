#!/usr/bin/env bash
# CLOSE THE LOOP: a live GR00T server drives SONIC's WBC in the Isaac viewer.
#   prompt -> GR00T (ZMQ server, port 5555) -> 64-dim token (live, per current env state)
#          -> injected into SONIC decode -> G1 moves.  GR00T sees the ACTUAL drifted state.
# The green reference skeleton (press V) is the TARGET; the robot follows GR00T's live token.
#
# Prereq: start the GR00T server FIRST, in the Isaac-GR00T venv:
#   cd dependencies/Isaac-GR00T && COMPILE_ACTION_HEAD_DISABLE=1 .venv/bin/python \
#     -m gr00t.eval.run_gr00t_server --model_path <ckpt-8000> \
#     --embodiment_tag unitree_g1_sonic --port 5555
#
#   bash scripts/gear_sonic_live.sh kick      # default motion = kick
#   bash scripts/gear_sonic_live.sh dance
# names: dance | lunge | macarena | kick | squat | jump | walk
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:-data/demo_robot_filtered.pkl}"
HOST="${GR00T_HOST:-127.0.0.1}"
PORT="${GR00T_PORT:-5555}"
ACTION_HORIZON="${ACTION_HORIZON:-40}"
CAM_RES="${CAM_RES:-[480,640]}"
RELAX="${RELAX:-1}"   # relax tracking-deviation terminations so the motion plays without 1-2s reset
# HEADLESS=True for a metrics-only smoke (no viewer). NOTE: IsaacLab's AppLauncher reads the
# env var $HEADLESS and does int(os.environ["HEADLESS"]) -> 'True' crashes it. So we consume it
# into a local, normalize to True/False, unset the env var, and drive headless via +headless= only.
case "${HEADLESS:-False}" in 1|True|true|yes) HL=True;; *) HL=False;; esac
unset HEADLESS
export DISPLAY="${DISPLAY:-:0}"

declare -A KEY=(
  [dance]=dance_in_da_party_001__A464  [lunge]=forward_lunge_R_001__A359_M
  [macarena]=macarena_001__A545        [kick]=neutral_kick_R_001__A543
  [squat]=squat_001__A359              [jump]=tired_one_leg_jumping_R_001__A359
  [walk]=walking_quip_360_R_002__A428
)
declare -A PROMPT=(
  [dance]="dance"   [lunge]="do a forward lunge" [macarena]="dance the macarena"
  [kick]="kick"     [squat]="squat"              [jump]="jump on one leg"
  [walk]="walk and turn around"
)
SEL="${1:-${MOTION:-kick}}"
k="${KEY[$SEL]:-}"; [[ -z "$k" ]] && { echo "[live] unknown motion '$SEL' (dance|lunge|macarena|kick|squat|jump|walk)"; exit 2; }
p="${PROMPT[$SEL]}"

# BOOTSTRAP=<N>: for one-shot motions (kick/squat/jump/lunge), replay N steps of the open-loop
# dump token after each reset to drive INTO the motion, then hand to live GR00T closed-loop.
BOOTSTRAP="${BOOTSTRAP:-0}"
PRED_DIR="${PRED_DIR:-$REPO_ROOT/datasets/sonic_vla_pred_8k_final}"
BOOT_ARGS=()
if [[ "$BOOTSTRAP" != "0" ]]; then
  NPZ="$PRED_DIR/$k.npz"
  [[ -f "$NPZ" ]] || { echo "[live] BOOTSTRAP needs dump tokens at $NPZ (run gr00t_dump_pred_tokens.py)"; exit 1; }
  BOOT_ARGS=("++callbacks.vla_live.bootstrap_npz=$NPZ" "++callbacks.vla_live.bootstrap_steps=$BOOTSTRAP")
  echo "[live] BOOTSTRAP=$BOOTSTRAP steps from $NPZ -> then hand to live GR00T"
fi

cd "$WBC_DIR"
echo "[live] motion=$SEL key=$k prompt='$p'  GR00T server=$HOST:$PORT  horizon=$ACTION_HORIZON"
echo "  Isaac viewer: F=free cam, V=green ref skeleton (target) vs robot (GR00T-driven live), R=reset."
echo "  NOTE: every $ACTION_HORIZON steps the sim briefly stalls (~0.4s) while GR00T infers a chunk."

# debug_vis (green ref skeleton) needs a viewer; in headless it errors -> only enable with GUI.
VIS_ARGS=()
[[ "$HL" == "False" ]] && VIS_ARGS=(++manager_env.commands.motion.debug_vis=True)

RELAX_ARGS=()
if [[ "$RELAX" == "1" ]]; then
  RELAX_ARGS=(
    ++manager_env.terminations.anchor_pos.params.threshold=99
    ++manager_env.terminations.ee_body_pos.params.threshold=99
    ++manager_env.terminations.anchor_ori_full.params.threshold=99
    ++manager_env.terminations.foot_pos_xyz.params.threshold=99
  )
fi

conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
    +checkpoint="$CKPT" \
    +headless="$HL" \
    ++num_envs=1 \
    ++manager_env.config.enable_cameras=True \
    "++manager_env.config.cameras.camera_resolution=$CAM_RES" \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    ++manager_env.commands.motion.use_paired_motions=True \
    "${VIS_ARGS[@]}" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$k" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
    "${RELAX_ARGS[@]}" \
    "++eval_callbacks=[vla_live]" \
    "++callbacks.vla_live._target_=gear_sonic.data.vla_live_injector.VlaLiveInjector" \
    "++callbacks.vla_live.host=$HOST" \
    "++callbacks.vla_live.port=$PORT" \
    "++callbacks.vla_live.prompt=$p" \
    "++callbacks.vla_live.action_horizon=$ACTION_HORIZON" \
    "${BOOT_ARGS[@]}"

echo "[live] viewer closed."
