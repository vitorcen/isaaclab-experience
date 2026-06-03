#!/usr/bin/env bash
# Drive SONIC WBC with GR00T's PREDICTED tokens in the Isaac viewer (Stage C deploy, in-process).
# Shows what the current finetuned GR00T checkpoint actually produces:
#   prompt -> GR00T -> 64-dim token (precomputed by gr00t_dump_pred_tokens.py)
#          -> injected into SONIC decode -> G1 moves.  NO C++/ZMQ.
# The green reference skeleton (press V) is the TARGET motion; the robot follows GR00T's token.
#
#   bash scripts/gear_sonic_inject.sh kick      # default motion = kick
#   bash scripts/gear_sonic_inject.sh dance
# names: dance | lunge | macarena | kick | squat | jump | walk
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:-data/demo_robot_filtered.pkl}"
PRED_DIR="${PRED_DIR:-$REPO_ROOT/datasets/sonic_vla_pred_8k_final}"   # checkpoint-8000 tokens
RELAX="${RELAX:-1}"   # 1 = relax tracking-deviation terminations (open-loop drifts off ref;
                      #     keeps only motion_timeout so the full motion plays without 1-2s reset)
export DISPLAY="${DISPLAY:-:0}"

declare -A KEY=(
  [dance]=dance_in_da_party_001__A464  [lunge]=forward_lunge_R_001__A359_M
  [macarena]=macarena_001__A545        [kick]=neutral_kick_R_001__A543
  [squat]=squat_001__A359              [jump]=tired_one_leg_jumping_R_001__A359
  [walk]=walking_quip_360_R_002__A428
)
SEL="${1:-${MOTION:-kick}}"
k="${KEY[$SEL]:-}"; [[ -z "$k" ]] && { echo "[inject] unknown motion '$SEL' (dance|lunge|macarena|kick|squat|jump|walk)"; exit 2; }
NPZ="$PRED_DIR/$k.npz"
[[ -f "$NPZ" ]] || { echo "[inject] $NPZ missing — run scripts/gr00t_dump_pred_tokens.py first"; exit 1; }

cd "$WBC_DIR"
echo "[inject] motion=$SEL key=$k  GR00T tokens=$NPZ"
echo "  Isaac viewer: press F=free cam, V=show green ref-skeleton (target) vs robot (GR00T-driven)."
echo "  Isaac viewer: F=free cam, V=ref skeleton, R=reset.  RELAX=$RELAX (1=play full motion, no deviation-reset)."

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
    +headless=False \
    ++num_envs=1 \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    ++manager_env.commands.motion.use_paired_motions=True \
    ++manager_env.commands.motion.debug_vis=True \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$k" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
    "${RELAX_ARGS[@]}" \
    "++eval_callbacks=[vla_injector]" \
    "++callbacks.vla_injector._target_=gear_sonic.data.vla_token_injector.VlaTokenInjector" \
    "++callbacks.vla_injector.token_npz=$NPZ"

echo "[inject] viewer closed."
