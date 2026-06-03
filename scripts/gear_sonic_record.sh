#!/usr/bin/env bash
# Record SONIC motion-token episodes for the architecture-side VLA route (Path T1, step 1).
#
# Runs the released SONIC WBC tracking each demo motion (headless), taps the 64-dim FSQ
# motion_token the actor emits + ego camera + proprio, and dumps one .npz per episode via
# the VlaTokenRecorder eval-callback. NO lerobot here — an offline converter later turns
# these raw .npz into the UNITREE_G1_SONIC LeRobot dataset for GR00T finetuning.
#
# Usage:
#   bash scripts/gear_sonic_record.sh                 # all 7 motions, 1 episode each
#   bash scripts/gear_sonic_record.sh kick            # single motion
#   SMOKE=1 bash scripts/gear_sonic_record.sh kick    # tiny run (30 steps) to validate the pipe
#   EPISODES=10 bash scripts/gear_sonic_record.sh     # 10 episodes/motion (diversity via RSI)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:-data/demo_robot_filtered.pkl}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/datasets/sonic_vla_raw}"
CAM_RES="${CAM_RES:-[480,640]}"        # [H,W]; matches features_sonic_vla ego_view
EPISODES="${EPISODES:-1}"               # episodes per motion
SMOKE="${SMOKE:-0}"
export DISPLAY="${DISPLAY:-:0}"

# short -> full motion key | full-pass frame count (from convert log) | prompt
declare -A KEY=(
  [dance]=dance_in_da_party_001__A464  [lunge]=forward_lunge_R_001__A359_M
  [macarena]=macarena_001__A545        [kick]=neutral_kick_R_001__A543
  [squat]=squat_001__A359              [jump]=tired_one_leg_jumping_R_001__A359
  [walk]=walking_quip_360_R_002__A428
)
declare -A LEN=(
  [dance]=497 [lunge]=399 [macarena]=1375 [kick]=165 [squat]=424 [jump]=500 [walk]=455
)
declare -A PROMPT=(
  [dance]="dance"   [lunge]="do a forward lunge" [macarena]="dance the macarena"
  [kick]="kick"     [squat]="squat"              [jump]="jump on one leg"
  [walk]="walk and turn around"
)
ORDER=(dance lunge macarena kick squat jump walk)

SEL="${1:-${MOTION:-all}}"
[[ "$SEL" == "all" ]] && TARGETS=("${ORDER[@]}") || TARGETS=("$SEL")

cd "$WBC_DIR"
[[ -f "$CKPT" ]] || { echo "[record] $CKPT missing — run scripts/gear_sonic_setup.sh"; exit 1; }
[[ -f "$PKL"  ]] || { echo "[record] $PKL missing — run scripts/gear_sonic_demo.sh first to build it"; exit 1; }

REC_TARGET="gear_sonic.data.vla_token_recorder.VlaTokenRecorder"

for m in "${TARGETS[@]}"; do
  k="${KEY[$m]:-}"; [[ -z "$k" ]] && { echo "[record] unknown motion '$m'"; continue; }
  steps="${LEN[$m]}"; [[ "$SMOKE" == "1" ]] && steps=30
  for (( ep=0; ep<EPISODES; ep++ )); do
    tag=$(printf "%03d" "$ep")
    seed=$(( ${SEED_BASE:-1000} + ep ))   # vary seed per episode -> different startup DR / start phase
    echo "[record] motion=$m key=$k steps=$steps episode=$tag seed=$seed -> $OUT_DIR/$k/episode_$tag.npz"
    conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
        +checkpoint="$CKPT" \
        +headless=True \
        ++seed="$seed" \
        ++num_envs=1 \
        ++manager_env.config.enable_cameras=True \
        "++manager_env.config.cameras.camera_resolution=$CAM_RES" \
        ++manager_env.observations.policy.enable_corruption=False \
        ++manager_env.observations.tokenizer.enable_corruption=False \
        ++manager_env.commands.motion.use_paired_motions=True \
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
        "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$k" \
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
        "++eval_callbacks=[vla_recorder]" \
        "++callbacks.vla_recorder._target_=$REC_TARGET" \
        "++callbacks.vla_recorder.output_dir=$OUT_DIR" \
        "++callbacks.vla_recorder.motion_key=$k" \
        "++callbacks.vla_recorder.prompt=${PROMPT[$m]}" \
        "++callbacks.vla_recorder.max_steps=$steps" \
        "++callbacks.vla_recorder.episode_tag=$tag" \
        || { echo "[record] FAILED on $m ep $tag"; exit 1; }
  done
done

echo "[record] done -> $OUT_DIR"
