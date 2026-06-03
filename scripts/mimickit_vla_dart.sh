#!/usr/bin/env bash
# DART (noise-injected BC, Laskey et al. 2017) data-side fix for the ACT student that
# falls in 0.5-0.8s under pure clean-rollout distillation (covariate shift). We record
# extra rollouts where the EXECUTED action is teacher + N(0, SIGMA) so the G1 visits
# recovery states off the clean tube, but the recorded LABEL stays the clean teacher
# action. Then we merge clean + DART episodes, convert, and retrain ACT identically to
# the clean baseline (chunk_size=16) for an apples-to-apples closed-loop comparison.
#
# Pipeline:  record DART (isaaclab) -> convert+merge (lerobot) -> train ACT (lerobot)
# Closed-loop eval is a separate step: scripts/mimickit_vla_act_closedloop.sh
#
# Usage:   scripts/mimickit_vla_dart.sh
#   knobs: SIGMA=0.07  NUM_DART=50  STEPS=10000  STAGE=all|record|convert|train
#   The go/pivot gate: closed-loop mean survival 20 -> >=50 frames => DART works.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_DIR="$REPO_ROOT/dependencies/MimicKit"
ISAAC_ENV="${ISAAC_ENV:-isaaclab}"
LEROBOT_ENV="${LEROBOT_ENV:-lerobot}"           # ACT trains in lerobot 0.5.2, NOT v040
SIGMA="${SIGMA:-0.07}"                            # action-units noise std (see probe)
NUM_DART="${NUM_DART:-50}"                        # DART episodes per motion
STEPS="${STEPS:-10000}"
STAGE="${STAGE:-all}"                             # all | record | convert | train

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/g1-lafan-vla-dart-img}"
REPO_ID="${REPO_ID:-vitorcen/g1-lafan-vla-dart-img}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/act_g1_lafan_dart}"

MOTIONS=(fight dance)
declare -A PROMPT=(
  [fight]="perform a boxing stance with quick jabs and hooks"
  [dance]="perform a rhythmic dance with arm swings and steps"
)

source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"

# ---- 1. record DART rollouts (isaaclab env, MimicKit) -----------------------
if [[ "$STAGE" == "all" || "$STAGE" == "record" ]]; then
  cd "$MIMICKIT_DIR"
  set +u; conda activate "$ISAAC_ENV"; set -u
  for m in "${MOTIONS[@]}"; do
    OUT="output/vla_rollouts/${m}_dart"
    rm -rf "$OUT"
    echo "================================================"
    echo "[dart] recording $NUM_DART ep of '$m'  sigma=$SIGMA  ($(date +%H:%M:%S))"
    echo "================================================"
    python mimickit/rollout_record.py \
      --env_config    "output/preview_envs/train_lafan_${m}_15s_env.yaml" \
      --engine_config "data/engines/isaac_lab_engine.yaml" \
      --agent_config  "data/agents/deepmimic_g1_ppo_agent.yaml" \
      --model_file    "output/train_lafan_${m}_15s/int_models/model_0000001500.pt" \
      --num_episodes  "$NUM_DART" \
      --task          "${PROMPT[$m]}" \
      --out_dir       "$OUT" \
      --action_noise  "$SIGMA"
  done
fi

# ---- 2. convert: merge clean (10ep) + DART (NUM_DART) per motion ------------
if [[ "$STAGE" == "all" || "$STAGE" == "convert" ]]; then
  set +u; conda activate "$LEROBOT_ENV"; set -u
  cd "$REPO_ROOT"
  DIRS=""
  for m in "${MOTIONS[@]}"; do
    DIRS+="$MIMICKIT_DIR/output/vla_rollouts/${m},"          # clean baseline episodes
    DIRS+="$MIMICKIT_DIR/output/vla_rollouts/${m}_dart,"     # DART episodes
  done
  DIRS="${DIRS%,}"
  echo "================================================"
  echo "[dart] converting (clean+DART merged) -> $DATASET_ROOT"
  echo "  dirs: $DIRS"
  echo "================================================"
  python scripts/mimickit_episodes_to_lerobot.py \
    --rollout_dirs "$DIRS" \
    --repo_id      "$REPO_ID" \
    --root         "$DATASET_ROOT" \
    --no_video \
    --overwrite
fi

# ---- 3. train ACT (lerobot env, chunk_size=16 to match baseline) -----------
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  set +u; conda activate "$LEROBOT_ENV"; set -u
  cd "$REPO_ROOT"
  echo "================================================"
  echo "[dart] training ACT  steps=$STEPS  -> $OUT_DIR"
  echo "================================================"
  rm -rf "$OUT_DIR"
  lerobot-train \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=act \
    --policy.chunk_size=16 \
    --policy.n_action_steps=16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="$OUT_DIR" \
    --job_name=act_g1_lafan_dart \
    --batch_size=32 \
    --steps="$STEPS" \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --save_checkpoint=true \
    --wandb.enable=false
  echo "[dart] training done -> $OUT_DIR/checkpoints"
  echo "  next: scripts/mimickit_vla_act_closedloop.sh  (eval vs clean baseline)"
fi
