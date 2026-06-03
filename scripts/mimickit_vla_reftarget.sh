#!/usr/bin/env bash
# INFO-GAP DIAGNOSTIC (2026-06-03). DART (data coverage) gave +2.3 frames -> the wall is
# NOT covariate shift. Hypothesis: the student falls because it lacks the teacher's
# privileged tar_obs (future reference frames) -- it only gets a 2-dim phase proxy. This
# script tests that by appending the REFERENCE joint target (the tar_obs information) to
# observation.state, holding everything else identical, then retraining ACT.
#
#   decisive read (closed-loop, vs baseline ~21f / DART ~23f):
#     survival jumps (>=50f)  => info gap confirmed: the VLA architecture MUST supply this
#                                info (WBC, or a learned phase->reference predictor).
#     still ~20f              => not info: ACT capacity / action-tracking accuracy wall.
#
# Pipeline: record clean+ref_target (isaaclab) -> convert (lerobot) -> train ACT (lerobot)
# Eval separately:  REF_TARGET=1 NACT=1 scripts/mimickit_vla_act_closedloop.sh <ref_ckpt>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_DIR="$REPO_ROOT/dependencies/MimicKit"
ISAAC_ENV="${ISAAC_ENV:-isaaclab}"
LEROBOT_ENV="${LEROBOT_ENV:-lerobot}"
NUM_EP="${NUM_EP:-15}"                  # clean episodes per motion
STEPS="${STEPS:-10000}"
STAGE="${STAGE:-all}"                   # all | record | convert | train

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/g1-lafan-vla-ref-img}"
REPO_ID="${REPO_ID:-vitorcen/g1-lafan-vla-ref-img}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/act_g1_lafan_ref}"

MOTIONS=(fight dance)
declare -A PROMPT=(
  [fight]="perform a boxing stance with quick jabs and hooks"
  [dance]="perform a rhythmic dance with arm swings and steps"
)

source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"

# ---- 1. record clean rollouts WITH reference target in the state ------------
if [[ "$STAGE" == "all" || "$STAGE" == "record" ]]; then
  cd "$MIMICKIT_DIR"
  set +u; conda activate "$ISAAC_ENV"; set -u
  for m in "${MOTIONS[@]}"; do
    OUT="output/vla_rollouts/${m}_ref"
    rm -rf "$OUT"
    echo "================================================"
    echo "[ref] recording $NUM_EP clean ep of '$m'  +ref_target  ($(date +%H:%M:%S))"
    echo "================================================"
    python mimickit/rollout_record.py \
      --env_config    "output/preview_envs/train_lafan_${m}_15s_env.yaml" \
      --engine_config "data/engines/isaac_lab_engine.yaml" \
      --agent_config  "data/agents/deepmimic_g1_ppo_agent.yaml" \
      --model_file    "output/train_lafan_${m}_15s/int_models/model_0000001500.pt" \
      --num_episodes  "$NUM_EP" \
      --task          "${PROMPT[$m]}" \
      --out_dir       "$OUT" \
      --include_ref_target true
  done
fi

# ---- 2. convert (state_dim auto-picked from rollout_meta = 95 + ndof) -------
if [[ "$STAGE" == "all" || "$STAGE" == "convert" ]]; then
  set +u; conda activate "$LEROBOT_ENV"; set -u
  cd "$REPO_ROOT"
  DIRS=""
  for m in "${MOTIONS[@]}"; do DIRS+="$MIMICKIT_DIR/output/vla_rollouts/${m}_ref,"; done
  DIRS="${DIRS%,}"
  echo "[ref] converting -> $DATASET_ROOT"
  python scripts/mimickit_episodes_to_lerobot.py \
    --rollout_dirs "$DIRS" --repo_id "$REPO_ID" --root "$DATASET_ROOT" --no_video --overwrite
fi

# ---- 3. train ACT (identical to baseline except +ref_target in state) ------
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  set +u; conda activate "$LEROBOT_ENV"; set -u
  cd "$REPO_ROOT"
  echo "[ref] training ACT  steps=$STEPS  -> $OUT_DIR"
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
    --job_name=act_g1_lafan_ref \
    --batch_size=32 \
    --steps="$STEPS" \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=4 \
    --save_checkpoint=true \
    --wandb.enable=false
  echo "[ref] done -> $OUT_DIR/checkpoints"
  echo "  eval:  REF_TARGET=1 NACT=1 NUM_EP=5 scripts/mimickit_vla_act_closedloop.sh \\"
  echo "         $OUT_DIR/checkpoints/last/pretrained_model"
fi
