#!/usr/bin/env bash
# 激进 sanity (design doc §7.2 step 2): record fight + dance rollouts from the
# trained DeepMimic PPO experts, convert to one LeRobot dataset, report dims.
# This is the go/pivot gate BEFORE scaling — see doc/mimickit_to_vla_dataset.html.
#
# Usage:  scripts/mimickit_vla_sanity.sh
#   env overrides: NUM_EP=10  ISAAC_ENV=isaaclab  LEROBOT_ENV=lerobot-v040
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_DIR="$REPO_ROOT/dependencies/MimicKit"
ISAAC_ENV="${ISAAC_ENV:-isaaclab}"
LEROBOT_ENV="${LEROBOT_ENV:-lerobot-v040}"
NUM_EP="${NUM_EP:-10}"
REPO_ID="${REPO_ID:-vitorcen/g1-lafan-vla-sanity}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/g1-lafan-vla-sanity}"

MOTIONS=(fight dance)
declare -A PROMPT=(
  [fight]="perform a boxing stance with quick jabs and hooks"
  [dance]="perform a rhythmic dance with arm swings and steps"
)

source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"

# ---- 1. record rollouts (isaaclab env, MimicKit) ----------------------------
cd "$MIMICKIT_DIR"
conda activate "$ISAAC_ENV"
ROLLOUT_DIRS=()
for m in "${MOTIONS[@]}"; do
  OUT="output/vla_rollouts/${m}"
  rm -rf "$OUT"
  echo "================================================"
  echo "[sanity] recording $NUM_EP episodes of '$m'  ($(date +%H:%M:%S))"
  echo "================================================"
  python mimickit/rollout_record.py \
    --env_config    "output/preview_envs/train_lafan_${m}_15s_env.yaml" \
    --engine_config "data/engines/isaac_lab_engine.yaml" \
    --agent_config  "data/agents/deepmimic_g1_ppo_agent.yaml" \
    --model_file    "output/train_lafan_${m}_15s/int_models/model_0000001500.pt" \
    --num_episodes  "$NUM_EP" \
    --task          "${PROMPT[$m]}" \
    --out_dir       "$OUT"
  ROLLOUT_DIRS+=("$MIMICKIT_DIR/$OUT")
done

# ---- 2. convert to LeRobot + modality.json (lerobot env) --------------------
conda activate "$LEROBOT_ENV"
cd "$REPO_ROOT"
DIRS_CSV=$(IFS=,; echo "${ROLLOUT_DIRS[*]}")
echo "================================================"
echo "[sanity] converting -> $DATASET_ROOT"
echo "================================================"
python scripts/mimickit_episodes_to_lerobot.py \
  --rollout_dirs "$DIRS_CSV" \
  --repo_id      "$REPO_ID" \
  --root         "$DATASET_ROOT" \
  --overwrite

echo "================================================"
echo "[sanity] dataset ready: $DATASET_ROOT"
echo "  next: train a small ACT/DP in $LEROBOT_ENV and closed-loop check (go/pivot)"
echo "================================================"
