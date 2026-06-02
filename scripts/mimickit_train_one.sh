#!/usr/bin/env bash
# Train a single MimicKit DeepMimic G1 policy for one motion profile.
# Usage:  scripts/mimickit_train_one.sh <motion_name> [max_iters] [init_ckpt]
#
# motion_name: pkl basename without extension. Must exist in
#   dependencies/MimicKit/data/motions/g1/<motion_name>.pkl
#
# max_iters: stop training when this iteration is reached (default 1500).
#   Set to 0 to never auto-stop (manual kill only).
#
# init_ckpt: optional path to a .pt to warm-start from. Empty = from scratch.
#
# Output: dependencies/MimicKit/output/train_<motion_name>/
set -euo pipefail

MOTION="${1:?usage: $0 <motion_name> [max_iters] [init_ckpt]}"
MAX_ITERS="${2:-1500}"
INIT_CKPT="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_DIR="$REPO_ROOT/dependencies/MimicKit"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
NUM_ENVS="${NUM_ENVS:-4096}"
BASE_ENV="${BASE_ENV:-deepmimic_g1_env}"
AGENT="${AGENT:-deepmimic_g1_ppo_agent}"

cd "$MIMICKIT_DIR"

MOTION_PKL="data/motions/g1/${MOTION}.pkl"
if [[ ! -f "$MOTION_PKL" ]]; then
  echo "❌ Missing $MOTION_PKL" >&2; exit 2
fi

mkdir -p output/preview_envs "output/train_${MOTION}"
ENV_YAML="output/preview_envs/train_${MOTION}_env.yaml"
# Swap motion_file in env yaml. Find current motion path and replace.
DEFAULT_MOTION=$(grep -oP 'motion_file:\s*"\K[^"]+' "data/envs/${BASE_ENV}.yaml")
sed "s|${DEFAULT_MOTION}|${MOTION_PKL}|" "data/envs/${BASE_ENV}.yaml" > "$ENV_YAML"

source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

ARGS=(
  --mode train --num_envs "$NUM_ENVS"
  --engine_config data/engines/isaac_lab_engine.yaml
  --env_config "$ENV_YAML"
  --agent_config "data/agents/${AGENT}.yaml"
  --visualize false --out_dir "output/train_${MOTION}"
  --logger txt --save_int_models true
)
[[ -n "$INIT_CKPT" ]] && ARGS+=(--model_file "$INIT_CKPT")

echo "🚀 train_one: motion=$MOTION  max_iters=$MAX_ITERS  init=${INIT_CKPT:-scratch}"
nohup python mimickit/run.py "${ARGS[@]}" > "output/train_${MOTION}/run.log" 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"

# Monitor — exit when iter reaches MAX_ITERS or process dies
while true; do
  sleep 30
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "[mimickit_train_one] process gone — exiting"
    break
  fi
  if [[ "$MAX_ITERS" -gt 0 ]]; then
    CUR_ITER=$(grep -aoE "Iteration \|\s+[0-9]+" "output/train_${MOTION}/run.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$" || echo 0)
    if [[ "${CUR_ITER:-0}" -ge "$MAX_ITERS" ]]; then
      echo "[mimickit_train_one] reached iter $CUR_ITER >= $MAX_ITERS — SIGTERM"
      kill -TERM "$TRAIN_PID" 2>/dev/null || true
      # Wait up to 60s for graceful, then SIGKILL
      for _ in $(seq 1 12); do
        sleep 5
        kill -0 "$TRAIN_PID" 2>/dev/null || break
      done
      kill -9 "$TRAIN_PID" 2>/dev/null || true
      break
    fi
  fi
done

LATEST=$(ls -t "output/train_${MOTION}/int_models/"model_*.pt 2>/dev/null | head -1 || echo "")
echo "[mimickit_train_one] done. Latest ckpt: ${LATEST}"
