#!/usr/bin/env bash
# Sequential GUI eval of MimicKit LAFAN policies — open one window at a time,
# wait for user to close it, then advance to the next.
#
# Usage:  scripts/mimickit_eval_chain.sh [motion1 motion2 ...]
#   default order: lafan_fight_15s lafan_run_15s lafan_dance_15s lafan_jumps_15s
#
# Source resolution per motion:
#   1. locally trained:  output/train_<motion>/int_models/model_0000001500.pt
#   2. fallback to HF:    snapshot_download($MIMICKIT_HF_REPO) → ~/.cache/huggingface
#      (download once, content-addressed cache, zero-copy reuse of the 11MB ckpt
#       + 26MB textured USD; only the 144KB motion pkl is staged into data/.)
# So users who never trained anything just run this and it pulls our weights.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_DIR="$REPO_ROOT/dependencies/MimicKit"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
NUM_ENVS="${NUM_ENVS:-4}"
HF_REPO="${MIMICKIT_HF_REPO:-wsagi/MimicKit-G1-LAFAN}"

if [[ $# -eq 0 ]]; then
  MOTIONS=(lafan_fight_15s lafan_run_15s lafan_dance_15s lafan_jumps_15s)
else
  MOTIONS=("$@")
fi

cd "$MIMICKIT_DIR"
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

PRETRAINED_DIR=""   # lazily resolved from HF on the first local miss

for motion in "${MOTIONS[@]}"; do
  CKPT="output/train_${motion}/int_models/model_0000001500.pt"
  ENV_YAML="output/preview_envs/train_${motion}_env.yaml"

  # Not trained locally → fall back to the HF bundle (downloads once, then cached).
  if [[ ! -f "$CKPT" || ! -f "$ENV_YAML" ]]; then
    if [[ -z "$PRETRAINED_DIR" ]]; then
      echo "[eval_chain] no local ckpt → fetching $HF_REPO into HF cache…"
      PRETRAINED_DIR="$(python - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]), end="")
PY
)"
      echo "[eval_chain] HF snapshot: $PRETRAINED_DIR"
    fi
    sub="${motion#lafan_}"                       # lafan_fight_15s → fight_15s
    CKPT="$PRETRAINED_DIR/$sub/model_0000001500.pt"
    ENV_YAML="$PRETRAINED_DIR/$sub/env.yaml"
    # env.yaml references data/motions/g1/<motion>.pkl relative to CWD — stage the
    # tiny (144KB) pkl there; the big files stay in cache, referenced in place.
    if [[ -f "$PRETRAINED_DIR/$sub/motion.pkl" ]]; then
      mkdir -p data/motions/g1
      cp -n "$PRETRAINED_DIR/$sub/motion.pkl" "data/motions/g1/${motion}.pkl"
    fi
    # textured USD from the same bundle, unless the caller pinned one already.
    if [[ -z "${MIMICKIT_G1_USD:-}" && -f "$PRETRAINED_DIR/assets/g1_textured.usd" ]]; then
      export MIMICKIT_G1_USD="$PRETRAINED_DIR/assets/g1_textured.usd"
    fi
  fi

  if [[ ! -f "$CKPT" || ! -f "$ENV_YAML" ]]; then
    echo "[eval_chain] SKIP $motion — not found locally or on HF ($CKPT)"
    continue
  fi
  mkdir -p "output/eval_${motion}"
  echo "================================================"
  echo "[eval_chain] >>> $motion  $(date +%H:%M:%S)"
  echo "  ckpt: $CKPT"
  echo "  关闭 Isaac Sim 窗口以进入下一个 motion"
  echo "================================================"
  python mimickit/run.py --mode test --num_envs "$NUM_ENVS" \
    --engine_config data/engines/isaac_lab_engine.yaml \
    --env_config "$ENV_YAML" \
    --agent_config data/agents/deepmimic_g1_ppo_agent.yaml \
    --visualize true \
    --model_file "$CKPT" \
    --out_dir "output/eval_${motion}" --logger txt \
    > "output/eval_${motion}/run.log" 2>&1 || true
  echo "[eval_chain] <<< $motion closed  $(date +%H:%M:%S)"
done

echo "[eval_chain] all motions reviewed"
