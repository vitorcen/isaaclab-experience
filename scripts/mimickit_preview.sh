#!/usr/bin/env bash
# MimicKit pretrained-policy preview launcher (Isaac Lab backend).
# Usage:  scripts/mimickit_preview.sh <profile>
# Profiles:
#   spinkick   — Unitree G1 旋踢   (deepmimic PPO)
#   jump       — Unitree G1 双跳箱 (deepmimic PPO, double_kong)
#   run        — Unitree G1 跑步   (ADD)
#   walk       — Unitree G1 走路   (LCP)
#   cartwheel  — Unitree G1 翻车轮 (无 ckpt → mocap kinematic 回放)
#   go2_pace   — Unitree Go2 溜花步 (deepmimic PPO)
#   lafan_fight — Unitree G1 LAFAN1 4min 格斗 (无 ckpt → mocap 回放)
#   lafan_dance — Unitree G1 LAFAN1 2min 舞蹈 (无 ckpt → mocap 回放)
#   lafan_jumps — Unitree G1 LAFAN1 4min 跳跃 (无 ckpt → mocap 回放)
#   lafan_run   — Unitree G1 LAFAN1 4min 奔跑 (无 ckpt → mocap 回放)
#   lafan_fight_5s  — LAFAN_fight 前 5s 切段（训练 warm-up 目标）
#   lafan_fight_15s — LAFAN_fight 中段 15s 切段（训练主目标）
#
# Env overrides: MIMICKIT_DIR (default = repo/dependencies/MimicKit),
#                CONDA_ENV (default = isaaclab), NUM_ENVS (default = 4).
set -euo pipefail

PROFILE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_DIR="${MIMICKIT_DIR:-$REPO_ROOT/dependencies/MimicKit}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
NUM_ENVS="${NUM_ENVS:-4}"

case "$PROFILE" in
  spinkick)  ENV_BASE=deepmimic_g1_env;   AGENT=deepmimic_g1_ppo_agent;  MODEL=deepmimic_g1_spinkick_model.pt;     MOTION_DIR=g1;  MOTION=g1_spinkick.pkl ;;
  jump)      ENV_BASE=deepmimic_g1_env;   AGENT=deepmimic_g1_ppo_agent;  MODEL=deepmimic_g1_double_kong_model.pt;  MOTION_DIR=g1;  MOTION=g1_double_kong.pkl ;;
  run)       ENV_BASE=add_g1_env;         AGENT=add_g1_agent;            MODEL=add_g1_run_model.pt;                MOTION_DIR=g1;  MOTION=g1_run.pkl ;;
  walk)      ENV_BASE=deepmimic_g1_env;   AGENT=lcp_g1_agent;            MODEL=lcp_g1_walk_model.pt;               MOTION_DIR=g1;  MOTION=g1_walk.pkl ;;
  cartwheel) ENV_BASE=view_motion_g1_env; AGENT="";                      MODEL="";                                  MOTION_DIR=g1;  MOTION=g1_cartwheel.pkl ;;
  go2_pace)  ENV_BASE=deepmimic_go2_env;  AGENT=deepmimic_go2_ppo_agent; MODEL=deepmimic_go2_pace_model.pt;        MOTION_DIR=go2; MOTION=go2_pace.pkl ;;
  lafan_fight) ENV_BASE=view_motion_g1_env; AGENT=""; MODEL=""; MOTION_DIR=g1; MOTION=lafan_fight1.pkl ;;
  lafan_dance) ENV_BASE=view_motion_g1_env; AGENT=""; MODEL=""; MOTION_DIR=g1; MOTION=lafan_dance1.pkl ;;
  lafan_jumps) ENV_BASE=view_motion_g1_env; AGENT=""; MODEL=""; MOTION_DIR=g1; MOTION=lafan_jumps1.pkl ;;
  lafan_run)   ENV_BASE=view_motion_g1_env; AGENT=""; MODEL=""; MOTION_DIR=g1; MOTION=lafan_run1.pkl ;;
  lafan_fight_5s)  ENV_BASE=view_motion_g1_env; AGENT=""; MODEL=""; MOTION_DIR=g1; MOTION=lafan_fight_5s.pkl ;;
  lafan_fight_15s) ENV_BASE=view_motion_g1_env; AGENT=""; MODEL=""; MOTION_DIR=g1; MOTION=lafan_fight_15s.pkl ;;
  *) echo "Usage: $0 {spinkick|jump|run|walk|cartwheel|go2_pace|lafan_fight|lafan_dance|lafan_jumps|lafan_run|lafan_fight_5s|lafan_fight_15s}" >&2; exit 2 ;;
esac

cd "$MIMICKIT_DIR"

# Asset gate — both 35 MB ckpts and mocap pkls only ship via SharePoint zip
if [[ -n "$MODEL" && ! -f "data/models/$MODEL" ]]; then
  echo "❌ Missing data/models/$MODEL" >&2
  echo "   Extract MimicKit_Data.zip into $MIMICKIT_DIR/data/ first." >&2
  echo "   URL: https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w" >&2
  exit 3
fi
if [[ ! -f "data/motions/$MOTION_DIR/$MOTION" ]]; then
  echo "❌ Missing data/motions/$MOTION_DIR/$MOTION (SharePoint zip not extracted?)" >&2
  exit 3
fi

# Auto-apply Isaac Lab v2.3+ API patch if needed
if grep -q "traverse_instance_prims=True" mimickit/engines/isaac_lab_engine.py 2>/dev/null; then
  echo "🩹 Applying patches/mimickit/isaaclab-v23-api.patch ..."
  git apply "$REPO_ROOT/patches/mimickit/isaaclab-v23-api.patch"
fi

# Generate temp env yaml with motion swap (keeps submodule clean)
mkdir -p output/preview_envs
ENV_YAML="output/preview_envs/${PROFILE}_env.yaml"
DEFAULT_MOTION=$(grep -oP 'motion_file:\s*"\K[^"]+' "data/envs/${ENV_BASE}.yaml" | awk -F/ '{print $NF}')
sed "s|${DEFAULT_MOTION}|${MOTION}|" "data/envs/${ENV_BASE}.yaml" > "$ENV_YAML"

# Activate conda
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

OUT_DIR="output/preview_${PROFILE}"
mkdir -p "$OUT_DIR"

ARGS=(
  --mode test
  --num_envs "$NUM_ENVS"
  --engine_config data/engines/isaac_lab_engine.yaml
  --env_config "$ENV_YAML"
  --visualize true
  --out_dir "$OUT_DIR"
  --logger txt
)
[[ -n "$AGENT" ]] && ARGS+=(--agent_config "data/agents/${AGENT}.yaml")
[[ -n "$MODEL" ]] && ARGS+=(--model_file "data/models/${MODEL}")

echo "🚀 MimicKit preview: $PROFILE"
echo "   env=$ENV_YAML"
echo "   agent=${AGENT:-Dummy (no policy)}"
echo "   model=${MODEL:-N/A}"
echo "   motion=data/motions/$MOTION_DIR/$MOTION"
echo "---"
exec python mimickit/run.py "${ARGS[@]}"
