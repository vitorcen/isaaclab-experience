#!/usr/bin/env bash
# Closed-loop go/pivot eval: ACT student drives the MimicKit G1 for fight + dance,
# records mp4s to eyeball whether prompt -> recognizable, balanced motion.
#
# Env-split: ACT runs in the lerobot env behind a socket server; the MimicKit env runs
# in the isaaclab env as the client. Both are launched with `conda run` so there is no
# activate/deactivate juggling, and the server stays a child of THIS script (one process
# tree) so it survives for the whole client run. See act_policy_server.py / rollout_act_eval.py.
#
# Usage:  scripts/mimickit_vla_act_closedloop.sh [ckpt_dir]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_DIR="$REPO_ROOT/dependencies/MimicKit"
ISAAC_ENV="${ISAAC_ENV:-isaaclab}"
LEROBOT_ENV="${LEROBOT_ENV:-lerobot}"
CKPT="${1:-$REPO_ROOT/outputs/act_g1_lafan_sanity/checkpoints/last/pretrained_model}"
PORT="${PORT:-5599}"
NUM_EP="${NUM_EP:-3}"
MAX_STEPS="${MAX_STEPS:-400}"
NACT="${NACT:-}"   # override execution horizon (1 = replan every step, best for balance)
REF_TARGET="${REF_TARGET:-}"   # 1 => student state includes reference joint target (info-gap diagnostic)
OUT_ROOT="$MIMICKIT_DIR/output/vla_act_eval"

declare -A PROMPT=(
  [fight]="perform a boxing stance with quick jabs and hooks"
  [dance]="perform a rhythmic dance with arm swings and steps"
)
MOTIONS=(fight dance)

source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"

# ---- 1. ACT server (lerobot env), kept as a child of this script ------------
echo "[closedloop] starting ACT server on :$PORT  ($CKPT)  n_action_steps=${NACT:-default}"
NACT_ARG=()
[[ -n "$NACT" ]] && NACT_ARG=(--n_action_steps "$NACT")
conda run --no-capture-output -n "$LEROBOT_ENV" \
  python "$REPO_ROOT/scripts/act_policy_server.py" --ckpt "$CKPT" --port "$PORT" "${NACT_ARG[@]}" \
  > /tmp/act_server.log 2>&1 &
SERVER_PID=$!
trap 'kill -9 $SERVER_PID 2>/dev/null; pkill -9 -f act_policy_server 2>/dev/null' EXIT
for i in $(seq 1 180); do
  grep -q "listening on" /tmp/act_server.log 2>/dev/null && break
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "[closedloop] server died:"; cat /tmp/act_server.log; exit 1; fi
  sleep 1
done
echo "[closedloop] server ready"

# ---- 2. MimicKit client per motion (isaaclab env) ---------------------------
cd "$MIMICKIT_DIR"
for m in "${MOTIONS[@]}"; do
  OUT="output/vla_act_eval/${m}"
  rm -rf "$OUT"
  echo "================================================"
  echo "[closedloop] >>> $m  ($(date +%H:%M:%S))"
  echo "================================================"
  REF_ARG=()
  [[ "$REF_TARGET" == "1" ]] && REF_ARG=(--include_ref_target true)
  conda run --no-capture-output -n "$ISAAC_ENV" \
    python mimickit/rollout_act_eval.py \
      --env_config    "output/preview_envs/train_lafan_${m}_15s_env.yaml" \
      --engine_config "data/engines/isaac_lab_engine.yaml" \
      --agent_config  "data/agents/deepmimic_g1_ppo_agent.yaml" \
      --task          "${PROMPT[$m]}" \
      --num_episodes  "$NUM_EP" \
      --server_port   "$PORT" \
      --max_steps     "$MAX_STEPS" \
      --out_dir       "$OUT" \
      "${REF_ARG[@]}"
done

echo "================================================"
echo "[closedloop] done. mp4s under $OUT_ROOT/{fight,dance}/"
find "$OUT_ROOT" -name "*.mp4" 2>/dev/null
echo "================================================"
