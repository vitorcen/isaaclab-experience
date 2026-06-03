#!/usr/bin/env bash
# Launch the ACT closed-loop eval with the Isaac Sim GUI so you can WATCH the student
# drive the G1 live. ACT runs in the lerobot env behind the socket server; the MimicKit
# env opens a GUI window in the isaaclab env.
#
# Usage:  scripts/mimickit_vla_act_gui.sh [fight|dance] [num_episodes]
set -uo pipefail
ROOT=/home/david/work/isaaclab-experience
MK=$ROOT/dependencies/MimicKit
CKPT=${CKPT:-$ROOT/outputs/act_g1_lafan_sanity/checkpoints/last/pretrained_model}
MOTION=${1:-fight}
NUM_EP=${2:-30}
PORT=${PORT:-5631}
NACT=${NACT:-}     # set to 1 for per-step replanning; empty = as-trained (16)
export DISPLAY=${DISPLAY:-:0}

case $MOTION in
  fight) PROMPT="perform a boxing stance with quick jabs and hooks";;
  dance) PROMPT="perform a rhythmic dance with arm swings and steps";;
  *) echo "unknown motion $MOTION"; exit 1;;
esac

source "$(conda info --base)/etc/profile.d/conda.sh"
pkill -9 -f act_policy_server 2>/dev/null || true
sleep 1
rm -f /tmp/act_gui_server.log

echo "[gui] starting ACT server on :$PORT  (n_action_steps=${NACT:-as-trained})"
NACT_ARG=()
[[ -n "$NACT" ]] && NACT_ARG=(--n_action_steps "$NACT")
conda run --no-capture-output -n lerobot \
  python "$ROOT/scripts/act_policy_server.py" --ckpt "$CKPT" --port $PORT "${NACT_ARG[@]}" \
  > /tmp/act_gui_server.log 2>&1 &
SPID=$!
trap 'kill -9 $SPID 2>/dev/null || true; pkill -9 -f act_policy_server 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  grep -q "listening on" /tmp/act_gui_server.log 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "[gui] server died:"; cat /tmp/act_gui_server.log; exit 1; }
  sleep 1
done
echo "[gui] server ready — opening Isaac Sim GUI for '$MOTION' ($NUM_EP episodes)"

cd "$MK"
conda run --no-capture-output -n isaaclab \
  python mimickit/rollout_act_eval.py \
    --env_config    "output/preview_envs/train_lafan_${MOTION}_15s_env.yaml" \
    --engine_config "data/engines/isaac_lab_engine.yaml" \
    --agent_config  "data/agents/deepmimic_g1_ppo_agent.yaml" \
    --task          "$PROMPT" \
    --num_episodes  "$NUM_EP" \
    --server_port   "$PORT" \
    --max_steps     450 \
    --visualize     true \
    --out_dir       "output/vla_act_eval_gui/${MOTION}"

echo "[gui] window closed / episodes done"
