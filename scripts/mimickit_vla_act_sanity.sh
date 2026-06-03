#!/usr/bin/env bash
# 激进 sanity step 2: train a small ACT on the g1-lafan-vla-sanity dataset (fight+dance)
# in lerobot-v040, then closed-loop check whether prompt → motion works (go/pivot gate).
#
# codex review 🟡#3: ACT only consumes observation.state when robot_state_feature exists.
# We hard-assert the dataset carries observation.state before training — otherwise the
# phase channel is silently dropped and a 0-result can't distinguish "phase scheme bad"
# from "state never wired".
#
# Usage:  scripts/mimickit_vla_act_sanity.sh
#   env overrides: STEPS=40000 BATCH=32 LEROBOT_ENV=lerobot-v040
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# lerobot 0.5.2 (torch 2.10), NOT lerobot-v040: v0.4.0's torch 2.7.1 editable install
# crashes this training path with intermittent interpreter-level corruption (sre_compile
# / posixpath UnboundLocalError / SIGSEGV during torch._dynamo import). 0.5.2 trains ACT
# clean. The v0.4-locks-ACT-quality rule applies to final benchmarks, not this go/pivot.
LEROBOT_ENV="${LEROBOT_ENV:-lerobot}"
# image dataset (use_videos=False): video decode (torchcodec AND pyav, any worker count)
# segfaults in this env, so the ACT sanity trains on a png-frame dataset — zero ffmpeg.
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/g1-lafan-vla-sanity-img}"
REPO_ID="${REPO_ID:-vitorcen/g1-lafan-vla-sanity-img}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/act_g1_lafan_sanity}"
STEPS="${STEPS:-40000}"
BATCH="${BATCH:-32}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
NWORKERS="${NWORKERS:-2}"

source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
# the lerobot env's binutils activate.d script references unbound vars (ADDR2LINE …);
# set -u would abort on it, so relax nounset just across activation.
set +u
conda activate "$LEROBOT_ENV"
set -u
cd "$REPO_ROOT"

# ---- hard pre-flight: observation.state must be present (codex 🟡#3) ----
python - "$REPO_ID" "$DATASET_ROOT" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata(sys.argv[1], root=sys.argv[2])
feats = meta.features
assert "observation.state" in feats, "dataset has no observation.state — ACT would drop phase; ABORT"
dim = feats["observation.state"]["shape"][0]
assert dim >= 10, f"observation.state dim={dim} too small; phase/proprio missing"
print(f"[pre-flight] observation.state OK: dim={dim}  (ACT will consume it)")
print(f"[pre-flight] action dim={feats['action']['shape'][0]}, tasks carried in episodes")
PY

echo "================================================"
echo "[act-sanity] training ACT  steps=$STEPS batch=$BATCH  -> $OUT_DIR"
echo "================================================"
rm -rf "$OUT_DIR"
# image dataset → png frames, no video decode. Data loading is the bottleneck (GPU is
# idle), so throw workers at it (NWORKERS).
lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT_DIR" \
  --job_name=act_g1_lafan_sanity \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=200 \
  --num_workers="$NWORKERS" \
  --save_checkpoint=true \
  --wandb.enable=false

echo "================================================"
echo "[act-sanity] training done -> $OUT_DIR/checkpoints"
echo "  next: closed-loop eval (load ACT into MimicKit env, prompt fight/dance)"
echo "================================================"
