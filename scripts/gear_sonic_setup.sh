#!/usr/bin/env bash
# One-time setup for NVIDIA GEAR-SONIC — the G1 whole-body controller (WBC) that walks /
# runs / jumps. This is the "balance base" for the WBC + finetune route: download it,
# preview it (gear_sonic_preview.sh), then finetune GR00T-N1.7 on top to emit SONIC
# latents for dance (no pre-finetuned G1 VLA checkpoint exists — must finetune ourselves).
#
# Heavy prereqs (per NVIDIA quickstart): TensorRT + a C++ deployment build + a dedicated
# .venv_sim. This script follows the documented commands; first run takes a while.
#
#   docs:  https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEP_DIR="$REPO_ROOT/dependencies"
WBC_DIR="$DEP_DIR/GR00T-WholeBodyControl"
ISAAC_ENV="${ISAAC_ENV:-isaaclab}"   # conda env holding Isaac Lab; preview/eval runs here

echo "============================================================"
echo "[gear-sonic] 1/3  init GR00T-WholeBodyControl submodule + apply patches"
echo "============================================================"
# Managed as a git submodule (pinned upstream commit); our local edits live as
# diffs under patches/gear-sonic/ and are re-applied idempotently here. This keeps
# the submodule working tree pristine-plus-patches and reproducible on fresh clones.
git -C "$REPO_ROOT" submodule update --init dependencies/GR00T-WholeBodyControl
bash "$REPO_ROOT/scripts/apply_gear_sonic_patches.sh"
cd "$WBC_DIR"

echo "============================================================"
echo "[gear-sonic] 2/4  install Isaac Lab eval deps into the '$ISAAC_ENV' conda env"
echo "  these are what the default preview (gear_sonic_preview.sh) needs — gear_sonic"
echo "  itself ships no requirements file, so we pin the few it imports at runtime."
echo "============================================================"
# easydict/loguru: tiny utils.  open3d: motion_lib mesh I/O (hard import).
# vector_quantize_pytorch: SONIC's VQ tokenizer (loaded via config _target_).
conda run --no-capture-output -n "$ISAAC_ENV" pip install -q \
  easydict loguru open3d vector_quantize_pytorch || \
  echo "[gear-sonic] WARN: eval-deps install failed — preview will error on import."
# trl 0.28.0 is the transition release the repo targets (both trl.trainer.ppo_trainer
# AND trl.experimental.ppo exist); newer trl removed the old path. --no-deps so it
# does not churn transformers/accelerate (repo needs transformers>=4.56.2, already met).
conda run --no-capture-output -n "$ISAAC_ENV" pip install -q --no-deps "trl==0.28.0" || \
  echo "[gear-sonic] WARN: trl==0.28.0 pin failed — eval will hit trl.trainer.ppo_trainer ImportError."

echo "============================================================"
echo "[gear-sonic] 3/4  (optional, sim2sim only) install MuJoCo sim deps — heavy"
echo "  ONLY needed for gear_sonic_preview_sim2sim.sh (DDS + C++ deploy). The default"
echo "  Isaac-eval preview does NOT need this. Needs TensorRT + builds C++ deployment."
echo "============================================================"
if [[ "${WITH_SIM2SIM:-0}" == "1" && -f install_scripts/install_mujoco_sim.sh ]]; then
  bash install_scripts/install_mujoco_sim.sh || {
    echo "[gear-sonic] install_mujoco_sim.sh failed — check repo README / TensorRT install."; }
else
  echo "[gear-sonic] skipped (set WITH_SIM2SIM=1 to build the MuJoCo sim2sim route)."
fi

echo "============================================================"
echo "[gear-sonic] 4/4  download GEAR-SONIC checkpoints (nvidia/GEAR-SONIC)"
echo "  encoder / decoder / planner ONNX + sample motions"
echo "============================================================"
if [[ -f download_from_hf.py ]]; then
  python download_from_hf.py --training --no-smpl || python download_from_hf.py --training || true
  python download_from_hf.py --sample || true
else
  echo "[gear-sonic] download_from_hf.py not found — fallback to snapshot_download:"
  python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="nvidia/GEAR-SONIC", local_dir="gear_sonic_deploy")
print("downloaded GEAR-SONIC ->", p)
PY
fi

echo "============================================================"
echo "[gear-sonic] setup done. repo: $WBC_DIR"
echo "  next: scripts/gear_sonic_preview.sh  (opens the keyboard-driven MuJoCo viewer)"
echo "============================================================"
