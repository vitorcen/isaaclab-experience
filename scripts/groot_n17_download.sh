#!/usr/bin/env bash
# Download the GR00T-N1.7-3B base VLA — the "brain" for the WBC + finetune route. This is
# the base only (NOT a finetuned G1 policy; no such checkpoint is published — we finetune
# it ourselves to emit UNITREE_G1_SONIC latents that GEAR-SONIC decodes into balanced
# whole-body motion). On its own it cannot drive the G1; it needs the SONIC WBC + finetune.
#
#   gated: nvidia/GR00T-N1.7-3B may require `huggingface-cli login` + access approval.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/dependencies/checkpoints/GR00T-N1.7-3B}"
REPO_ID="${REPO_ID:-nvidia/GR00T-N1.7-3B}"

echo "[groot] downloading $REPO_ID -> $DEST"
mkdir -p "$DEST"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$REPO_ID" --local-dir "$DEST" || {
  echo "[groot] download failed — likely gated. Run:  huggingface-cli login"
  echo "        and request access at https://huggingface.co/$REPO_ID"
  exit 1
}
echo "[groot] done. base VLA at $DEST"
echo "  next (finetune route): encode MimicKit motions through the SONIC encoder to get"
echo "  latent action labels, build a UNITREE_G1_SONIC LeRobot dataset, then finetune via"
echo "  Isaac-GR00T with --base-model-path $REPO_ID."
