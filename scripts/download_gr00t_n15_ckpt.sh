#!/usr/bin/env bash
# Download the LightwheelAI/leisaac-pick-orange-v0 GR00T N1.5 fine-tuned ckpt
# (~7.4 GB) to ~/models/leisaac-pick-orange-v0. Idempotent.

set -euo pipefail

DEST="${LEISAAC_N15_CKPT_DIR:-$HOME/models/leisaac-pick-orange-v0}"
REPO_ID="LightwheelAI/leisaac-pick-orange-v0"

if [ -f "$DEST/config.json" ] && [ -f "$DEST/model.safetensors.index.json" ]; then
    have=$(du -sh "$DEST" 2>/dev/null | awk '{print $1}')
    echo "[INFO] checkpoint already at $DEST ($have)"
    exit 0
fi

mkdir -p "$DEST"
echo "[INFO] downloading $REPO_ID -> $DEST (~7.4 GB)"

# Prefer huggingface-cli; fall back to python snapshot_download.
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO_ID" --local-dir "$DEST"
else
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$REPO_ID", local_dir="$DEST")
PY
fi
echo "[DONE] $DEST"
