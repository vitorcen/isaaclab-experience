#!/usr/bin/env bash
# Pre-fetch a HuggingFace model into the default HF cache
# (~/.cache/huggingface/hub/). Native idempotent — re-runs just verify etags.
#
# Usage:
#   bash scripts/download_hf_model.sh REPO_ID
#
# Examples:
#   bash scripts/download_hf_model.sh LightwheelAI/leisaac-pick-orange-v0
#   bash scripts/download_hf_model.sh edge-inference/smolvla-so101-pick-orange
#
# After this, downstream code can refer to the model purely by repo_id:
#   from transformers import AutoModel
#   AutoModel.from_pretrained("LightwheelAI/leisaac-pick-orange-v0")
#
# Inspect cache: huggingface-cli scan-cache

set -euo pipefail

REPO_ID="${1:-}"
if [ -z "$REPO_ID" ]; then
    echo "usage: $0 REPO_ID" >&2
    exit 2
fi

echo "[INFO] downloading $REPO_ID -> HF default cache"
# `hf download` is the v0.34+ replacement for the deprecated
# `huggingface-cli download`. Fall back to the python API if neither is present.
if command -v hf >/dev/null 2>&1; then
    hf download "$REPO_ID"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO_ID"
else
    python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="$REPO_ID"))
PY
fi
echo "[DONE] $REPO_ID"
