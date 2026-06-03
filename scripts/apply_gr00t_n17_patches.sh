#!/usr/bin/env bash
# Apply our local patches to the Isaac-GR00T (N1.7) submodule.
# Idempotent: an already-applied patch is skipped.
# Run after `git submodule update --init dependencies/Isaac-GR00T`.
#
# Covers the architecture-side VLA route (GR00T -> SONIC token) finetune fix only:
#   0001  GR00T_OPTIM=adafactor + GR00T_GRAD_CKPT=1 so the 1.6B-trainable action head
#         fits a 24GB 4090 (adamw momentum state OOMs). Defaults preserve upstream.
# NOTE: the unitree_g1_sonic embodiment config/tag is already committed upstream in this
# checkout (commit "Add SONIC embodiment"); only the OOM-fit edits need re-applying.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_DIR="$ROOT_DIR/dependencies/Isaac-GR00T"
PATCH_DIR="$ROOT_DIR/patches/gr00t-n17"

if [ ! -e "$GR00T_DIR/.git" ]; then
    echo "[ERROR] Isaac-GR00T submodule not initialized at $GR00T_DIR" >&2
    echo "        Run: git submodule update --init dependencies/Isaac-GR00T" >&2
    exit 1
fi

shopt -s nullglob
patches=( "$PATCH_DIR"/*.patch )
if [ ${#patches[@]} -eq 0 ]; then
    echo "[INFO] no patches under $PATCH_DIR"
    exit 0
fi

cd "$GR00T_DIR"
for p in "${patches[@]}"; do
    name="$(basename "$p")"
    if git apply --check --reverse "$p" >/dev/null 2>&1; then
        echo "[SKIP] $name (already applied)"
        continue
    fi
    if ! git apply --check "$p" >/dev/null 2>&1; then
        echo "[ERROR] $name does not apply cleanly. Upstream may have moved." >&2
        echo "        Inspect: git apply --reject $p" >&2
        exit 1
    fi
    git apply "$p"
    echo "[APPLIED] $name"
done
