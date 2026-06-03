#!/usr/bin/env bash
# Apply our local patches to the GR00T-WholeBodyControl (GEAR-SONIC) submodule.
# Idempotent: an already-applied patch is skipped.
# Run after `git submodule update --init dependencies/GR00T-WholeBodyControl`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$ROOT_DIR/dependencies/GR00T-WholeBodyControl"
PATCH_DIR="$ROOT_DIR/patches/gear-sonic"

if [ ! -e "$WBC_DIR/.git" ]; then
    echo "[ERROR] GR00T-WholeBodyControl submodule not initialized at $WBC_DIR" >&2
    echo "        Run: git submodule update --init dependencies/GR00T-WholeBodyControl" >&2
    exit 1
fi

shopt -s nullglob
patches=( "$PATCH_DIR"/*.patch )
if [ ${#patches[@]} -eq 0 ]; then
    echo "[INFO] no patches under $PATCH_DIR"
    exit 0
fi

cd "$WBC_DIR"
for p in "${patches[@]}"; do
    name="$(basename "$p")"
    # `git apply --check --reverse` succeeds iff the patch is already applied.
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
