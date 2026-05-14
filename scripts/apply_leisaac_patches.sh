#!/usr/bin/env bash
# Apply our local patches to the LeIsaac submodule.
# Idempotent: if a patch is already applied, skip it.
# Run after `git submodule update --init --recursive`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEISAAC_DIR="$ROOT_DIR/LeIsaac"
PATCH_DIR="$ROOT_DIR/patches/leisaac"

if [ ! -d "$LEISAAC_DIR/.git" ] && [ ! -f "$LEISAAC_DIR/.git" ]; then
    echo "[ERROR] LeIsaac submodule not initialized at $LEISAAC_DIR" >&2
    echo "        Run: git submodule update --init --recursive" >&2
    exit 1
fi

shopt -s nullglob
patches=( "$PATCH_DIR"/*.patch )
if [ ${#patches[@]} -eq 0 ]; then
    echo "[INFO] no patches under $PATCH_DIR"
    exit 0
fi

cd "$LEISAAC_DIR"
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
