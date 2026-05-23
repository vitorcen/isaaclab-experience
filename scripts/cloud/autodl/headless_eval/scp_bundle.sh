#!/usr/bin/env bash
# scp 本机 isaaclab-experience repo 到 AutoDL 云端，准备 headless eval。
#
# Skip 大目录:
#   - dependencies/Isaac-GR00T (10GB submodule, 云端用 pip install lerobot 不需要)
#   - dependencies/IsaacLab    (云端从 pip install isaacsim 装)
#   - LeIsaac/datasets/raw     (HF 自动 cache)
#   - LeIsaac/outputs          (训练产物)
#   - LeIsaac/wandb            (训练日志)
#   - **/__pycache__ + *.pyc
#   - .git (大且无关 eval)
#
# Usage:
#   bash scp_bundle.sh <host> <port> <pass>
# Example:
#   bash scp_bundle.sh connect.cqa1.seetacloud.com 42863 UoRCWwwyzWog

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

HOST="${1:?host required (e.g. connect.cqa1.seetacloud.com)}"
PORT="${2:?ssh port required}"
PASS="${3:?ssh password required}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/isaaclab-experience}"

EXCLUDES=(
    --exclude='.git'
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='dependencies/Isaac-GR00T'
    --exclude='dependencies/IsaacLab'
    --exclude='LeIsaac/datasets/raw'
    --exclude='LeIsaac/outputs'
    --exclude='LeIsaac/wandb'
    --exclude='logs'
    --exclude='results/benchmark/snapshots'
    --exclude='*.safetensors'
    --exclude='*.pt'
    --exclude='*.npz'
    --exclude='*.bin'
)

echo "[scp-bundle] root: $ROOT_DIR"
echo "[scp-bundle] target: $REMOTE_ROOT on $HOST:$PORT"
echo "[scp-bundle] estimated size:"
du -sh --exclude='__pycache__' --exclude='.git' --exclude='dependencies/Isaac-GR00T' \
       --exclude='dependencies/IsaacLab' --exclude='LeIsaac/outputs' \
       --exclude='LeIsaac/datasets/raw' --exclude='LeIsaac/wandb' \
       "$ROOT_DIR" 2>/dev/null | head -1

# mkdir + rsync via ssh; rsync is resumable + diff-only
SSHPASS="$PASS" sshpass -e ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$HOST" \
    "mkdir -p $REMOTE_ROOT"

echo "[scp-bundle] rsyncing ..."
SSHPASS="$PASS" sshpass -e rsync -avz --partial --progress \
    -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    "${EXCLUDES[@]}" \
    "$ROOT_DIR/" "root@$HOST:$REMOTE_ROOT/"

echo "[scp-bundle] done. next: ssh in + bash $REMOTE_ROOT/scripts/cloud/autodl/headless_eval/01_install.sh"
