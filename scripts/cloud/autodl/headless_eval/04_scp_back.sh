#!/usr/bin/env bash
# rsync cloud results/benchmark/*.metrics.json + *.distribution.* back to local.
# Run on LOCAL machine. Then `python aggregate_strict_leaderboard.py` to merge.
#
# Usage:
#   bash 04_scp_back.sh <host> <port> <pass>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

HOST="${1:?host required}"
PORT="${2:?port required}"
PASS="${3:?pass required}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/isaaclab-experience}"

mkdir -p "$ROOT_DIR/results/benchmark"
echo "[scp-back] rsyncing results from $HOST:$PORT"

SSHPASS="$PASS" sshpass -e rsync -avz --partial --progress \
    -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    --include='*.metrics.json' \
    --include='*.distribution.md' \
    --include='*.distribution.svg' \
    --include='*.summary.txt' \
    --include='*.gpu.csv' \
    --exclude='*' \
    "root@$HOST:$REMOTE_ROOT/results/benchmark/" \
    "$ROOT_DIR/results/benchmark/"

echo "[scp-back] done."
echo "[scp-back] rebuild leaderboard:"
echo "    python3 $ROOT_DIR/scripts/benchmark/aggregate_strict_leaderboard.py \\"
echo "        --results_dir $ROOT_DIR/results/benchmark \\"
echo "        --out $ROOT_DIR/scripts/benchmark/STRICT_LEADERBOARD.md"
