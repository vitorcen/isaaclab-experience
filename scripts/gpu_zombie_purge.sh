#!/usr/bin/env bash
# Aggressively purge CUDA zombie memory on the primary GPU (RTX 4090) without
# rebooting.  Walks a ladder of increasingly invasive moves:
#
#   1. fuser -k on /dev/nvidia* (kill any proc holding nvidia char dev)
#   2. reload nvidia_uvm kernel module (clears UM contexts)
#   3. nvidia-smi --gpu-reset (fails on primary GPU but worth trying)
#
# After each step, check `nvidia-smi memory.used`. Stop when GPU < $THRESHOLD MiB.
#
# Usage:
#   bash scripts/gpu_zombie_purge.sh           # default threshold=3000 MiB
#   THRESHOLD=2000 bash scripts/gpu_zombie_purge.sh
#
# Requires sudo (will prompt once).

set -uo pipefail

THRESHOLD="${THRESHOLD:-3000}"

gpu_used() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

check() {
    local used=$(gpu_used)
    echo "  → GPU used: ${used} MiB"
    if (( used < THRESHOLD )); then
        echo "✅ GPU clean (< ${THRESHOLD} MiB), exiting"
        exit 0
    fi
}

echo "[purge] starting, threshold=${THRESHOLD} MiB"
echo "[purge] initial state:"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null
check

# Step 1: any user-space proc holding nvidia* device files
echo
echo "[purge] step 1: fuser -k /dev/nvidia* (kill procs holding GPU devices)"
sudo fuser -v /dev/nvidia* 2>&1 || true
sudo fuser -k /dev/nvidia* 2>&1 || true
sleep 3
check

# Step 2: reload nvidia_uvm kernel module
echo
echo "[purge] step 2: reload nvidia_uvm module"
sudo rmmod nvidia_uvm 2>&1 && sleep 2 && sudo modprobe nvidia_uvm && echo "  → module reloaded ok" || echo "  → rmmod failed (module busy or missing)"
sleep 2
check

# Step 3: try gpu-reset (will fail on primary GPU but cheap to attempt)
echo
echo "[purge] step 3: nvidia-smi --gpu-reset"
sudo nvidia-smi --gpu-reset 2>&1 || echo "  → gpu-reset failed (primary GPU)"
sleep 2
check

echo
echo "❌ GPU still busy after all 3 steps. Reboot required."
echo "Current state:"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null
exit 1
