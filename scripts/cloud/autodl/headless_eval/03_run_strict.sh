#!/usr/bin/env bash
# Run STRICT 20-round eval on the lerobot baselines (云端专注小模型).
# Defaults ONLY=smolvla-self,smolvla-other,act-self,act-other,xvla-self
# (5 baselines × ~50min = ~4h sequential)
#
# Results: /root/autodl-tmp/isaaclab-experience/results/benchmark/<slug>.metrics.json
# Run after install + smoke pass.
#
# Use scripts/cloud/autodl/headless_eval/04_scp_back.sh to pull results to local.

set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/isaaclab-experience}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ONLY="${ONLY:-smolvla-self,smolvla-other,act-self,act-other,xvla-self}"
STRICT_ROUNDS="${STRICT_ROUNDS:-20}"

export KIT_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=Y
export ACCEPT_EULA=Y
export DISPLAY=""
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[cloud-strict] ONLY=$ONLY STRICT_ROUNDS=$STRICT_ROUNDS"
echo "[cloud-strict] env CONDA_ENV=$CONDA_ENV"
echo "[cloud-strict] HF_HOME=$HF_HOME"

cd "$REMOTE_ROOT"
ONLY="$ONLY" STRICT_ROUNDS="$STRICT_ROUNDS" CONDA_ENV="$CONDA_ENV" \
    bash scripts/benchmark/run_all_strict.sh

echo "[cloud-strict] DONE. results in $REMOTE_ROOT/results/benchmark/"
echo "[cloud-strict] next: bash 04_scp_back.sh <local_host>"
