#!/usr/bin/env bash
# AutoDL headless eval install — 快速 CN 配方。
# 复用 memory: autodl-uv-sync-cn-strategy + cn-pypi-mirror-aliyun + hf-upload-tricks
#
# 速度提升点（vs 默认）：
#   1. uv 替代 pip → install 速度 3-5×
#   2. aliyun pypi 默认 → 跳出 pypi.org TLS handshake EOF
#   3. torch/torchvision 直接 URL (aliyun pytorch-wheels) → 跳过 download.pytorch.org TLS EOF
#   4. no_proxy bypass academic_turbo for aliyun + pypi.nvidia.com → 不被代理 choke 到 1MB/s
#   5. HF_HUB_ENABLE_HF_TRANSFER=1 → ckpt 下载 5-10×
#   6. x86_64 only required-env in pyproject → uv 不去解析 aarch64 wheels
#
# Idempotent: detect existing install, skip what's already there.
# ~30 min total (was ~1.5h with naive pip).

set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/isaaclab-experience}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
PY_VER="${PY_VER:-3.11}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"

echo "[install] REMOTE_ROOT=$REMOTE_ROOT CONDA_ENV=$CONDA_ENV PY=$PY_VER"
START_T=$(date +%s)

# === AutoDL academic proxy for github/huggingface, BUT NOT for pypi/aliyun ===
source /etc/network_turbo 2>/dev/null || echo "[install] no /etc/network_turbo (non-AutoDL?)"
export no_proxy="localhost,127.0.0.1,mirrors.aliyun.com,pypi.nvidia.com,aliyuncs.com,tencentyun.com"

# === HF transfer turbo + HF cache → 大盘 ===
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# === 1. miniconda (AutoDL images usually pre-install at /root/miniconda3) ===
# Don't trust `command -v conda` in non-interactive ssh — PATH may not include it.
# Check binary directly.
if [ -x /root/miniconda3/bin/conda ]; then
    echo "[install] (1) /root/miniconda3 already exists, skip"
elif [ -x /opt/conda/bin/conda ]; then
    echo "[install] (1) /opt/conda exists, symlink to /root/miniconda3"
    ln -sfn /opt/conda /root/miniconda3
else
    echo "[install] (1) miniconda3 install"
    cd /root
    # Tsinghua mirror more reliable than aliyun for miniconda
    curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    bash miniconda.sh -b -p /root/miniconda3
    rm miniconda.sh
fi
export PATH="/root/miniconda3/bin:$PATH"

# === 2. uv (fast pip replacement) ===
if ! command -v uv >/dev/null 2>&1; then
    echo "[install] (2) uv"
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi

# === 3. conda env ===
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "[install] (3) create env $CONDA_ENV"
    conda create -y -n "$CONDA_ENV" "python=$PY_VER" pip
fi
PY="/root/miniconda3/envs/$CONDA_ENV/bin/python"
PIP="$PY -m pip"

# === 4. pip mirror config (aliyun default + nvidia extra) ===
mkdir -p /root/.config/uv /root/.pip
cat > /root/.config/uv/uv.toml <<'EOF'
[[index]]
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true

[[index]]
url = "https://pypi.nvidia.com"
EOF
cat > /root/.pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
extra-index-url = https://pypi.nvidia.com
EOF

# === 5. torch 2.7.0+cu128 + torchvision 0.22.0 (matched pair on aliyun) ===
# aliyun cu128 has torch 2.7.0/2.7.1 + only torchvision 0.22.0 (which strict-needs torch 2.7.0)
# So pair them as 2.7.0 + 0.22.0 to satisfy uv resolver.
echo "[install] (5) torch 2.7.0+cu128 + torchvision 0.22.0+cu128 via aliyun direct URL"
uv pip install --python "$PY" --no-cache \
    https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl \
    https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl

# === 6. isaacsim 5.1 (~15GB, only HF/nvidia have it) ===
# Use pip (not uv) — uv can't supply EULA stdin "Yes" answer.
# OMNI_KIT_ACCEPT_EULA=Y + ACCEPT_EULA=Y silence prompts.
# isaacsim[all] pulls heavy ~15GB extras incl _isaac_sim/ + kit binaries.
# Check for _isaac_sim/python.sh (heavy install marker) not __init__.py (stub).
# Check via isaacsim-rl pkg presence (isaaclab.sh's fallback also keys on this).
# heavy Kit binaries are downloaded only on first sim launch, not at pip install.
if ! $PY -m pip show isaacsim-rl >/dev/null 2>&1; then
    echo "[install] (6) isaacsim[all]==5.1.0 + isaacsim-rl (heavy extras, ~15GB)"
    export OMNI_KIT_ACCEPT_EULA=Y
    export ACCEPT_EULA=Y
    yes Yes | $PIP install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
        "isaacsim[all]==5.1.0"
else
    echo "[install] (6) isaacsim+isaacsim-rl already installed, skip"
fi
# Persist EULA so future imports don't prompt
mkdir -p /root/.nvidia-omniverse/config
[ -f /root/.nvidia-omniverse/config/omniverse.toml ] || cat > /root/.nvidia-omniverse/config/omniverse.toml <<'EULA_EOF'
[privacy]
performance = true
personalization = true
usage = true
crashreporting = true
[eulaAccepted]
NVIDIA_OMNIVERSE_LICENSE_AGREEMENT = true
EULA_EOF

# === 7. IsaacLab (clone if missing + editable install) ===
ISAACLAB_DIR="${ISAACLAB_DIR:-$REMOTE_ROOT/IsaacLab}"
if [ ! -d "$ISAACLAB_DIR/source" ]; then
    echo "[install] (7) git clone IsaacLab"
    git clone --depth=1 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
echo "[install] (7) pip install IsaacLab editable"
cd "$ISAACLAB_DIR"
# isaaclab.sh checks $CONDA_PREFIX to pick the right python — set it explicitly
# so it uses our env, not the missing kit python at _isaac_sim/python.sh
export CONDA_PREFIX="/root/miniconda3/envs/$CONDA_ENV"
# isaaclab.sh handles all submodule pip install -e, uses pip not uv (some submodules need it)
./isaaclab.sh --install 2>&1 | tail -20

# === 8. LeIsaac editable ===
echo "[install] (8) LeIsaac editable"
cd "$REMOTE_ROOT/LeIsaac/source/leisaac"
uv pip install --python "$PY" -e .

# === 9. lerobot 0.4.0 (frozen for ACT/SmolVLA/DP) ===
if ! $PY -c "import lerobot; assert lerobot.__version__.startswith('0.4')" 2>/dev/null; then
    echo "[install] (9) lerobot==0.4.0"
    uv pip install --python "$PY" --no-cache "lerobot==0.4.0"
fi

# === 10. HF cache + transfer env in .bashrc ===
grep -q "HF_HOME=$HF_HOME" /root/.bashrc 2>/dev/null || {
    echo "export HF_HOME=$HF_HOME" >> /root/.bashrc
    echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> /root/.bashrc
    echo "export PATH=/root/.local/bin:\$PATH" >> /root/.bashrc
}

# === 11. validate ===
END_T=$(date +%s)
echo ""
echo "[install] === validation ==="
$PY -c "
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())
import isaacsim; print('isaacsim ok')
import isaaclab; print('isaaclab ok')
import leisaac; print('leisaac ok')
import lerobot; print('lerobot', lerobot.__version__)
import hf_transfer; print('hf_transfer ok')
"

echo ""
echo "[install] === DONE in $(( (END_T - START_T) / 60 ))min ==="
echo "[install] next: bash 02_smoke.sh"
