#!/usr/bin/env bash
#
# Initialize the GR00T inference server box (Ubuntu + NVIDIA GPU).
# Installs only what server/start_server.sh and server/verify_server.sh
# need to run the GR00T model service. Isaac Sim / LeIsaac live on a
# different host and are NOT touched here.
#
# Idempotent: every step checks before installing.
#
# Usage:
#   bash server/init_server.sh            # full setup
#   bash server/init_server.sh --no-clone # skip cloning Isaac-GR00T
#   bash server/init_server.sh --no-sync  # skip `uv sync` warm-up

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
have()      { command -v "$1" >/dev/null 2>&1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(cd "$ROOT_DIR/.." && pwd)"
GR00T_DIR="${GR00T_DIR:-$ROOT_DIR/dependencies/Isaac-GR00T}"
GR00T_REPO="${GR00T_REPO:-https://github.com/NVIDIA/Isaac-GR00T.git}"

DO_CLONE=1
DO_SYNC=1
for arg in "$@"; do
    case "$arg" in
        --no-clone) DO_CLONE=0 ;;
        --no-sync)  DO_SYNC=0 ;;
        -h|--help)
            sed -n '2,15p' "$0"; exit 0 ;;
        *) log_error "unknown arg: $arg"; exit 1 ;;
    esac
done

# 1. Sanity: Ubuntu + NVIDIA driver
check_os() {
    if [ ! -f /etc/os-release ] || ! grep -qi ubuntu /etc/os-release; then
        log_warn "non-Ubuntu host detected; apt steps may fail"
    else
        log_info "OS: $(. /etc/os-release; echo "$PRETTY_NAME")"
    fi
}

check_gpu() {
    if have nvidia-smi; then
        local line
        line="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"
        log_info "GPU: $line"
    else
        log_warn "nvidia-smi not found. GR00T inference needs a CUDA GPU + driver."
    fi
}

# 2. apt packages: only install what's missing, single apt-get update
ensure_apt_pkgs() {
    local pkgs=(git git-lfs curl ca-certificates build-essential python3 python3-pip python3-venv)
    local missing=()
    for p in "${pkgs[@]}"; do
        dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
    done
    if [ ${#missing[@]} -eq 0 ]; then
        log_info "apt deps already present: ${pkgs[*]}"
        return
    fi
    log_info "installing apt deps: ${missing[*]}"
    sudo apt-get update -y
    sudo apt-get install -y "${missing[@]}"
    have git-lfs && git lfs install --skip-repo >/dev/null 2>&1 || true
}

# 3. uv (used by start_server.sh: `uv run --extra=gpu`)
ensure_uv() {
    if have uv; then
        log_info "uv already installed: $(uv --version)"
        return
    fi
    log_info "installing uv (astral.sh installer)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin or ~/.cargo/bin depending on version
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! have uv; then
        log_error "uv install failed; add ~/.local/bin to PATH and re-source your shell rc"
        exit 1
    fi
    log_info "uv installed: $(uv --version)"
}

# 4. Python tools the scripts in this repo rely on (system pip).
#    huggingface_hub[cli] -> `hf download`
#    pyzmq               -> verify_servers.sh ZMQ_PING=1
ensure_pip_tools() {
    local py
    if have python3; then py=python3; elif have python; then py=python; else
        log_error "python3 not found"; exit 1
    fi
    local need_hf=0 need_zmq=0
    "$py" -c "import huggingface_hub" 2>/dev/null || need_hf=1
    "$py" -c "import zmq"             2>/dev/null || need_zmq=1
    if [ "$need_hf" -eq 0 ] && [ "$need_zmq" -eq 0 ]; then
        log_info "pip tools already present: huggingface_hub, pyzmq"
        return
    fi
    local pkgs=()
    [ "$need_hf"  -eq 1 ] && pkgs+=("huggingface_hub[cli]" "hf_transfer")
    [ "$need_zmq" -eq 1 ] && pkgs+=("pyzmq")
    log_info "installing pip tools: ${pkgs[*]}"
    "$py" -m pip install --user --upgrade "${pkgs[@]}"
}

# 5. Isaac-GR00T submodule at $ROOT_DIR/dependencies/Isaac-GR00T
ensure_gr00t_repo() {
    if [ "$DO_CLONE" -eq 0 ]; then
        log_info "skipping Isaac-GR00T clone (--no-clone)"
        return
    fi
    if [ -d "$GR00T_DIR/.git" ]; then
        log_info "Isaac-GR00T already at $GR00T_DIR"
        return
    fi
    if [ -e "$GR00T_DIR" ]; then
        log_error "$GR00T_DIR exists but is not a git repo; refusing to overwrite"
        exit 1
    fi
    log_info "cloning $GR00T_REPO -> $GR00T_DIR"
    git clone --depth 1 "$GR00T_REPO" "$GR00T_DIR"
}

# 6. Pre-warm `uv sync` so first `start_server.sh` doesn't spend 10+ min
#    resolving and downloading torch/cu12 wheels under nohup.
warm_uv_sync() {
    if [ "$DO_SYNC" -eq 0 ]; then
        log_info "skipping uv sync warm-up (--no-sync)"
        return
    fi
    if [ ! -d "$GR00T_DIR" ]; then
        log_warn "no GR00T dir, skipping uv sync warm-up"
        return
    fi
    log_info "warming uv env (this may take a while on first run)"
    ( cd "$GR00T_DIR" && uv sync --extra=gpu )
}

# 7. Friendly summary of next steps
print_next_steps() {
    cat <<EOF

${GREEN}[DONE]${NC} server init complete.

Next steps on this host:
  1) (if not yet) download model weights to default HF cache:
       hf auth login                  # or: export HF_TOKEN=hf_xxx
       HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/GR00T-N1.6-3B

  2) start the GR00T server (binds 0.0.0.0:5555 by default):
       bash server/start_server.sh --gr00t-only

  3) check it's up:
       bash server/status_server.sh
       tail -f logs/gr00t_server.log

  4) open the firewall so the sim host can reach this server:
       sudo ufw allow 5555/tcp        # if ufw is enabled

From the sim host:
       bash server/verify_server.sh <this-host-ip>
EOF
}

main() {
    check_os
    check_gpu
    ensure_apt_pkgs
    ensure_uv
    ensure_pip_tools
    ensure_gr00t_repo
    warm_uv_sync
    print_next_steps
}

main "$@"
