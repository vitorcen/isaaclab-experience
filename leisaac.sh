#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEISAAC_DIR="$ROOT_DIR/LeIsaac"
ASSETS_DIR="$LEISAAC_DIR/assets"
CACHE_DIR="${LEISAAC_ENV_CACHE:-$ROOT_DIR/.cache/leisaac_env}"
HF_REPO_URL="https://huggingface.co/LightwheelAI/leisaac_env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log_error "缺少命令: $1"
        exit 1
    fi
}

ensure_submodule() {
    if [ ! -d "$LEISAAC_DIR/.git" ]; then
        log_warn "未发现 LeIsaac 子模块，正在初始化..."
        git -C "$ROOT_DIR" submodule update --init --recursive LeIsaac
    fi
}

sync_hf_repo() {
    if [ ! -d "$CACHE_DIR/.git" ]; then
        log_info "首次下载 leisaac_env（体积较大，请耐心等待）..."
        mkdir -p "$(dirname "$CACHE_DIR")"
        git clone --depth 1 "$HF_REPO_URL" "$CACHE_DIR"
    else
        log_info "更新本地缓存仓库..."
        git -C "$CACHE_DIR" pull --ff-only
    fi
}

copy_assets() {
    mkdir -p "$ASSETS_DIR"
    log_info "同步资产到 LeIsaac/assets ..."
    rsync -a "$CACHE_DIR/assets/" "$ASSETS_DIR/"
}

verify_assets() {
    local required_files=(
        "$ASSETS_DIR/robots/so101_follower.usd"
        "$ASSETS_DIR/scenes/kitchen_with_orange/scene.usd"
    )

    for f in "${required_files[@]}"; do
        if [ ! -f "$f" ]; then
            log_error "缺少关键资产: $f"
            exit 1
        fi
    done

    log_info "资产校验通过。"
}

main() {
    log_info "========================================="
    log_info "LeIsaac 资产下载脚本"
    log_info "========================================="

    require_cmd git
    require_cmd rsync

    ensure_submodule
    sync_hf_repo
    copy_assets
    verify_assets

    echo ""
    log_info "完成。你现在可以运行 LeIsaac.ipynb 里的推理单元。"
}

main "$@"
