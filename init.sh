#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 IsaacSim-OpenVLA
# SPDX-License-Identifier: Apache-2.0
#
# IsaacSim 运行环境初始化脚本
# 此脚本用于初始化 IsaacSim 开发环境，包括依赖检查、安装和构建

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查 Git 是否安装
check_git() {
    log_info "检查 Git..."
    if ! command_exists git; then
        log_error "Git 未安装，请先安装 Git"
        log_info "运行: sudo apt-get install git"
        exit 1
    fi
    log_info "Git 已安装: $(git --version)"
}

# 检查并安装 Git LFS
check_git_lfs() {
    log_info "检查 Git LFS..."
    if ! command_exists git-lfs; then
        log_warn "Git LFS 未安装，正在安装..."
        sudo apt-get update
        sudo apt-get install -y git-lfs
    fi
    log_info "Git LFS 已安装: $(git-lfs --version)"
}

# 检查并安装 build-essential
check_build_tools() {
    log_info "检查构建工具..."
    if ! command_exists make; then
        log_warn "build-essential 未安装，正在安装..."
        sudo apt-get update
        sudo apt-get install -y build-essential
    fi
    log_info "构建工具已安装"
}

# 检查 GCC/G++ 版本
check_gcc_version() {
    log_info "检查 GCC/G++ 版本..."

    if ! command_exists gcc || ! command_exists g++; then
        log_error "GCC/G++ 未安装"
        install_gcc_11
        return
    fi

    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    GPP_VERSION=$(g++ -dumpversion | cut -d. -f1)

    log_info "当前 GCC 版本: $(gcc --version | head -n1)"
    log_info "当前 G++ 版本: $(g++ --version | head -n1)"

    if [ "$GCC_VERSION" != "11" ] || [ "$GPP_VERSION" != "11" ]; then
        log_warn "IsaacSim 需要 GCC/G++ 11，当前版本不匹配"
        log_warn "正在安装 GCC/G++ 11..."
        install_gcc_11
    else
        log_info "GCC/G++ 版本正确 (v11)"
    fi
}

# 安装 GCC/G++ 11
install_gcc_11() {
    log_info "安装 GCC/G++ 11..."
    sudo apt-get update
    sudo apt-get install -y gcc-11 g++-11

    # 设置为默认版本
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200

    log_info "GCC/G++ 11 安装完成"
    log_info "新版本: $(gcc --version | head -n1)"
}

# 检查 NVIDIA 驱动
check_nvidia_driver() {
    log_info "检查 NVIDIA 驱动..."
    if command_exists nvidia-smi; then
        log_info "NVIDIA 驱动已安装:"
        nvidia-smi --query-gpu=gpu_name,driver_version --format=csv,noheader
    else
        log_warn "未检测到 NVIDIA 驱动，请确保已安装正确的 GPU 驱动"
        log_warn "参考: https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html"
    fi
}

# 初始化 Git Submodule
init_submodule() {
    log_info "初始化 Git Submodule..."

    git submodule sync --recursive

    # 只初始化顶层 submodule，避免递归拉取 LeIsaac/dependencies/IsaacLab 造成重复克隆
    git submodule update --init
    log_info "Submodule 初始化完成"
}

# 将 LeIsaac/dependencies/IsaacLab 软链到仓库顶层 IsaacLab，避免重复仓库副本
link_leisaac_isaaclab() {
    if [ ! -d "LeIsaac" ] || [ ! -d "IsaacLab" ]; then
        log_warn "LeIsaac 或 IsaacLab 不存在，跳过软链接配置"
        return
    fi

    mkdir -p LeIsaac/dependencies
    local dep_path="LeIsaac/dependencies/IsaacLab"
    local link_target="../../IsaacLab"

    if [ -L "$dep_path" ]; then
        log_info "LeIsaac 依赖 IsaacLab 已是软链接"
        return
    fi

    if [ -d "$dep_path" ]; then
        if [ -z "$(ls -A "$dep_path")" ]; then
            rmdir "$dep_path"
            ln -s "$link_target" "$dep_path"
            log_info "已创建软链接: $dep_path -> $link_target"
        else
            log_warn "$dep_path 已存在且非空，跳过自动软链接（避免覆盖现有内容）"
        fi
        return
    fi

    ln -s "$link_target" "$dep_path"
    log_info "已创建软链接: $dep_path -> $link_target"
}

# 配置 Git LFS
setup_git_lfs() {
    log_info "配置 Git LFS..."
    cd IsaacSim

    # 初始化 Git LFS
    git lfs install

    # 拉取 LFS 文件
    log_info "拉取 Git LFS 文件（这可能需要一些时间）..."
    git lfs pull

    cd ..
    log_info "Git LFS 配置完成"
}

# 检查网络代理设置
check_proxy() {
    if [ -n "${http_proxy:-}" ] || [ -n "${https_proxy:-}" ]; then
        log_info "检测到代理设置:"
        [ -n "${http_proxy:-}" ] && log_info "  HTTP_PROXY: $http_proxy"
        [ -n "${https_proxy:-}" ] && log_info "  HTTPS_PROXY: $https_proxy"
    else
        log_info "未检测到代理设置"
        log_info "如果在公司防火墙后或需要代理，请设置:"
        log_info "  export http_proxy=\"http://your-proxy:port\""
        log_info "  export https_proxy=\"http://your-proxy:port\""
    fi
}

# 可选：安装 Podman（用于容器化开发）
install_podman_optional() {
    if [ "${INSTALL_PODMAN:-no}" = "yes" ]; then
        log_info "安装 Podman..."
        if ! command_exists podman; then
            log_info "正在安装 Podman..."
            sudo apt-get update
            sudo apt-get install -y podman

            # 检查是否有 nvidia-container-toolkit
            if command_exists nvidia-ctk; then
                log_info "配置 Podman 使用 NVIDIA 容器工具..."
                # 生成 CDI 配置
                sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
            fi
        else
            log_info "Podman 已安装: $(podman --version)"
        fi
    fi
}

# 构建 IsaacSim
build_isaacsim() {
    if [ "${BUILD_NOW:-no}" = "yes" ]; then
        log_info "开始构建 IsaacSim..."
        log_warn "首次构建可能需要很长时间（取决于网络速度和硬件配置）"

        cd IsaacSim
        ./build.sh
        cd ..

        log_info "IsaacSim 构建完成！"
    else
        log_info "跳过构建步骤"
        log_info "如需构建，请运行: cd IsaacSim && ./build.sh"
    fi
}

# 显示下一步操作
show_next_steps() {
    echo ""
    log_info "========================================="
    log_info "初始化完成！"
    log_info "========================================="
    echo ""
    log_info "下一步操作："
    echo ""
    echo "  1. 下载 LeIsaac 资产（新机器必做）:"
    echo "     ./leisaac.sh"
    echo ""
    echo "  2. 构建 IsaacSim:"
    echo "     cd IsaacSim"
    echo "     ./build.sh"
    echo ""
    echo "  3. 运行 IsaacSim:"
    echo "     cd IsaacSim/_build/linux-x86_64/release"
    echo "     ./isaac-sim.sh"
    echo ""
    echo "  4. 查看文档:"
    echo "     https://docs.isaacsim.omniverse.nvidia.com/latest/index.html"
    echo ""
    log_warn "注意：首次运行可能需要几分钟来加载扩展和缓存着色器"
    echo ""
}

# 主函数
main() {
    log_info "========================================="
    log_info "IsaacSim 运行环境初始化"
    log_info "========================================="
    echo ""

    # 检查系统依赖
    check_git
    check_git_lfs
    check_build_tools
    check_gcc_version
    check_nvidia_driver
    check_proxy

    echo ""
    log_info "========================================="
    log_info "配置 IsaacSim"
    log_info "========================================="
    echo ""

    # 初始化 submodule
    init_submodule
    link_leisaac_isaaclab

    # 配置 Git LFS
    setup_git_lfs

    # 可选安装
    install_podman_optional

    # 可选构建
    build_isaacsim

    # 显示下一步
    show_next_steps
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_NOW=yes
            shift
            ;;
        --install-podman)
            INSTALL_PODMAN=yes
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --build            初始化后立即构建 IsaacSim"
            echo "  --install-podman   安装 Podman（需要 sudo 权限）"
            echo "  --help             显示此帮助信息"
            echo ""
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            echo "运行 '$0 --help' 查看帮助"
            exit 1
            ;;
    esac
done

# 运行主函数
main
