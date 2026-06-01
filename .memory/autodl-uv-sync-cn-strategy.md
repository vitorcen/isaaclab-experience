---
name: autodl-uv-sync-cn-strategy
description:  AutoDL 上 uv sync Isaac-GR00T 的最终成功配方 — 6 个 patch + no_proxy 关键
metadata: 
  node_type: memory
  type: reference
  originSessionId: 39899a1b-ef9c-4403-93fc-2e3491bfb440
---

# AutoDL × CN × uv sync Isaac-GR00T 成功配方

实战 2026-05-22 session 累计 3 小时踩 7 个坑后总结出的最快路径。下次照抄即可。

## 总配方（关键 6 点）

### 1. `~/.config/uv/uv.toml`：aliyun 设为 default PyPI index
```toml
[[index]]
name = "aliyun-pypi"
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
```
**为什么**：`files.pythonhosted.org` 从 CN 都 TLS handshake EOF；aliyun PyPI mirror 是 PEP 503 兼容 simple index。

### 2. Isaac-GR00T `pyproject.toml` [tool.uv.sources]：torch/torchvision 改成 aliyun 直接 URL
```toml
[tool.uv.sources]
torch = [
    { url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.10'" },
]
torchvision = [
    { url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.10'" },
]
```
**为什么**：`download.pytorch.org`  TLS EOF；aliyun mirror 是 Apache 文件列表（非 PEP 503）所以不能当 `[[tool.uv.index]]` 用，但可以做 `{ url = ... }` direct source。

### 3. Isaac-GR00T `pyproject.toml` [tool.uv]：限定 x86_64 only
```toml
[tool.uv]
required-environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
```
**为什么**：upstream 默认要求 x86_64 AND aarch64，但 aliyun 不全 mirror aarch64 wheel（triton-3.3.1 aarch64 缺失 → uv 解析失败）。

### 4. Isaac-GR00T `pyproject.toml` dependencies：**砍掉 tensorrt-cu12**
```toml
# REMOVED FOR CN: "tensorrt-cu12>=10.15.1.29; platform_machine == 'x86_64'",
```
**为什么**：tensorrt-cu12-libs 走 wheel_stub，pypi.nvidia.com 在 AutoDL 上 SHA mismatch 死循环。**而 gr00t/experiment/ + gr00t/configs/ + gr00t/data/ + gr00t/model/ 全部 grep 后 0 个 `import tensorrt`**，只有 scripts/deployment/ 用 tensorrt build engine 做 inference 加速。**训练不需要**。
教训：失败 build > 10 min 立刻 grep 训练代码确认依赖必要性；不必要就砍掉。

### 5. `no_proxy` 让 aliyun/nvidia 直连不走 AutoDL proxy
```bash
source /etc/network_turbo
export no_proxy="localhost,127.0.0.1,mirrors.aliyun.com,pypi.nvidia.com,modelscope.com,aliyuncs.com,tencentyun.com"
```
**为什么**：AutoDL proxy `10.37.1.23:12798` 把大文件下载限速到 ~338 KB/s；aliyun 直连同一个 URL 是 **15-16 MB/s**（45× 提升）。github 仍要走 proxy（直连 11 KB/s 几乎封死）。

### 6. flash-attn：用 GitHub release URL（依赖 AutoDL proxy 覆盖 github.com）
```toml
flash-attn = [
    { url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.10'" },
]
```
**为什么**：`cxx11abiFALSE` 变体才与 PyTorch 官方 cu128 wheel 的 ABI 匹配；本机 sdist build 出来的 wheel 是 `cxx11abi=TRUE` 默认，会触发 `undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs` ImportError。

## 执行顺序

```bash
# 1. 一次性写 uv.toml + 推 patched pyproject
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml <<EOF
[[index]]
name = "aliyun-pypi"
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
EOF
# scp pyproject 到位（含 patches 2/3/4/6）

# 2. uv lock 重新解析（清掉 stale tensorrt entries）
cd dependencies/Isaac-GR00T
uv lock

# 3. uv sync 装包（带 no_proxy 优化）
export PATH=/root/.local/bin:/root/miniconda3/bin:$PATH
source /etc/network_turbo >/dev/null 2>&1
export no_proxy="localhost,127.0.0.1,mirrors.aliyun.com,pypi.nvidia.com,aliyuncs.com,tencentyun.com"
uv sync
```

## 预期速度
- uv lock：~13s
- uv sync：~3-5 min（aliyun 15 MB/s torch 991MB ≈ 70s，flash-attn 395MB via proxy 慢些但能完成）

## 已验证 import
```python
import torch; print(torch.__version__)  # → 2.7.1+cu128
import flash_attn; print(flash_attn.__version__)  # → 2.7.4.post1
import transformers; print(transformers.__version__)  # → 4.57.3
```

## 关联
- [[cn-pypi-mirror-aliyun]] — aliyun mirror 基础
- [[feedback-autodl-cost-discipline]] — 这套配方源于成本反思（之前死磕 tensorrt 浪费 ¥18）
- `LeIsaac/scripts/cloud/autodl/uv_sync_v3.sh` 是同款配置脚本
- `LeIsaac/docs/training/autodl_cloud_finetune_playbook.html` §4 + §7 故障 playbook
