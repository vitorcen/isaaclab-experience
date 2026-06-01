---
name: cn-pypi-mirror-aliyun
description:  uv sync / pip install 必备：PyTorch CDN 不通时切 aliyun mirror，14 MB/s 稳定，避免 TLS handshake EOF
metadata: 
  node_type: memory
  type: reference
  originSessionId: 39899a1b-ef9c-4403-93fc-2e3491bfb440
---

# CN PyPI / PyTorch mirror — Aliyun 优先

## Why
`download.pytorch.org` / `download-r2.pytorch.org` 从本机（无 VPN）**TLS handshake EOF**，uv 默认配置直接挂在 `torch==2.7.1+cu128` 下载。AutoDL 学术代理也不一定覆盖 PyTorch CDN（覆盖 huggingface.co / github.com / githubassets.com）。Aliyun `pytorch-wheels` mirror 实测 14 MB/s 稳定且与官方完全同步。

## How to apply

### A. Isaac-GR00T pyproject.toml 改 [tool.uv.sources] 为 direct URL（推荐）

⚠️ **不能直接 sed 替换 index URL** — aliyun 的 `/pytorch-wheels/cu128/` 是 Apache 文件列表，**不是 PEP 503 simple index**，uv 会报 "no version of torch{sys_platform == 'linux'}==2.7.1"。必须改成 `{ url = "..." }` 直接 URL：

```toml
# 替换 pyproject.toml 中的 [tool.uv.sources] 块
[tool.uv.sources]
torch = [
    { url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.10'" },
]
torchvision = [
    { url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.10'" },
]
# triton: 删 pytorch-cu128 引用，让 uv 走 PyPI 默认
```

`prefetch_uv_cache.sh` 已自动 patch + restore 不污染 git。

### B. 临时 pip install
```bash
pip install torch torchvision --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128
```

### C. 全局 pip 配置（写入 ~/.config/pip/pip.conf）
```ini
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
extra-index-url = https://mirrors.aliyun.com/pytorch-wheels/cu128
```

## Mirror coverage matrix

| Repo | 直连 | hf-mirror | AutoDL proxy | **aliyun** |
|------|------|-----------|--------------|------------|
| pypi.org（常规 Python 包）| ✓ 慢 | × | × | ✓ <code>mirrors.aliyun.com/pypi/simple/</code> |
| download.pytorch.org（torch/vision）| ✗ TLS EOF | × | ? | ✓ <code>mirrors.aliyun.com/pytorch-wheels</code> |
| pypi.nvidia.com（tensorrt 等）| 慢 | × | ✓ proxy 必经 | × |
| huggingface.co（模型）| 不稳 | ✓ 公开 only | ✓ gated | × |

## 自动化集成
[[autodl-cloud-finetune-playbook]] §4 + `LeIsaac/scripts/cloud/local/prefetch_uv_cache.sh` 已内置 sed 替换 + 自动 restore pyproject。下次跑 `prefetch_uv_cache.sh` 不需要手动改。

## 验证
```bash
curl -sI --max-time 5 "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
# 期待 HTTP/2 200 + content-type: application/zip
```
