---
name: feedback-autodl-cost-discipline
description: "AutoDL 云端训练每步都先问\"这步需要 GPU 模式吗\"，无卡模式 ¥0.1/h vs GPU 模式 ¥6-8/h 差 60-80×；常态反思如何省钱"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 39899a1b-ef9c-4403-93fc-2e3491bfb440
---

# AutoDL 云端训练成本纪律

**Rule**：在 AutoDL 上做任何 step 之前先自问 *"这步真的需要 GPU 模式吗"*；能在无卡模式做的就在无卡模式做（¥0.1/h vs ¥6-8/h 差 60-80×）。训练流程整体目标：**最小化 GPU 模式时长**。

**Why**：本 session 实战教训 — uv sync 在 AutoDL proxy 限速下花了 2.5 小时反复挂在 tensorrt build / SHA mismatch / wheel_stub stuck，期间 GPU 模式按 ¥6-8/h 计费空烧 ~¥18-20。
正确认知应该一上来就：
- clone repo / 下 dataset / 下 base model：无卡模式（¥0.1/h × 1h = ¥0.1）
- uv sync 卡 tensorrt：发现 tensorrt 训练时 0 import 后**立刻砍掉 dep**，不要硬磕
- 真训练才切 GPU 模式

## How to apply

每个 step 跑之前过一遍 checklist：

| Step | 需要 GPU? | 原因 |
|------|-----------|------|
| git clone / git lfs pull | ❌ | 纯网络 |
| hf download / scp model / dataset | ❌ | 纯网络 |
| `uv lock` / 解析依赖 | ❌ | 纯 metadata |
| `uv sync` install from cache | ❌ | 纯 copy/link |
| `uv sync` build wheel（如 tensorrt-cu12-libs）| ⚠️ 仅当确实需要 | 大部分时间该砍依赖 |
| import test（torch / flash_attn / transformers）| ❌ | python import 不真用 GPU |
| smoke 50 step | ✅ | 真训练 |
| 正式训练 | ✅ | 真训练 |
| ckpt 上传 HF | ❌ | 纯网络 |

**强制反思触发点**：
- 任何步骤跑超 10 分钟没明显进展 → 停下来问"这真的是 GPU 模式才能做的事吗？能砍依赖 / 换路径 / 切无卡模式吗？"
- 任何依赖 build 失败 → 先 grep 训练代码确认这个依赖是否真的需要（70% 的 inference-only 依赖训练用不到）
- 任何代理限速导致下载 < 1 MB/s → 测 aliyun 直连 / no_proxy 绕过

## Why the rule's contrapositive matters
"宁愿在无卡模式多花 30 min 试错也不要 GPU 模式硬挺 30 min" — 因为前者 ¥0.05，后者 ¥3-4。失败重试在无卡模式几乎免费，在 GPU 模式按分钟烧。

## 关联
- [[autodl-uv-sync-cn-strategy]] — uv sync 在 CN 的 5 个网络墙 + 砍 tensorrt
- [[cn-pypi-mirror-aliyun]] — aliyun 直连不走 proxy 是 15 MB/s vs 1 MB/s
- `LeIsaac/docs/training/autodl_cloud_finetune_playbook.html` §6 模式切换 SOP
- `LeIsaac/scripts/cloud/autodl/uv_sync.sh` / `uv_sync_offline.sh` 已自动 detect 无卡模式
