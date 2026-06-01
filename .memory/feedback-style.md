---
name: feedback-style
description: 用户在 LeIsaac / π0.5 项目里多次确认的协作偏好 — 设计先行、目录语义清晰、开源化思维、本地优先
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

## 设计文档先行 + 关键描述中英对照

写非平凡代码前先出设计文档，**HTML 格式 + 文件名英文 + 内容中文 + 内嵌 SVG 框图 + 关键描述插入英文对照**。

**Why:** 用户明确说 "先写个训练设计文档html，文件名英文，内容中文写，内嵌svg 框图"。后续追加 "文档在关键描述时候插入中英对照……方便开源协作"。HTML 在浏览器直接看，SVG inline 无外部依赖；中文表达技术细节更精准，英文对照让开源协作者 / 外部贡献者也能读

**How to apply:**
- 启动重要工作（训练 recipe / 仓库重构 / 新功能）前 30 分钟先写设计文档放 `docs/<name>.html`。不要用 Markdown 替代——失去 SVG inline + 排版
- 关键描述（标题、TL;DR、根因小标题、warn/info/bad/ok 框、清单核心项）下方加 `<span class="en">...</span>` 英文对照（CSS：斜体灰色 #555 / font-size 0.92em / display block）。不要加 "EN:" 前缀；不要每句话都加（次要叙述纯中文）；只在<b>关键描述</b>加

## 目录按语义分类，不按工具

`scripts/finetune/` 跟 `scripts/train/` 是两种不同语义（有 vs 无 pretrained base），不能混。
按 model family 分子目录（`smolvla/`, `openpi/`, `groot/`），**不**按训练框架（不要顶层 `mlx/`, `pytorch/`，那是 backend 子目录）。

**Why:** 用户原话 "不叫mlx了，放到scripts/finetune 里面按smolvla和openpi目录分，diffusion policy这些不是微调是训练 放 scripts/train是否合适？"

**How to apply:** 新增训练脚本前先想这是 finetune 还是 train，归到对应 family 子目录。框架（MLX/PyTorch/JAX）作为 family 内的 backend 子目录

## 开源化是默认目标

scaffold 要满足"方便别人复用"：pyproject.toml + LICENSE (Apache-2.0) + .gitignore + 真正的 Python 包结构 (`src/<pkg>/`)，不是脚本堆。

**Why:** 用户原话 "脚手架也合理整理存放，方便开源项目别人复用"

**How to apply:** 看到 `scripts/foo.py` 散文件就考虑升级为 `src/<pkg>/foo.py`；硬编码 `/home/david/...` 路径全清掉，改 CLI args 或环境变量；模块顶层 docstring 写清 Usage 例子

## 本地优先，远程是补充

"本地来就行 忽略远程"。能在本机解决就别绕远程。

**Why:** 用户原话 "本地来就行 本地有 ~/work/LeIsaac，忽略远程吧"。远程 Mac M5 Max 是有用的训练资源，但日常 dev / 文档 / 重构动作都在本机做

**How to apply:** 重构、写文档、调脚本都用本机路径；只在需要远端独有资源（如 MLX 训练、Apple Silicon-only 工具）时才 SSH

## 负面结果如实写

调试 / 实验失败要诚实记录（不洗白成"待优化"），并分析根因。

**Why:** 用户接受 0/3 的结果，要求继续诊断原因（缺陷分析）；从未要求我美化结论。明确说 "不洗白成"待优化""

**How to apply:** README / 模型卡 / 设计文档要有 "Known Results"、"Limitations" 章节；负面结果用红色高亮（如设计文档里的 ❌ 0/3）；列出排除的假设 + 锁定的根因

## 不问废话，直接做事

用户说 "no clarifying questions / 你来定" 就别再确认。做出合理决策、继续推进、问题暴露时再 redirect。

**Why:** session-level system reminder 多次强调 "operate autonomously"

**How to apply:** 边界 case 自己判断（如双仓重构 → 先小后大；OOM 处理 → bf16 优先）；用户重定向了再调整方向

相关：[[pi05-pytorch-training]]、[[pi05-repo-layout]]
