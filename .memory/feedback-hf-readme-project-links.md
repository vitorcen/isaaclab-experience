---
name: feedback-hf-readme-project-links
description: 发到 HF Hub 的 model card README 顶部必须放 vitorcen/isaaclab-experience + vitorcen/LeIsaac 两个项目仓库链接 — 给自己带流量
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

发布到 HuggingFace Hub 的 model card README（`pretrained_model/README.md`）顶部，**图片之后、TL;DR 之前**插入项目仓库链接区，引流回我们的 GitHub。

**Why:** 用户原话 "相当于给自己带点流量"。HF 模型卡是被搜索/直接打开的入口（dataset 页面 filter / Google 搜索），但 GitHub repo 的 star/issue/exposure 完全独立于 HF。在每个发布的 ckpt README 顶部加两个 repo 链接 ≈ 0 成本 cross-promotion。

**How to apply:**

固定格式（中文为主 + 英文对照标点）：

```markdown
**🔗 项目仓库 / Project repos**：
- [vitorcen/isaaclab-experience](https://github.com/vitorcen/isaaclab-experience) — Isaac Lab + LeIsaac 多策略横评（parent project）
- [vitorcen/LeIsaac](https://github.com/vitorcen/LeIsaac) — LeIsaac fork（训练脚本 + 设计文档 / training scripts + design docs）
```

插入位置：模型截图之后、`## TL;DR` 之前（如果没截图就在描述段之后）。

对于特殊模型（如 pi05），第一个链接的 description 可以追加："含 π0.5 PT server `server/serve_pi05.sh`" 这种额外引导。

**已应用到的 ckpt READMEs**（截至 2026-05-17）：
- `wsagi/ACT-PickOrange` (act-leisaac-pick-orange/checkpoints/010000/pretrained_model/README.md)
- `wsagi/DiffusionPolicy-PickOrange` (dp-leisaac-pick-orange-v2/checkpoints/last/pretrained_model/README.md)
- pi05 v3 (pi05-leisaac-pt-v3/pretrained_model/README.md，待 upload HF)

后续新发的任何 model card README 都必须有这一段。

关联：[[feedback-style]] 设计文档先行 + 中英对照规则
