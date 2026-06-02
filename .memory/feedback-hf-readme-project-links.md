---
name: feedback-hf-readme-project-links
description: HF model card 顶部插入与本 ckpt 任务相关的 GitHub repo 链接（不是固定两个，按相关性选）— 给自己引流但不强塞无关项目
metadata:
  type: feedback
---

发布到 HuggingFace Hub 的 model card README（`pretrained_model/README.md`）顶部，**图片之后、TL;DR 之前**插入"项目仓库 / Project repo(s)"链接区，引流回 GitHub。

**Why:** 用户原话 "相当于给自己带点流量"，但**只链相关项目** — 2026-06-02 在 `wsagi/MimicKit-G1-LAFAN` (motion-tracking) 上强塞 LeIsaac (pick-orange) 链接被用户砍掉："不想干的"。无关项目的反向链接稀释 HF SEO 信号 + 让卡片看着像 spam。

**How to apply:**

按 ckpt 所在领域选链接：

| ckpt 领域 | 链接 |
|---|---|
| Manipulation / pick-orange (ACT, DP, π0.5, GR00T, SmolVLA, X-VLA) | `vitorcen/isaaclab-experience` + `vitorcen/LeIsaac` |
| Motion-tracking (MimicKit, ProtoMotions, AMP, ADD) | `vitorcen/isaaclab-experience` 单条即可 |
| Locomotion / RL-only (rsl_rl baseline) | `vitorcen/isaaclab-experience` 单条即可 |
| 双线都涉及（罕见） | 两条都加 |

固定格式（中文为主 + 英文对照标点，**复数 / 单数**视情况）：

```markdown
**🔗 项目仓库 / Project repo(s)**：
- 🧪 [vitorcen/isaaclab-experience](https://github.com/vitorcen/isaaclab-experience) — <这个 ckpt 在该 repo 的具体角色：脚本路径 / 设计文档 / 训练入口>
- 🍊 [vitorcen/LeIsaac](https://github.com/vitorcen/LeIsaac) — <仅 manipulation ckpt 才放，注明是 pick-orange leaderboard / sim eval framework>
```

插入位置：模型截图之后、`## TL;DR` 之前（如果没截图就在描述段之后）。

**判断 rule of thumb：** 在 README 里把每条链接的 description 写出来；如果你写不出"为什么这个 repo 跟当前 ckpt 直接相关"，就别放它。

**已应用到的 ckpt READMEs：**
- `wsagi/ACT-PickOrange` — 两条都放（manipulation）
- `wsagi/DiffusionPolicy-PickOrange` — 两条都放（manipulation）
- `wsagi/MimicKit-G1-LAFAN` (2026-06-02) — 只放 isaaclab-experience（motion-tracking，LeIsaac 无关）

关联：
- [[feedback-style]] — 设计文档先行 + 中英对照规则
- [[feedback-hf-frontmatter-datasets-basemodel]] — YAML frontmatter 必填 datasets + base_model
