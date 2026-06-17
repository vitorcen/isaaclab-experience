---
name: gr00t-n17-v4-v5-recipe-align-live
description: GR00T-N1.7 配方对齐实验 v4–v11 已收尾(2026-06-17)——真根因=warm完整预训练头,最优=冻视觉+batch48=v10-4500=81%已发布V2(超hi-space67% +14);全过程见recipe HTML文档
metadata:
  type: project
---

# GR00T-N1.7 配方对齐实验 v4–v11（已收尾 2026-06-17）

**全过程设计文档(权威)**：`LeIsaac/docs/training/gr00t_n17_recipe_alignment_experiments.html`（v4–v11 全景：翻案 TL;DR + 根因弧线 + 架构图 + 2×2 矩阵 + GNS + 发布）。本文件只留 durable 结论。

## 🏆 最终结论
- **SOTA = v10-4500（warm 完整预训练头 + 冻视觉 + batch48，5.95ep）= 81.0%@100-round（81.1%@60）**，超 hi-space N1.7 67% **+14 点**。已发布 [`wsagi/GR00T-N1.7-V2-PickOrange`](https://huggingface.co/wsagi/GR00T-N1.7-V2-PickOrange)（旧 cold-start v1 归档 `n1.7-v1` 分支）。
- **2×2 矩阵（batch × freeze，真交叉）**：冻视觉{batch32 v8=70%@20 → batch48 v10=81.1%@60，+11}；解冻视觉{batch32 v9=74.4%@60 → batch48 v11=68.9%@60，−5.5}。**batch48 帮冻视觉、害解冻视觉；解冻视觉天花板 ~72-75% 加大 batch 救不了。全局最优=冻视觉+batch48。**

## 🎯 真根因（逐步翻案后锁定）
起点误判"差异=batch/jitter/bf16 三项配方"全被受控实验证伪（v4≈v6 证 bf16 无关；去 jitter 的 v7≈带 jitter v5；纯 batch32 冻视觉 v6 只 25%）。**真根因 = 我们 v4–v7 全部 cold-start from Cosmos（`skip_weight_loading=True`），动作头是半尺寸 16 层（249 张量）从零随机训；hi-space 微调的是 GR00T-N1.7-3B 自带的完整 32 层预训练头（537 张量 + vl_self_attention）。**
- **确切 bug**：`launch_finetune_ckpt_n17.py` 用 `get_default_config()`（吃过时的 `gr00t/configs/model/gr00t_n1d7.py` 16 层默认），不从 base 的 config.json 读架构 → 32 层权重塞进 16 层壳，多余层静默丢弃。
- **修复 = v8/v9 WARM full-head sync**：warm-start 时从 base config.json 同步 `diffusion_model_cfg.num_layers=32` + `vl_self_attention_cfg`，**保留 select_layer=12**（hi-space 值，非 base 的 16），assert num_layers==32。v8 冻视觉即 70%（vs 从零头 v6 的 25% = +45 点），坐实根因。
- **官方 launcher 不踩此坑**：GR00T 官方 `gr00t/experiment/launch_finetune.py` 走 `from_pretrained` 读 ckpt 自己的 config.json 重建架构 + 严格 missing/unexpected 校验 → 不可能静默丢张量。**只有 PickOrange 魔改的 cold-start launcher 才半头从零**；LeSONIC 的 GR00T 线（走官方 launcher）干净无此问题。关联 [[starvla-gr00t-v2-n17-head]]（StarVLA QwenGR00T_v2 是另一回事=Qwen 骨干 head 真从零的硬限制，非 bug）。

## 🧪 GNS 临界批次探针（48→64 该不该加）= 失败但有结论
两跑法被 step-based LR schedule + 不同 θ 混淆（tr(Σ) 算出负=废）。codex+mimo 双评审：①诊断对（GNS 是固定-θ 恒等式）②**AdamW 坑**=GNS 是 SGD 理论、grad_norm 是原始梯度，Adam 预条件后真实临界批次差 2-10×③最优正解=单-ckpt 配对微批估计器（同一 θ 采 K 微批，~10 行/3min）。**bottom line：48→64 只降噪 25% < ±9% eval 噪声地板=看不见 → batch 封板 48，卡留给数据量/视觉分辨率。**

## 关键教训（已外化到专门记忆）
- **MSE 膝点定 best**（v10/v11 双案例）→ [[feedback-three-tier-eval-funnel]]
- **行为质感骗眼睛**：v11 看 GUI"更聪明"(env_success≥1橙~90%)但 oranges% 反低；定榜必须严格 oranges% × ≥60-round。
- **quick wall_cap 必须==strict 180**（150 截断慢策略）→ [[feedback-headed-eval-default]]
- **HF 大文件上传 XET 坑**：hub≥0.36 默认 XET 在代理链路卡死 → `HF_HUB_DISABLE_XET=1` → [[hf-upload-tricks]]
- 真 epoch = step × effective_batch / 36293（trainer_state.epoch 是 GR00T shard 伪值，别用）。
