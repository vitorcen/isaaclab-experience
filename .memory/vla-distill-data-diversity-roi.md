---
name: vla-distill-data-diversity-roi
description: 从 deterministic RL teacher 蒸馏 VLA 数据集时，多样性来源的 ROI 排序 — prompt 侧 > DR/RSI/action-noise > multi-clip > 训更长；DR 替代不了 semantic diversity
metadata:
  type: feedback
---

从 deterministic RL policy（如 MimicKit DeepMimic PPO）蒸馏 prompt-conditioned VLA 数据集时，
**怎么花算力堆数据多样性**有明确 ROI 排序。2026-06-02 Claude×GPT-5.4 第二轮评审定的。

**Why:** deterministic policy 同 seed 重放无信息增益（单 ep 多录无用）。有效多样性必须靠外部注入，
但不同注入方式 ROI 差一个量级，且**互相不能替代**。搞混了就会"花最贵的算力买最低的多样性"。

**How to apply（ROI 从高到低）：**
1. 🥇 **Prompt 侧多样化** — 同义词 / 风格模板随机（"fight"/"box"/"spar"/"fighting stance"）。几乎白送，直接打 language 泛化。**最先用，零额外算力。**
2. 🥈 **Rollout 侧 DR / RSI / action-noise** — 光照/贴图/相机位/初始姿态随机 + PPO output 加 N(0,0.02)。买 student robustness（covariate shift recovery）。
3. 🥉 **Multi-clip experts** — 多个不同源 clip 各训一个 expert。semantic / style diversity 的**主来源**。
4. 🔻 **同一 clip 训更长** — 最低，除非明确要学**长 choreography 的 phrase 顺序/过渡/节拍**。

**关键判定（最容易踩的坑）：**
- **DR ≠ semantic diversity。** `1 个 expert + 再猛的 DR` 学不会两个不同动作（如 dance1 vs dance2）的区别，只会更稳地模仿那一个。语义多样性只能靠 multi-clip / multi-task 拿。
- **"训长"对 VLA 价值 ≈ "训同段 15s"** —— 但有反例：若 student **不显式吃 phase 特征**，或目标是学长编舞内部时序结构，更长 clip 才有额外监督。默认场景（family-level prompt 切换）下不值。
- multi-clip 数量定标：**2-3 个够验证"多clip有没有用"，4-6 个才算 family coverage**，<2 = single-clip overfit 变体，>6 当前阶段不划算。
- clip 选择优先级：**不同编舞/任务 ＞ 同源不同 phrase ＞ 同 phrase 不同 subject**。"先扩内容，再扩演法。"

**元方法论：** 铺数据前先问"这条蒸馏管线通不通"（recorder/format/camera/student 能否学），管线没验证就堆数据 = 把错误管线喂得更饱。先 10ep×2task 激进 sanity 判 go/pivot。

关联：
- [[mimickit-to-vla-distill-plan]] — 这条规则的首个用例（MimicKit→VLA）
- [[pi05-pytorch-expertonly-phase15-negative]] — SigLIP@224 视觉瓶颈，提醒 vision 不是万能 cue
