---
name: feedback-vla-epoch-budget-6ep
description: PickOrange(60-demo 小数据)VLA 微调 best ckpt 通常落在 3-4 epoch;新训练默认 max 训到 ~6 epoch(覆盖峰+略过看悬崖)
metadata:
  type: feedback
---

**规则**:这个 60-demo leisaac-pick-orange 小数据 regime 下,VLA 微调(冻结-VLM 训 head / lerobot 从头)的 **best ckpt 通常在 3-4 epoch**;**新训练默认 `max_steps` 设到 ~6 epoch 附近**,eval 从 ~3 ep 起每 ~0.5-1 ep 一个点。

**Why(实测 best epoch,N=36293 frames=1 ep)**:
- GR00T-N1.7 best 5.3 ep · GR00T-N1.6 best 2.9 ep · StarVLA-8B-GR00T best 3.3 ep · 4B-GR00T best 4.0 ep · **PI_v3-8B best 4.3 ep(78k=63.3%)**。多数落 3-5 ep。
- <6 ep 可能在峰之前切断(**2B PI_v3 计划 30k=3.3 ep 偏早**,可能没到峰);>6 ep 纯浪费/过拟合(**DP 训到 22 ep 仍 0/60**;OpenVLA 0.6 ep / π0.5 1.3 ep 则严重欠训)。6 ep 覆盖峰区 + 略过一点正好看到悬崖。

**How to apply**:
- `max_steps = round(6 × total_frames / global_batch)`;`save_interval = round(1 × total_frames / global_batch)`(每 epoch 存一个,eval 3/4/5/6 ep)。total_frames 取数据集 `meta/info.json` 的 `total_frames`。
- 用 epoch(不是 step)做跨 run 比较与早停判断,见 [[feedback-smoke-500step-quick-gate]] 与 README leaderboard 的 Epoch 列口径(`step×global_batch/36293`)。
- ⚠️ caveat:此 3-4 ep 经验是**60-demo 小数据 + 冻结/LoRA 微调**专属;数据量变大峰 epoch 会显著前移,需重新标定。

关联 [[feedback-smoke-500step-quick-gate]] [[three-box-sweep-live-state]] [[starvla-vlm-variant-2b-4b-8b]]。
