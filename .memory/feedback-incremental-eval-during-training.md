---
name: feedback-incremental-eval-during-training
description: 长训练默认按总步数 10 等分，每完成 1/10 跑一次 5-round quick eval（快筛=5round，2026-06-12 user 定，3-round variance 太大），早发现配置错误避免白训；步数定法=~6epoch(=6×frames/batch)、save_freq取整百≈steps/10→10个ckpt；DP从零训例外(6ep可能偏短,爬坡则resume到12ep)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1e92da1e-d4b0-4cfa-b922-39f17de1ca36
---

# 长训练每 1/10 步必须 quick eval

**Rule**: 任何 LeIsaac 训练任务（ACT / DP / SmolVLA / VLA / 任何 lerobot policy）默认按总 step 数 **10 等分**（e.g. 100k → 10k 一格，20k → 2k 一格）。每完成一格，对最新 ckpt 跑 **5-round quick eval** 确认 policy 真的在学任务，而不是只在降 loss。

> **快筛轮数 = 5-round（2026-06-12 user 定，覆盖之前的 3-round）**：单次 3-round closed-loop variance ±20-40%（见 [[eval-5round-mandatory]]），用 3-round 做峰判定/abort 判定都不可靠 —— **训练过程中所有快筛默认 5-round**（sweep watcher / 增量 eval 一律）。

**Why**: 2026-05-22 DP v0.4.0 实际事故 — 跑 10h 训到 100k step，loss 从 0.554 → 0.011 看着收敛，但 lerobot v0.4.0 的默认 `crop_shape=(84,84)` 从 480×640 帧切 84×84 块（**1.7% 视野**），policy 收敛到 "do nothing" 退化先验。每个 rollout 都 stuck @ 35s, 0/15 oranges。10h 全废 — 训练完才发现。**任何一次 10k 处的 1-round eval 都能立刻抓到**。

**How to apply**:
- 设 `STEPS=100000 SAVE_FREQ=10000`（10 等分）/ `STEPS=20000 SAVE_FREQ=2000` 类似
- 每个新 ckpt 落地 → 立即跑 **X-VLA-style quick eval**: `EVAL_ROUNDS=5 EPISODE_S=60 MAX_ROUND_WALL_S=90`（5-8min/eval × 10 slice）。canonical 实现见 `LeIsaac/scripts/auto_sweep_xvla_ckpts.sh`；StarVLA sweep watcher 默认也已是 5-round（`scripts/evaluation/starvla_*sweep_watcher.sh` 的 `EVAL_ROUNDS:-5`）
- 两个 abort 信号:
  1. **oranges placed = 0** 跨连续 3 个 slice（如 30%, 40%, 50%）→ abort
  2. **arm joint range < 0.1 rad / stuck < 30s/ep** 连续 3 slice → policy 学到退化先验，abort
- abort 后先 diff `train_config.json` 跟公开 baseline (shadowHokage/act_policy for ACT, wsagi/DiffusionPolicy-PickOrange for DP)，找配置差异

**Cross-check 在 train start 前**: 启动长跑前先 diff 新 train_config.json vs 公开 baseline，任何字段差异都要 deliberately 判断是不是故意。DP 事故里 `crop_shape: [84,84]` (ours) vs `crop_shape: None` (wsagi) 就是 smoking gun，事前 diff 就能发现。

## Step 总数 + save_freq 定法（2026-06-13 user 定）

**总步数 = ~6 epoch**（小数据默认预算，见 [[feedback-vla-epoch-budget-6ep]]）；**save_freq 取整百，≈ 总步数/10 → 正好 ~10 个 ckpt**（10 等分对齐增量 eval）。

公式：`steps_per_epoch = total_frames / batch_size`；`steps ≈ 6 × steps_per_epoch`（圆到整千/整百）；`save_freq = steps/10`（圆到**整百**，user 偏好整百好记）。

**PickOrange 实例**（36293 frames, batch 16）：steps_per_epoch ≈ 2268 → 6 ep ≈ 13.6k → 取 **`--steps=14000 --save_freq=1400` = 10 ckpt（6.2 ep）**。lerobot 还会在末步额外存 final。整百备选：1000(14 ckpt 更密)/1500+steps15000(10 ckpt 更圆)。

**别存太密**：DP/lerobot ckpt 含 optimizer state ~1.3GB/个，10 个≈13GB（够），20 个≈26GB 易 ENOSPC + eval 一遍更久。watcher "拉回本机即删云端" 防爆盘。

**DP-from-scratch 例外**：DP 是从零训 CNN（非预训练 VLA），6 ep 可能偏短（原始 DP baseline 用 100k 步≈44ep）。先按 6ep/10ckpt 跑看 quick-eval 曲线；**若 ep6 仍爬坡(best 落最后一个 ckpt)→ resume 续到 12ep**；best 在中段就停。预训练 VLA 微调严守 ~6ep。

**已写入** `LeIsaac/CLAUDE.md`（项目级强制 instruction，英文）+ 本 memory（feedback type，跨 session 持久化）。

## 关联

- [[act-framework-drift-root-cause]] ACT 框架 drift 调查（成功，3/15 → 8/15）
- [[feedback-pre-run-gpu-check]] 启动 GPU 任务前 nvidia-smi 检查（同类前置检查 hygiene）
- [[feedback-training-resume-chunks]] 长训练拆 5 段 resume（与本 rule 兼容，每段结束都该 eval）
