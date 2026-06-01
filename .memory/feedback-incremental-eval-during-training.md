---
name: feedback-incremental-eval-during-training
description: 长训练默认按总步数 10 等分，每完成 1/10 跑一次 1-round quick eval，早发现配置错误避免白训
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1e92da1e-d4b0-4cfa-b922-39f17de1ca36
---

# 长训练每 1/10 步必须 quick eval

**Rule**: 任何 LeIsaac 训练任务（ACT / DP / SmolVLA / VLA / 任何 lerobot policy）默认按总 step 数 **10 等分**（e.g. 100k → 10k 一格，20k → 2k 一格）。每完成一格，对最新 ckpt 跑 **1-round 3-ep quick eval**（不需要 full 5-round 协议）确认 policy 真的在学任务，而不是只在降 loss。

**Why**: 2026-05-22 DP v0.4.0 实际事故 — 跑 10h 训到 100k step，loss 从 0.554 → 0.011 看着收敛，但 lerobot v0.4.0 的默认 `crop_shape=(84,84)` 从 480×640 帧切 84×84 块（**1.7% 视野**），policy 收敛到 "do nothing" 退化先验。每个 rollout 都 stuck @ 35s, 0/15 oranges。10h 全废 — 训练完才发现。**任何一次 10k 处的 1-round eval 都能立刻抓到**。

**How to apply**:
- 设 `STEPS=100000 SAVE_FREQ=10000`（10 等分）/ `STEPS=20000 SAVE_FREQ=2000` 类似
- 每个新 ckpt 落地 → 立即跑 **X-VLA-style quick eval**: `EVAL_ROUNDS=3 EPISODE_S=60 MAX_ROUND_WALL_S=90`（3-5min/eval × 10 slice = 30-50min 总开销）。canonical 实现见 `scripts/auto_sweep_xvla_ckpts.sh`
- 两个 abort 信号:
  1. **oranges placed = 0** 跨连续 3 个 slice（如 30%, 40%, 50%）→ abort
  2. **arm joint range < 0.1 rad / stuck < 30s/ep** 连续 3 slice → policy 学到退化先验，abort
- abort 后先 diff `train_config.json` 跟公开 baseline (shadowHokage/act_policy for ACT, wsagi/DiffusionPolicy-PickOrange for DP)，找配置差异

**Cross-check 在 train start 前**: 启动长跑前先 diff 新 train_config.json vs 公开 baseline，任何字段差异都要 deliberately 判断是不是故意。DP 事故里 `crop_shape: [84,84]` (ours) vs `crop_shape: None` (wsagi) 就是 smoking gun，事前 diff 就能发现。

**已写入** `LeIsaac/CLAUDE.md`（项目级强制 instruction，英文）+ 本 memory（feedback type，跨 session 持久化）。

## 关联

- [[act-framework-drift-root-cause]] ACT 框架 drift 调查（成功，3/15 → 8/15）
- [[feedback-pre-run-gpu-check]] 启动 GPU 任务前 nvidia-smi 检查（同类前置检查 hygiene）
- [[feedback-training-resume-chunks]] 长训练拆 5 段 resume（与本 rule 兼容，每段结束都该 eval）
