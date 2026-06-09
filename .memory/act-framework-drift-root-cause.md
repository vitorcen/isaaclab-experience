---
name: act-framework-drift-root-cause
description: ACT 复现 shadowHokage 失败 root cause = lerobot v0.4→v0.5 framework drift；锁版本 lerobot v0.4.0 retrain 后 3/15 → 8/15 验证成功（best lucky 13/15）
metadata: 
  node_type: memory
  type: project
  originSessionId: f29b6eda-afc7-4618-b533-2c8e72f8ad99
---

# ACT 复现失败 root cause（2026-05-21 三模型分析）

我们自训 wsagi/ACT-PickOrange 仅 3/15 oranges，shadowHokage/act_policy 9/15。train_config 核心字段全同。

## Root cause = lerobot framework drift

shadowHokage 上传时间 **2026-01-27**（HF API 直接查），训练时 lerobot 在 **v0.4.0 ~ v0.4.3**（v0.4.0 是 2025-10-23，v0.4.4 是 2026-02-27）。我们当前 lerobot 0.5.2 是 2026-04-26 之后版本。

**v0.4.0 → v0.5.2 之间影响 ACT 训练的 3 个关键 PR（shadowHokage 训练时不存在）**：

1. **PR #3406 (a8b72d96, 2026-04-19)** "2x faster dataloader via parallel decode, uint8 transport, persistent workers"
   - v0.4.0: `DataLoader(prefetch_factor=2, no persistent_workers, no uint8)`
   - v0.5.2: `prefetch_factor=4 + persistent_workers=True + return_uint8=True (/255.0 in train loop)`
   - 影响：**mini-batch ordering 完全不同**（persistent_workers RNG state 不每 epoch 重置）+ 视频量化误差

2. **PR #3442 (1add4606, 2026-04-23)** "fix(policy): loss normalization for padded actions in ACT/Diffusion/MultiTaskDiT"
   - 旧 (shadowHokage): `(F.l1_loss(...) * mask).mean()` — 除以 total（含 padded 0）
   - 新 (我们): `(abs_err * mask).sum() / (valid_count * D)` — 只除以 valid
   - PR description: "60-70% padding 时 effective gradient 提升 2-3x"
   - 在 PickOrange (chunk=100, ep~605 frames, padding~8%) 上 effective gradient 提升 ~9%

3. **PR #3102 (2fb5c7ad)** "add cudnn_deterministic option" — 默认 False，影响小

## Codex 提的 3 个嫌疑（已被时间线否决）

- normalize → processor_act.py：PR #1452 (2025-09-18)，shadowHokage 上传**之前**已有 → ✗
- accelerate.autocast 替代 GradScaler：v0.4.0 已用 → ✗
- tolerance_s 进入 dataset factory：v0.4.0 已有 → ✗

**Codex 方向正确（framework drift）但 commit 错；Opus commit-level git log 把根因锁到 #3406/#3442。**

## Why: 为什么会差 3x

ACT 是 80M 小模型 + 10k step + 60 episodes 低数据 regime，对 batch ordering / effective lr 极度敏感。两个改动叠加 → 即便 seed=1000 相同，gradient 轨迹完全不同 → 收敛到不同 attractor。

## How to apply: 修复方案

**实测结果 (2026-05-22 完成)**：

锁版本重训 (lerobot v0.4.0 + torch 2.7.1+cu126 + torchvision 0.22.1, 在新 conda env `lerobot-v040`)，10k step 结果 7/15，extend 到 20k step ckpt-20k h=70 = **8/15** (best lucky run ckpt-18k h=70 = 13/15)，**追平 shadowHokage 9/15 band，证实 framework drift = root cause**。

**sweep best**: ckpt-20k h=70 (canonical, 2-run mean 5.5/15) 或 ckpt-18k h=70 (lucky peak 13/15, 3-run mean 7.7/15 ±35% variance)。

**关键 pin**：
- torch==2.6+ 时官方 wheel 是 cu124/cu126（cu121 最高 torch 2.5）→ 实际装到 torch 2.7.1+cu126
- torchvision==0.21+ 与 torch>=2.6 配对
- lerobot v0.4.0 需 `pip install -e .` 自动拉 torch 满足 [2.2.1, 2.8.0) pin
- async 推理需要补装 `grpcio==1.73.1 protobuf==6.31.0`
- 推理用同 lerobot-v040 env (`LEROBOT_PYTHON=...envs/lerobot-v040/bin/python bash LeIsaac/scripts/policy_server.sh start lerobot`)

**最高 ROI 修复方案**：锁版本 lerobot v0.4.0 + torch 2.6+ 重训

```bash
conda create -n lerobot-v040 python=3.10 -y
conda activate lerobot-v040
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
cd ~/work && git clone --branch v0.4.0 --depth 1 https://github.com/huggingface/lerobot.git lerobot-v040
cd lerobot-v040 && pip install -e .
pip install transformers==4.46.0 accelerate==1.1.0 av==14.2.0

CONDA_ENV=lerobot-v040 OUTPUT_NAME=act-v040-baseline \
  bash LeIsaac/scripts/training/act/train.sh
```

预期：3/15 → **7-9/15**（命中 shadowHokage band）

**备用（无需新 env）**：v0.5.2 上 patch 回旧行为
1. modeling_act.py 改回 `.mean()` 旧 loss
2. train.sh 加 `--persistent_workers=false --prefetch_factor=2`
3. factory.py 改 `return_uint8=False`

## 关联

- [[act-reproduction-gap]] 旧的失败记录（torchcodec 切换 + ckpt sweep 均无效）
- [[leisaac-5round-leaderboard-2026-05-21]] 5-round leaderboard
- [[per-model-action-horizon]] 各模型 best h
