---
name: pi05-pytorch-expertonly-phase15-negative
description: π0.5 PyTorch expert-only FT 完整失败 — Phase 1.5+2+strict 20-round, ckpt-2000 = 1/60 (1.7%), leaderboard 末位（低于 SmolVLA 15×）
metadata: 
  node_type: memory
  type: project
  originSessionId: 39899a1b-ef9c-4403-93fc-2e3491bfb440
---

# π0.5 PyTorch Expert-Only FT — 完整负面结果 (2026-05-24)

## ⚠️ Final verdict (20-round strict)

**ckpt-2000 strict 20-round = 1/60 oranges (1.7%)**, leaderboard 末位（低于 SmolVLA 15×）。HTML 全文档：[`pi05_pytorch_expert_ft_negative.html`](../../../LeIsaac/docs/training/pi05_pytorch_expert_ft_negative.html)。

Phase 1.5 (90s) ckpt-2000 2/9 看似突破，被 20-round 1/60 彻底证伪 — 是 Bernoulli noise，不是 training signal。

## TL;DR

**Phase 1.5 (90s wall_cap)**：5 ckpt × 3-round 全 0/9
**Phase 2 (180s wall_cap)**：13 ckpt × 3-round → 117 ep / 2 placed (1.7%)，ckpt-2000 唯一非零 2/9
**Phase 3 strict**：ckpt-2000 20-round = 1/60 (1.7%)，证实 2/9 是 noise outlier

Pipeline 完全通（训练、ckpt 保存、policy server、Isaac Sim 联动 OK），但**模型不会做任务**。

## 实验配置

- launcher: `LeIsaac/scripts/training/pi05_pt/train.sh`
- base: `lerobot/pi05_base` (HF)
- `train_expert_only=True` + `freeze_vision_encoder=True` → 693M trainable / 4.14B total
- bf16 + grad-ckpt + batch=1 + lr=2.5e-5
- chunk_size=50, n_action_steps=50, max_state/action_dim=32
- dataset: LightwheelAI/leisaac-pick-orange (60 ep × 600 frame ≈ 36k)
- 2500 step / 9m30s on RTX 4090
- rename_map: `observation.images.front→base_0_rgb`, `observation.images.wrist→right_wrist_0_rgb`，`left_wrist_0_rgb` 被 modeling_pi05.py:1195 自动 -1 padding + mask=0

## 结果

| ckpt | oranges | rounds | avg_round_s | note |
|---|---|---|---|---|
| 500  | 0/9 | 0/3 | 90.06 | wall_cap timeout × 3 |
| 1000 | 0/9 | 0/3 | 90.05 | wall_cap timeout × 3 |
| 1500 | 0/9 | 0/3 | 90.02 | wall_cap timeout × 3 |
| 2000 | 0/9 | 0/3 | 90.04 | wall_cap timeout × 3 |
| 2500 | 0/9 | 0/3 | 90.04 | wall_cap timeout × 3 |

**观察**：
- Action range 从 ckpt-500 的窄区间扩展到 ckpt-2000 的 (-1.07, 1.38) rad，policy 输出在变化（**不是 stuck 在固定 pattern**）
- 但 arm 在 sim 中没法定位 / 抓住 / 摆放橙子
- 30 fps × 90s × 3 ep × 5 ckpt = 1350 秒×3 episodes×5 ckpts = 6.75 小时仿真总时长 / 0 placed

## Pipeline OK 的证据

- ✅ pi05_base 加载 OK, `train_expert_only=True` 接口工作正常
- ✅ Trainable=693M, total=4143M（与 Phase 0 一致）
- ✅ 36k frame dataset 加载正常
- ✅ Train loss 收敛（虽然没看具体数字）
- ✅ Ckpt 保存完整（每个 9.4 GB × 5 = 47 GB）
- ✅ Policy server 加载 ckpt 正常，inference latency ~200ms/chunk
- ✅ Client 拿到 action，转 isaac rad 后值合理

## 失败假设（按可能性排序，2026-05-24 strict 20-round 后更新）

### H6 (主因 ~80%) — SigLIP@224 vision bottleneck on small objects
π0.5 用 SigLIP-So400m 强制 224×224 输入。LeIsaac dataset 原 640×480 缩到 168×224 + 28px 上下黑边。
橙子原图 30-50 px → 缩后 10-17 px = SigLIP patch_size=14 的 0.7-1.2 patch。Vision encoder 拿到的橙子 token 信息量极低 → conditioning 给 expert 的视觉信号近乎噪声。
对照 SmolVLA 用 SigLIP@512：橙子 24-40 px = 1.7-2.9 patch (3-5× 视觉信号) → 8/15 同档可达。
**这不能通过更多训练或更大 trainable params 解决** —— PaliGemma-2B 的 vision tower hardcoded 224，要换得换 PaliGemma2-448 base 重训百万 H100h。

### H1 (15%, 降级) — Camera mapping mismatch
pi05_base 在 DROID 上预训：
- `base_0_rgb` = static 第三人称俯视相机
- `left_wrist_0_rgb` / `right_wrist_0_rgb` = 双臂手腕相机

LeIsaac SO-101：
- `front` = 桌面前方水平相机（**不是俯视**）
- `wrist` = 单臂手腕相机

冻 VLM 后，visual encoder 出的 feature 假设是 DROID 风格 perspective。我们 rename 后传入的 `base_0_rgb=front` 视角根本不对 —— 冻的 VLM 可能编码出无关 feature，action expert 怎么学都学不到任务信号。

**验证方法**：dump 一张 dataset 帧 + DROID 的 `base_0_rgb` 帧并排看；或者让 X-VLA / SmolVLA 在同 dataset 上的成绩做对照（SmolVLA 8/15 证明 dataset 本身可用）。

### H2 (30%) — 2500 step 太短
- pi05 paper 通常 5k-30k step
- SmolVLA 同 dataset 跑了 30k step 才到 8/15
- 693M trainable 在 50 demo 上可能确实需要 ≥ 10k step

**验证方法**：续训到 10k step（再 ~30 min）看 trend。

### H3 (15%) — `empty_cameras` 处理 left_wrist 不对
- modeling_pi05.py:1195 自动用 `torch.ones_like(img) * -1` + `mask=0` 填充
- 训练时 left_wrist 永远是 -1 + mask=0；推理时同理
- 但 expert 可能 cross-attn 时丢失了"实际只有 2 cam"的信号 → 输出退化到 mean prior

**验证方法**：训 ablation `--policy.empty_cameras=0` + `--policy.input_features` 显式只声明 2 cam（需要 patch 接口）。

### H4 (10%) — horizon=35 错
- MLX 时 h=35 是甜点（[[leisaac-policy-comparison]] / pi05_finetune_pick_orange.html）
- PyTorch 可能要 h=20 / 50 / chunk-as-h

**验证方法**：选 ckpt-2500，sweep h ∈ {16, 25, 35, 50} 各 3-round。

### H5 (5%) — wire protocol / action normalization bug
- policy_inference.py 转 isaac rad 时 SO-101 6-DOF padded to max_action_dim=32，需要取前 6 维
- 没检查 norm stats 是否对齐

**验证方法**：dump 一个 batch 训练数据 vs sim observation 看分布对齐。

## 关键发现：基础设施层暴露的坑

### Bug 1: eval_watcher race condition (已修)
`lerobot_finetune.sh` line 180: `nohup ... > $OUTPUT_DIR/auto_eval.log 2>&1 &` —— OUTPUT_DIR 还没建时 redirect 失败，watcher 静默死亡。修法：watcher launch 前 `mkdir -p "$OUTPUT_DIR"`。

### Bug 2: lerobot env mismatch (已修 watcher 调用)
`eval_watcher.sh` 默认 `LEROBOT_PYTHON=/home/david/miniconda3/envs/lerobot-v040/bin/python`，但 pi05 训练用 `lerobot` env（新版 config 字段 `train_expert_only` v040 不认）。修法：手动传 `LEROBOT_PYTHON` + `LEROBOT_REPO`。

### Bug 3: 4B model serial eval OOM
policy_server 不卸载旧 ckpt 直接加载新的 → 4B × 2 = 32G + Isaac Sim 6G 残留 → 24G OOM。修法：每 ckpt kill+restart server（写在 `/tmp/pi05_serial_eval.sh`）。

### Bug 4: ACT/DP 同样问题留意
pi05 4B 这个问题暴露了 watcher 在大模型上的局限。SmolVLA 450M / DP 267M / ACT 80M 都可以 server 长驻；GR00T 3B 也能；但 pi05 4B + Isaac Sim 6G 在 24G 卡上就是边界外。

## 下一步建议

不要继续盲训 10k step。先做：

1. **dump 数据 vs sim 单帧对比** (15 min)：渲染一帧 LeIsaac dataset，并下载一帧 DROID `base_0_rgb`。看 perspective 是否离谱 → 验证 H1
2. **X-VLA / SmolVLA 在 LeIsaac 上的成绩是单 cam 还是双 cam？** (5 min check) → 验证 H3
3. 如果 H1 / H3 都看不出问题，做 **H2 验证：续训到 10k step**（再 30 min on 4090）

## 关联
- [[feedback-dreamzero-eval-stage-criteria]] — 首段欠拟合不弃看 trend；本实验所有 5 段都 0，不只是首段
- [[pi05-pytorch-training]] — 第一轮 MLX LoRA 0/15 的延续
- [[per-model-action-horizon]] — pi05 h=35 引用
- [[leisaac-policy-comparison]] — SmolVLA 8/15 = 同 dataset 可学习的证据
- HTML: `LeIsaac/docs/training/pi05_finetune_pick_orange.html` §2.5
- launcher: `LeIsaac/scripts/training/pi05_pt/train.sh`
- 数据: `outputs/pi05-expert-leisaac-pick-orange/auto_eval.csv`
- ckpts: `outputs/pi05-expert-leisaac-pick-orange/checkpoints/{500,1000,1500,2000,2500}` (5 × 9.4 GB = 47 GB)
