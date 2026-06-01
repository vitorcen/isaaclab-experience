---
name: mimickit-lafan-fight-training-plan
description: Resume target — train MimicKit DeepMimic G1 on LAFAN_fight slice until visible motion, then launch GUI eval for user. compact 后直接 pick up.
metadata:
  type: project
---

**任务：训 MimicKit DeepMimic G1 + LAFAN_fight 切段，到能做出动作启动 GUI eval 给用户看。**

**Why:** 用户 2026-06-01 当面要求 — 验证 LAFAN-G1 motion tracking 在自家 4090 24G 上可行性，且 ecosystem 没人发 LAFAN_fight 的 G1 ckpt（TWIST 通用 tracker 用 legged_gym 不兼容；ProtoMotions 自己 framework；ASAP 不发 ckpt；MimicKit 自家只有 spinkick/walk/jump/run）— 必须自训。

**How to apply:** compact 恢复后直接从「立即执行」起步。所有路径都用绝对路径 / 从 repo root 出发。已建成的 pipeline 全部测过、5s + 15s 切段已落盘可直接用。

## 已建成（无需重做）

| 资产 | 位置 | 状态 |
|---|---|---|
| MimicKit submodule | `dependencies/MimicKit` @ 3ab064b9b2b (v0.91) | ✅ |
| Isaac Lab v2.3 API patch | `patches/mimickit/isaaclab-v23-api.patch`，driver 自动 apply | ✅ |
| Preview driver | `scripts/mimickit_preview.sh`（10 profile，含 lafan_fight） | ✅ smoke test 过 |
| npz→pkl 转换器 + 切段 | `scripts/lafan_g1_npz_to_mimickit.py`（支持 `--start_frame/--end_frame`） | ✅ |
| **5s warm-up 切段** | `dependencies/MimicKit/data/motions/g1/lafan_fight_5s.pkl` (frames 0–150, 47 KB) | ✅ 已生成 |
| **15s 主训练切段** | `dependencies/MimicKit/data/motions/g1/lafan_fight_15s.pkl` (frames 600–1050, 141 KB) | ✅ 已生成 |
| ipynb | `MimicKit.ipynb`（31 cells） | ✅ |
| conda env | `isaaclab`（Isaac Lab v2.3.0, IsaacSim 5.1, torch 2.7+cu128） | ✅ |

## 立即执行（compact 恢复后第一步）

### Step 1：5s pipeline 信心确认（30 min 训练）

```bash
cd /home/david/work/isaaclab-experience/dependencies/MimicKit
mkdir -p output/preview_envs output/train_lafan_fight_5s
# 临时 env yaml，把 motion_file swap 到 5s 切段
sed 's|data/motions/g1/g1_walk.pkl|data/motions/g1/lafan_fight_5s.pkl|' \
  data/envs/deepmimic_g1_env.yaml > output/preview_envs/train_lafan_fight_5s_env.yaml
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
nohup python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config output/preview_envs/train_lafan_fight_5s_env.yaml \
  --agent_config data/agents/deepmimic_g1_ppo_agent.yaml \
  --visualize false --out_dir output/train_lafan_fight_5s \
  --logger txt --save_int_models \
  > output/train_lafan_fight_5s/run.log 2>&1 &
```

监视：`tail -F output/train_lafan_fight_5s/log.txt`，关注 `mean_episode_reward` ↑ + `mean_episode_length` 接近 `episode_length: 10.0` (env yaml 设的)。Isaac Lab kit 启动 30–60s，然后 ~30K env-steps/sec。

**判定能 GUI eval 的指标：** `mean_episode_length ≥ 5s`（一半 episode_length）AND `mean_episode_reward > 50% 的 max possible`。预计 **30 min 看到第一个 ckpt** 起色，**1–2h ship-quality**。

### Step 2：GUI eval（出动作就启）

```bash
# 找最新 ckpt
LATEST=$(ls -t output/train_lafan_fight_5s/model_*.pt | head -1)
python mimickit/run.py --mode test --num_envs 4 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config output/preview_envs/train_lafan_fight_5s_env.yaml \
  --agent_config data/agents/deepmimic_g1_ppo_agent.yaml \
  --visualize true --model_file "$LATEST" \
  --out_dir output/eval_lafan_fight_5s --logger txt
```

蓝色 = 学到的 policy，绿色 = mocap 老师（ref_char_offset 偏 2m）。用户能直接看 G1 重现 5s 拳击。

### Step 3（可选，5s 出来后）：升 15s 主训练

把 Step 1 命令里 `lafan_fight_5s` 全部替换为 `lafan_fight_15s`，预期 4–6h ship-quality。

## 已知雷点

- `episode_length: 10.0` 是默认值，5s 切段时应改为 5.0；不改也能跑（policy 学循环），但 reward 上限被截
- `pose_termination_dist: 1.0` 默认 walk 调过，拳击大幅动作可能频繁 terminate。若 `mean_episode_length < 1s` 卡死 → 改 1.5 or 2.0
- LAFAN fight 含**高踢腿**和**下潜闪躲**，要照 cartwheel 思路把 `wrist_yaw_link` 加 `contact_bodies` 容忍，参考 env yaml 里那行注释
- num_envs 4096 在 4090 24G 实测稳，6000 会 OOM
- 训练日志 `log.txt` 而非 stdout — 看 stdout 是 Isaac Lab kit warning 噪音；看 `log.txt` 是 mean reward / mean ep len

## 关联记忆

- [[lafan-g1-ecosystem]] — 哪里能找到现成 ckpt（结论：只有 TWIST 通用 tracker，不兼容 MimicKit）
- [[feedback-gpu-util-as-efficiency-anchor]] — 训练吞吐怎么看
- [[auto-eval-watcher-standard]] — 自动 eval watcher 范式
- [[feedback-incremental-eval-during-training]] — 长训练每 1/10 步 quick eval

## 兜底

如果 30 min 后 `mean_episode_reward` 还在初始 baseline，说明 reward 调不对。降级方案：
1. 改用 g1_spinkick.pkl 当 motion_file 重训（已知 MimicKit 发的 spinkick ckpt 是这套 hyperparam 训出来的，肯定收敛）
2. 收敛后说明 pipeline 通，然后改回 fight_5s 调 reward
