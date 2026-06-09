---
name: mimickit-lafan-fight-training-plan
description: 🏆 SHIPPED 2026-06-02 — MimicKit DeepMimic G1 × 4 LAFAN motion (fight/run/dance/jumps) 15s 切段，4 h pipeline，3/4 触顶 ≥98%；USD 材质修复 + HTML 文档 doc/mimickit_lafan_training.html
metadata:
  type: project
---

## 🏆 4-motion SHIPPED (2026-06-02 ~04:00)

15 s 中段切段、scratch 训练、每 motion max 1500 iter，串行队列总 ~4 h，4090 24 G。

| Motion | 训时 | Final Return | Final Len/300 | Coverage |
|---|---|---|---|---|
| fight_15s | 84 min | 224.1 | 297.6 | **99 %** ✅ |
| run_15s | 56 min | 155.0 | 189.5 | 63 % ⚠️ plateau |
| dance_15s | 50 min | 221.4 | 294.6 | **98 %** ✅ |
| jumps_15s | 51 min | 203.4 | 294.2 | **98 %** ✅ |

run plateau 原因：含连续 sprint + 蹦跳，single-task DeepMimic PPO (SGD lr=1e-4, action_std=0.05) 探索能力受限。下一步候选：换 ADD framework / motion curriculum / init from fight_15s ckpt sequence transfer。

**Ship ckpts**: `dependencies/MimicKit/output/train_lafan_<motion>_15s/int_models/model_0000001500.pt` × 4

**USD 材质修复**: 详见 [[mimickit-g1-usd-material-fix]]。MJCF per-geom rgba 在 g1.usd 里被压成单一 white material → `scripts/g1_usd_recolor.py` 反推 + 重绑 → `g1_textured.usd` → `MIMICKIT_G1_USD` env var 切入。

**HTML 文档**: `doc/mimickit_lafan_training.html`（中英对照 + 内嵌 SVG 训练曲线 + debug journey）。



## 🏆 Result (2026-06-01 22:18)

**Ship 在 iter 1100，比 plan 预估 1-2h 快 5-10×**。4090 24G，num_envs=4096，DeepMimic PPO，default reward shape，episode_length=10s（motion=5s WRAP loop 2 次）。

| iter | Test_Return | Test_Ep_Length (steps) | 秒 | 节点 |
|---|---|---|---|---|
| 0 | 12.9 | 29.0 | 0.97s | random init |
| 100 | 23.5 | 43.6 | 1.45s | |
| 400 | 55.6 | 75.5 | 2.52s | inflection |
| 500 | 69.9 | 95.3 | 3.18s | **首次 GUI eval（与训练并行）** |
| 600 | 104.9 | 138.6 | 4.62s | 跟完 motion 92% |
| 700 | 126.9 | 166.1 | 5.54s | 超 motion 边界 |
| 800 | 233.7 | 297.0 | 9.90s | 接近 episode ceiling |
| 1000 | 253.6 | 300.0 | 10.0s | 满 ceiling |
| **1100** | **258.6** | **300.0** | **10.0s** | **plateau，stop** |

**关键发现：**
- 5s motion 训练比预想容易：iter 400 inflection、iter 600 ship、iter 800 episode 满 ceiling
- GUI eval 与训练并行不打架（4090 24G 共享 13.9G/99% util，训练 throughput 0.55→0.33 iter/s 还可接受）
- `--save_int_models` MimicKit arg_parser 要 `true/false` 不能 bare flag（踩坑）
- log.txt 是单行无 newline 20-col 滚动格式（踩坑），用 Python `data[-c:]` parser 取最后行
- ckpt 落 `int_models/model_XXXXXXXXXX.pt`，顶层 `model.pt` 是最新 pointer

**最终 ckpt:** `dependencies/MimicKit/output/train_lafan_fight_5s/int_models/model_0000001100.pt`

**Replay 命令（任何时候）：**
```bash
cd ~/work/isaaclab-experience/dependencies/MimicKit
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
python mimickit/run.py --mode test --num_envs 4 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config output/preview_envs/train_lafan_fight_5s_env.yaml \
  --agent_config data/agents/deepmimic_g1_ppo_agent.yaml \
  --visualize true \
  --model_file output/train_lafan_fight_5s/int_models/model_0000001100.pt \
  --out_dir output/eval_lafan_fight_5s_ship --logger txt
```

**下一步可选：** 用同套 pipeline 训 15s 切段（已生成 `lafan_fight_15s.pkl`），预计 1-2h ship。Step 3 命令在下文「立即执行」段。

---

## 🔁 In-flight 扩训 (2026-06-01 23:54)

用户反馈 5s ckpt 动作"不够自然"+"还需要训练更多动作" — 决策（详见 [[lafan-g1-ecosystem]]）：
- 自然度根因 = 5s 切段太短，loop boundary 不连续 → 训 15s 中段
- 4 个 LAFAN motion 全铺（fight / run / dance / jumps）

**可复用脚手架（已落 scripts/）：**
- `scripts/mimickit_train_one.sh <motion> [max_iters] [init_ckpt]` — 单 motion 训练 + iter 上限自动 SIGTERM
- `scripts/mimickit_train_queue.sh [max_iters]` — 串行训 4 × 15s（fight → run → dance → jumps，按难度排序）
- `scripts/mimickit_train_pipeline.sh` — 等当前 fight_5s_resume 到 iter 2500 → SIGTERM → kick queue
- 3 个新 pkl 已落地：`lafan_dance_15s.pkl`（frames 1746-2196）/ `lafan_jumps_15s.pkl`（3441-3891）/ `lafan_run_15s.pkl`（3341-3791），各 450 frames=15s

**当前 pipeline 状态：**
- Phase 1: `fight_5s_resume`（init=model_1100，目标 iter 2500，预计 ~15 min）→ `output/train_lafan_fight_5s_resume/`
- Phase 2 队列：`fight_15s → run_15s → dance_15s → jumps_15s`，每个 max 1500 iter ≈ 1-1.5h，总 4-6h
- 进度日志：`output/pipeline.log` + `output/queue_summary.log` + `output/pipeline_summary.log`

**如何接力**（compact 后或人工 resume）：
```bash
# 看进度
cat ~/work/isaaclab-experience/dependencies/MimicKit/output/pipeline_summary.log
# 单独 replay 任一 ckpt
cd ~/work/isaaclab-experience/dependencies/MimicKit
bash $REPO/scripts/mimickit_preview.sh lafan_fight_15s   # （但 preview 是 mocap 回放，不是 policy）
# 直接 test mode 看 policy：
python mimickit/run.py --mode test --num_envs 4 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config output/preview_envs/train_lafan_<MOTION>_env.yaml \
  --agent_config data/agents/deepmimic_g1_ppo_agent.yaml \
  --visualize true \
  --model_file output/train_lafan_<MOTION>/int_models/model_*.pt \
  --out_dir output/eval_<MOTION> --logger txt
```

---

**原任务：训 MimicKit DeepMimic G1 + LAFAN_fight 切段，到能做出动作启动 GUI eval 给用户看。**

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
cd ~/work/isaaclab-experience/dependencies/MimicKit
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
