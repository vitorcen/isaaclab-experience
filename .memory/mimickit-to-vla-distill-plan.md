---
name: mimickit-to-vla-distill-plan
description: MimicKit DeepMimic PPO → prompt-conditioned 人形 VLA 蒸馏路径的方向决策 — 基建优先于 expert，训多不训长，prompt 多样化是最便宜杠杆。设计文档 doc/mimickit_to_vla_dataset.html
metadata:
  type: project
---

把已 ship 的 4 个 MimicKit DeepMimic G1 (29-DoF) motion-tracking 策略 (fight/run/dance/jumps，15s 切段)
蒸馏成一个 **prompt-conditioned 人形 VLA**（仿真里输入 "fight"/"dance" → 输出对应 motion）。
完整设计 + 多模型调研在 `doc/mimickit_to_vla_dataset.html`（§1-6 四模型并行；§7 第二轮 Claude×GPT-5.4 决策评审）。

**当前状态（2026-06-03）：基建落地 + ACT sanity GO。** 3 块基建已建（`rollout_record.py` +
`rollout_common.py` + `rollout_record_tiled.py`[scale待验] + `mimickit_episodes_to_lerobot.py`）。
sanity 数据集 `datasets/g1-lafan-vla-sanity-img`（fight+dance 各 10ep，image 格式）。
ACT 10k 步训成（lerobot 0.5.2，chunk_size=16）：train loss→0.149，任务分离比 3.84（无 mode collapse）。
**闭环 eval 已跑（env-split：ACT server[lerobot] × MimicKit client[isaaclab]，socket+portable-序列化）。**

## 🔴🔴 数据侧蒸馏彻底负面（2026-06-03）：纯离线 BC 学不出反应式平衡（四重证据闭环）
**三组干预 + 一个 harness 验证，闭环 mean 存活帧（NACT=1，各 5ep/motion，teacher 基线=300帧满段）**：
| 实验 | 给 ACT 什么 | 存活 | 排除 |
|---|---|---|---|
| baseline | 基础 proprio + 2维 phase | 21.0 | — |
| **DART** σ=0.10/50ep+10clean | +6×数据+恢复态(120ep/35455帧) | 23.3 | ❌ covariate shift |
| **ref-target** 95→124维 | +teacher 特权参考关节目标(tar_obs信息) | 20.5 | ❌ 信息不足 |
| **teacher-replay** | teacher 真动作喂进 eval harness | **300** | ❌ harness bug |
- **真根因 = 纯离线 BC（任何变体）无法从 vision+proprio 复现 teacher 的高频反应式平衡控制**。不是数据/覆盖/信息问题，是 obs→29DoF关节目标 这个映射本身学不出反应式平衡。给"标准答案"(ref-target)都救不了，~20帧雷打不动。
- **harness 已铁证忠实**：teacher-replay（`rollout_act_eval.py --teacher_model`，用 agent._decide_action）在完全相同 env/终止/stepping 下活满 300 帧 = 和 recorder 一致 → ~20帧墙是 ACT 自身动作质量，非假象。open-loop MSE 0.33 vs std 0.52 早预警。
- 印证文献：公开 humanoid 全身**无一靠纯离线 BC**（DeepMimic RSI/OmniH2O DAgger/PHC·PULSE latent+RL）；mimo 审查也判 "29-DoF 平衡纯离线 BC 在能work的硬边缘"。
**→ 架构侧 WBC pivot 被验证为正确方向（正确理由）**：不是缺数据/信息，是平衡控制函数本身离线 BC 学不出。GEAR-SONIC WBC **已自带**反应式平衡控制器，VLA 只需出 latent 去 steer 它，根本不复现精细平衡。用户已在跑 GEAR-SONIC 预览（见 [[gear-sonic-preview-setup]]）。
**新基建（可复用，已落地）**：`rollout_record.py`/`rollout_act_eval.py` 统一复用 `rollout_common.compute_state`（消除重复=根除 train/serve skew）；`--include_ref_target`（reference 关节角，**从 motion_lib 现算，不读 `_ref_dof_pos` 缓存**——headless 下该缓存全零的坑）；`--teacher_model`（teacher-replay harness 验证）；脚手架 `mimickit_vla_{dart,reftarget}.sh` + `mimickit_vla_act_eval_compare.sh`（REF_TARGET/NACT env）。坑：dart 脚本曾误用 num_workers=0（image 数据集该用 4，否则 1.4 step/s 超时）。

## 📍 数据侧 DART 实验过程（2026-06-03，已完成见上方终判）
**recorder 已加 DART**：`rollout_record.py` 新增 `--action_noise σ`，执行 `a_exec=teacher+N(0,σ)` 喂 env，**label 仍存 clean teacher action**。
**mimo 独立审查（/tmp/mimo_dart_clean.txt）改了我两处选择**：
- **prev_action 改记 executed 而非 clean**（已改）：student 部署时 prev_action = 它自己(带误差)的输出=实际驱动动力学的量，因果一致性才是部署成立的关系。`prev_action = action_exec.clone()`。clean rollout(σ=0)下 executed==clean 不变，现有 10ep clean 数据仍可直接合并（同 95 维）。
- **σ 上调**：mimo 说 0.02-0.05 太弱(覆盖不到离轨态)，推荐 0.10、teacher 摔回退 0.07。**用探针裁决**：σ=0.05 探针 fight 3/3 跑满 300 帧(teacher 轻松存活=有余量上探)；σ=0.10 探针进行中。规则=取存活率≥80% 的最大 σ。
- mimo 其余：isotropic 噪声够用(别 per-joint)；clean+DART 混合别 DART-only；目标 50-100 DART/motion(少则升 σ 补)；DART-first 对(DAgger 要在线跑 student 更重)；go=DART 后存活≥50 则继续 DAgger，<5 帧改善则查 ACT 容量/action 表示/clean 数据质量。
**驱动脚本**：`scripts/mimickit_vla_dart.sh`（SIGMA/NUM_DART/STEPS/STAGE 可调，record→merge clean+DART→train ACT chunk_size=16）。数据集 `datasets/g1-lafan-vla-dart-img`，ckpt `outputs/act_g1_lafan_dart`。闭环判定：mean 帧数 20→≥50 则 DART 有效继续 DAgger 到 300。
**clean baseline 事实**：fight/dance 各 10ep，全 300 帧(dance 有 1 个 32 帧早摔)；现有 baseline ckpt = chunk_size=16/n_action_steps=16/n_obs_steps=1/dim_model=512。
**另一条路（用户暂缓）架构侧 WBC+finetune**：`UNITREE-G1.ipynb` + `LeSONIC/scripts/gear_sonic_{setup,preview}.sh` + `groot_n17_download.sh`，SONIC-encoder→latent label→finetune GR00T-N1.7。
**复用环境/坑全在 [[lerobot-v040-convert-segfault-fix]]**：视频 decode 必崩→image 数据集+num_workers=0；ACT 训练用 lerobot 0.5.2 不是 v040；闭环 recorder teardown 用 os._exit(0)；进程按 output_dir 名 pkill。

## 🔴 闭环 go/pivot 终判（2026-06-03）：pipeline GO，naive-BC 平衡 PIVOT
- **基建全通**：record→convert→train→闭环 eval 端到端首次跑通（§7.2 的 #1 未知"链通不通"已验证 = 通）。
- **任务条件化通**：open-loop 分离比 3.84，fight/dance 动作截然不同，无 average-motion。
- **动态平衡失败**：闭环里 ACT 驱动 G1 **15-24 帧（0.5-0.8s）必摔**（done=FAIL），teacher 同 RSI 起点跑满 300 帧。
- **关键对照**：`n_action_steps=16`(开环0.53s) vs `n_action_steps=1`(每步重规划) **几乎无差**（仍 15-24 帧摔）
  → 瓶颈**不是 open-loop horizon，是控制精度/covariate-shift**。BC 单靠离线 demo 进不了 teacher 没访问过的恢复态。
- **诊断**：open-loop per-frame MSE≈action 方差（0.33 vs std0.52）本就预警精度不足；闭环坐实。design 风险#2 命中。
- **下一步方向**（三模型调研定，2026-06-03 Opus×GPT-5.5×MiMo，doc/vla_data_expansion_covariate_shift.html）：
  **两条治本路互补**。① 数据侧（便宜、沿用现有 ACT）：**DART(录制时给 teacher 动作注入噪声 σ=0.02-0.05rad，label 存 clean teacher action) → 必须先 DART 再 DAgger** → RSI+push+dynamics-DR(非 visual)。半天出 go/pivot：mean 帧数 20→≥50 则 DART 有效继续 DAgger 到 300。② 架构侧（天花板高、架设重）：**GEAR-SONIC WBC 当平衡底座**。
  铁律：公开 humanoid 全身**无一靠纯离线 BC**（DeepMimic RSI / OmniH2O DAgger / HumanPlus teleop / PHC·PULSE latent+RL）。
- **🏗️ 架构侧 = WBC + finetune（用户主推，2026-06-03 核实）**：`nvidia/GEAR-SONIC` 是 G1 全身控制器(WBC，会 walk/run/jump，encoder/decoder/planner ONNX，可 MuJoCo 键盘预览)。`UNITREE_G1`/`UNITREE_G1_SONIC` 是 GR00T 的**动作空间 tag(posttrain)不是模型**；**无现成微调好的 G1 VLA**，必须从 `nvidia/GR00T-N1.7-3B` base 自己 finetune。路线：SONIC encoder 把 motion 转 latent 当 label → 建 UNITREE_G1_SONIC LeRobot 数据集 → finetune GR00T → VLA 出 latent→WBC decode→不摔。脚手架：`UNITREE-G1.ipynb` + `LeSONIC/scripts/gear_sonic_{setup,preview}.sh` + `groot_n17_download.sh`（GEAR-SONIC 安装重：TensorRT+C++ build+.venv_sim，首次需验证）。相机取景差（G1 占画面极小）也要修。
- 脚手架：`scripts/act_policy_server.py` + `rollout_act_eval.py` + `mimickit_vla_act_closedloop.sh`（NACT=n_action_steps override）。
踩坑全记在 [[lerobot-v040-convert-segfault-fix]]。

**早先状态（2026-06-02）：仍是规划。** 4 个 15s expert ckpt 已 ship（见 [[mimickit-lafan-fight-training-plan]]）。

## 核心 pipeline
挂第三人称 TiledCamera 录 RGB + per-episode language prompt → 转 LeRobot v2 + `meta/modality.json`
→ ACT/DP 先验证 → GR00T N1.7 / π0.5 / SmolVLA 作 phase2 对照。导出 **30Hz**（不是 60Hz），15s×30Hz=450 frame/ep。
规模分阶段 **50 → 200 → 500 ep/motion**。

## 第二轮评审定的执行优先级（§7，关键）
1. **基建优先于 expert**：先补 3 块缺失件 —— ① G1 locomotion rollout recorder (~200 行) ② 29-DoF `modality.json` ③ 第三人称 camera mount。仓库已有 80% 基建（`LeIsaac/scripts/convert/isaaclab2lerobot.py` v2/v3、`MimicKit/.../isaac_lab_recorder.py`、`LeIsaac/scripts/training/lerobot_finetune.sh`），只缺这 3 块。
2. **激进 sanity**：10 ep × 2 motion（`fight + dance`，prompt 区分度最强）→ GR00T/ACT 短 finetune → closed-loop qualitative check → go/pivot。
3. 通了才加 dance expert：第 2 个必须 **dance2（不同编舞）**，第 3 个 dance1/dance2 **不同 phrase**。
4. student 语义混淆 / average-motion → 再扩 **4-6 clips**。

**为什么基建先行**：最大未知数不是 teacher 不够多，是 recorder/modality/camera/phase 接法/student 能不能学这条链通不通。管线没验证就铺 expert = 把错误管线喂得更饱。expert 1h/个不贵，真正贵的是 P2 录制 + P4 finetune。

## 已识别的高风险（doc §4）
- 🔴 **SigLIP@224 视觉瓶颈**（参考 [[pi05-pytorch-expertonly-phase15-negative]]）：单帧难分 "fight" vs "dance"。缓解：vision 只作 body-style 参考，proprio history 驱动 dynamics，prompt+phase 定 identity/timing。
- 🔴 **dynamic balance reproducibility**：DeepMimic 60Hz 闭环学的精细平衡，VLA 10-30Hz + flow-matching 噪声极可能 takeoff/landing 摔。缓解：action chunking + PD 内插到 60Hz + success rate 而非 trajectory match 评估。
- 🟡 **state-based → visual-based 分布失配**：teacher 吃 ref motion phase，student 没有。`observation.state` 必须含 `phase_sin/phase_cos`，否则 VLA 没法从 RGB 重建 timing。
- 公开文献无完全对应工作（最近的 HumanoidGen 是 manipulation）→ 开拓性，做好 negative result 准备。

关联：
- [[mimickit-lafan-fight-training-plan]] — teacher 来源（4 个 15s expert）
- [[vla-distill-data-diversity-roi]] — 数据多样性 ROI 排序（这次评审的可复用方法论）
- [[pi05-pytorch-expertonly-phase15-negative]] — SigLIP@224 瓶颈的前车之鉴
