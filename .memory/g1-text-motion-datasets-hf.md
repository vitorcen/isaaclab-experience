---
name: g1-text-motion-datasets-hf
description: G1 text-motion 数据集 HF 调研——HumanML3D-for-G1 不存在/AMASS_Retargeted_for_G1/LeVERB 62GB 全是视频坑
metadata:
  type: reference
---

**2026-06-11 实搜 HF**（`HfApi().list_datasets`）。给 SONIC/MaskBeT 找 text→G1 motion 数据时核过。
关联 [[lafan-g1-ecosystem]] [[maskbet-route-b-submodule]] [[autodl-hf-download-speed]]。

## 关键事实
- **「HumanML3D 的 G1 版」HF 上不存在**。HumanML3D = AMASS 子集 + 文本标注，motion 就是 AMASS。
  要 G1 版只能拼：motion 侧 = AMASS retarget 到 G1，文本侧 = 原 HumanML3D 仓（按
  `<AMASS子集>/<序列>/<clip>` 路径配对）。HumanML3D/AMASS 都**无官方 HF 仓**，全是社区镜像。
- **AMASS license = MPI 非商业**；镜像可能未授权重分发，研究用别再发布/商用。

## ember-lab-berkeley（Berkeley，G1 生态最全）实测大小
| repo | 总大小 | 大头 | 备注 |
|---|---|---|---|
| `ember-lab-berkeley/AMASS_Retargeted_for_G1` | 11.9 GB | .npz 全部 | `g1/<子集>/<序列>/<clip>_120_jpos.npz`，120fps G1 关节位置，目录镜像 AMASS |
| `ember-lab-berkeley/LeVERB-Bench-Dataset` | **62.8 GB** | **.mp4 62.1 GB(98.8%)** / parquet 仅 0.66 GB | text→G1 全身 lerobot parquet，**大全在渲染视频** |
| `ember-lab-berkeley/LAFAN-G1` | 0.47 GB | .npz | |
| `fleaven/Retargeted_AMASS_for_robotics` | — | .npz | 多机器人 retarget；另有 bxi_elf2 / FourierN1 |

## 🎯 hojjunekim/g1_sonic_* = HF 上唯一第三方 SONIC token 空间数据（2026-06-11 全网扫描）
- 7 仓 `g1_sonic_orange_to_plate_{mani,locomani,turn}[_psi0|_teleop]`，schema 与
  `wsagi/SONIC-VLA-LAFAN` **逐字段同构**（`action.motion_token[64]`+`state[43]`+teleop.*）
  ——同一套 GR00T-WBC 录制管线。**~549k 去重帧（95× flow3），零 tokenize 直接可用**。
- 三 caveat：fps=30（token 离散→nearest 重采样保格，或 state 关节流插值 50fps 重过 encoder）；
  域=orange-to-plate loco-mani、文本仅 ~3 条（买 token 时序结构不买文本泛化）；**无 LICENSE**
  （训练可，发布衍生权重先联系作者）。
- 排序结论（MaskBeT→SONIC WBC 用）：①AMASS-G1 预训练主粮 ②hojjunekim 零工程即时语料
  ③LeVERB 文本泛化+唯一 Apache 干净线 ④GeorgiaTech/g1_lafan1_50hz（LAFAN G1 npz **@50Hz
  正对口免重采样**，同域收尾首选）→ flow3+BonesSeed(3815帧@50fps,7条dance) 微调。
- 另查：Berkeley-Humanoids/Lite-Motion-Tracking（264k帧 joint[74] CC-BY-NC）备选；
  unitreerobotics G1_WBT_*（家务 teleop）对 motion 先验贡献低。

## LeVERB schema 实查（2026-06-11，给 MaskBeT 预训练选型时核的）
- **action = 29 维 G1 关节角 @30fps**（连续，kinematic retarget），**不是** SONIC FSQ token——
  喂 SONIC 系模型必须先过冻结 SONIC encoder tokenize + 30→50fps 插值。
- obs 含 joint_pos(29)+body_pos(30×3)+body_quat(30×4) 全运动学，够构造 encoder 输入。
- 3696 ep / 445,944 帧 / mean 4.0s / **1069 条文本**（同动作多措辞，"stand up from the pink
  chair" 式）；任务分布偏坐立/行走/日常，**无 combat/dance 烈度**——域窄但文本配对干净。
- **license = Apache-2.0**（HF card 核查）——G1 text-motion 配对里唯一干净可再发布的。

## 坑 & 命令
- **LeVERB 大几倍 ≠ 动作多，是带了第三人称视频**。motion-token 路线（视觉弱条件）不需要那 62GB：
  `hf download ember-lab-berkeley/LeVERB-Bench-Dataset --repo-type dataset --exclude "*.mp4"`
  → 62.8GB 降到 ~0.75GB（只 parquet+jsonl）。
- 下到默认 cache（`~/.cache/huggingface/hub/`）不加 `--local-dir`；大集开
  `pip install hf_transfer` + `HF_HUB_ENABLE_HF_TRANSFER=1`（box 直下 ~64MB/s）。
- 选型：要原汁 HumanML3D text-motion 自控流程 → AMASS_Retargeted_for_G1 + humanml3d 文本；
  要开箱即用 text→G1 喂训练（最贴 SONIC/MaskBeT 口径）→ LeVERB-Bench-Dataset（排除 mp4）。
