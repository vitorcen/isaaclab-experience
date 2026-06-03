---
name: sonic-wbc-vla-route
description: 架构侧 GR00T×SONIC-WBC 路线 + 路 A 动作源（本地 deploy demo → robot_filtered）的可执行计划与评审验过的坑
metadata:
  type: project
---

**目标（架构侧，区别于数据侧 DART 蒸馏 [[mimickit-to-vla-distill-plan]]）**：GR00T N1.7 VLA 只输出
SONIC WBC 的 64 维 FSQ 潜 token，WBC（`sonic_release/last.pt`）当现成平衡底座，按 prompt 出舞蹈/武术。
设计文档：`doc/groot_sonic_wbc_route.html`（架构 Stage A/B/C）+ `doc/sonic_dance_motion_source.html`（动作源三路对比）。
2026-06-03 经 codex gpt-5.5 + mimo 两轮独立评审（见 [[feedback-mimo-independent-review]]），结论：**先走路 A**。

## 下一步 = 路 A（也是架构侧 Stage A / Milestone 0 去风险实验）
把本地 SONIC deploy demo 动作转成 Isaac-eval 的 `robot_filtered`，跑 eval 看 **WBC 跟不跟得住踢/舞而不摔**。
```bash
cd dependencies/GR00T-WholeBodyControl
# ⚠️ --fps 必须 50（eval target_fps=50），转换器默认 30；先确认 deploy CSV 源帧率，源≠50 用 --fps_source
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input gear_sonic_deploy/reference/example/neutral_kick_R_001__A543 \
    --output data/kick_robot.pkl --fps 50
python gear_sonic/eval_agent_trl.py +checkpoint=sonic_release/last.pt +headless=False ++num_envs=1 \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/kick_robot.pkl" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy"
```
启动方式同 [[gear-sonic-preview-setup]]（isaaclab env、preview.sh 同款 hydra override）。

## 评审验过的关键事实（别再踩）
- **动作源**：`gear_sonic_deploy/reference/example/` 本地已有 **13 条**（含 `_M`）：macarena/dance_in_da_party（舞）、
  neutral_kick（踢）、lunge、one_leg_jumping、squat、walking_quip_360。CSV(joint_pos/body_pos/body_quat)。
- **fps 是唯一真坑**：converter 默认 `--fps 30`，必须传 `50`；源帧率不符会动作变速变形。smpl_joints **不是**坑。
- **smpl 可 dummy**：`smpl_motion_file=dummy`（`motion_lib_base.py:252`）；converter `:301` 已输出 `smpl_joints=zeros(T,24,3)`；
  g1 encoder 只吃 `command_multi_future_*`+`motion_anchor_ori`（robot FK 派生），**根本不读 smpl_joints**（mimo 追源码证）。
- **robot_filtered schema**：`{motion_key:{root_trans_offset(T,3), pose_aa(T,30,3), dof(T,29), root_rot(T,4 xyzw), smpl_joints(T,24,3), fps}}`，joblib(zlib)。
- **转换器** `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` 输入支持 CSV 目录 / deploy PKL，输出正是上面 schema；内部 `MJ_TO_IL` 处理关节重排（建议跑一次 A/B sanity）。
- **HF `nvidia/GEAR-SONIC` 没有多动作 robot_filtered**（只 walk）；30G `bones_seed_smpl` 是 SMPL human data，不是 robot_filtered 必经前置 → **别为拿动作下 30G**。
- **降调**：deploy demo 可能精选"能跑"的 → 路 A 全过是 **go/no-go 信号**，不等于"覆盖任意武术"。

## 扩容优先级（评审一致）
路 A（13 条 demo，验架构）→ **Bones-SEED 的 G1 retargeted CSV 子集**（关键词筛 dance/kick/jump，同转换器，非 30G SMPL）→ LAFAN→robot（有物理 gap，最后）→ 全量 30G SMPL。

## 架构侧 Stage C 要点（评审给的，等路 A 过了再用）
- token 注入：`UniversalTokenModule.decode("g1_dyn",{token_flattened(64),proprioception})` 或 `forward(latent_residual_mode="pre_quantization_replace")`。in-process，免 C++。
- **GR00T 训练目标用 pre-quantization 连续 latent**（非分类、非量化后值），推理时过 SONIC 同一 FSQ quantizer 再 decode（codex+mimo 一致）。
- in-process decode 要复用 SONIC 的 obs 构造 / proprio history / 50Hz+token插值；`sonic_release` 的 `running_mean_std:false`，真正要对齐的是 obs term 布局/history/clipping。
- 14 维手部绕过 WBC（无平衡兜底），要单独处理（mimo 提）。
- 必须三曲线对照：官方 SONIC eval vs external-token playback vs direct-decode，逐帧比动作均值。

关联：[[gear-sonic-preview-setup]]、[[mimickit-to-vla-distill-plan]]（数据侧对照）、[[gr00t-multi-release-env-split]]（N1.7 env）、[[mimickit-lafan-fight-training-plan]]（LAFAN dance/fight 来源）。
