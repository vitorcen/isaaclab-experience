---
name: mimickit-to-vla-distill-plan
description: MimicKit DeepMimic PPO → prompt-conditioned 人形 VLA 蒸馏路径的方向决策 — 基建优先于 expert，训多不训长，prompt 多样化是最便宜杠杆。设计文档 doc/mimickit_to_vla_dataset.html
metadata:
  type: project
---

把已 ship 的 4 个 MimicKit DeepMimic G1 (29-DoF) motion-tracking 策略 (fight/run/dance/jumps，15s 切段)
蒸馏成一个 **prompt-conditioned 人形 VLA**（仿真里输入 "fight"/"dance" → 输出对应 motion）。
完整设计 + 多模型调研在 `doc/mimickit_to_vla_dataset.html`（§1-6 四模型并行；§7 第二轮 Claude×GPT-5.4 决策评审）。

**当前状态（2026-06-02）：仍是规划，未启动。** 4 个 15s expert ckpt 已 ship（见 [[mimickit-lafan-fight-training-plan]]）。

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
