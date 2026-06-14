---
name: flow-matching-policy-survey
description: 5个开源flow-matching策略选型(替代ACT/DP的非-VLM baseline);2个点云派被RGB-only淘汰,A2A最佳外部但绑RoboVerse要移植,结论=自研最小CFM head复用lerobot DP backbone只换DDPM→rectified-flow;三方评审一致最强警告=DP 8.3%是240×320缩放视觉瓶颈非head,先E1 DP高清重测再A/B
metadata:
  type: project
---

# Flow-Matching 策略选型调研（2026-06-13）

任务：为 PickOrange（SO-101 单臂，RGB-only 2cam 480×640，6-D 绝对关节角，LeRobotDataset，
Isaac 闭环，现 baseline ACT 43.3% / DP 8.3%）找一个**非-VLM** flow-matching 策略替代 ACT/DP，
或自己写。调研 5 个仓库 + 三方评审（Claude + codex/gpt-5.5 + mimo-v2.5-pro，结论一致）。

**5 仓评分（fit /5）**：
- PointFlowMatch（Freiburg CoRL'24 GPL-3.0）= **点云/深度**派 → RGB-only 淘汰，1/5
- FlowHeads（UESTC AAAI'25 MIT）= **3D 点云** DP3 派生，1-step 19.9ms consistency-FM → RGB-only 淘汰，1/5
- **A2A_Flow_Matching**（NTU 杨建飞 RSS'26 Apache-2.0）= RGB+proprio历史，"action-from-history"流源→1-step 0.56ms，
  跑过 Isaac Sim 5.0，打赢 ACT/DDPM-UNet。**最佳外部 3/5**，但**绑死 RoboVerse**要写 LeRobot 桥+serve 适配，
  ResNet-18 视觉分辨率存疑，proprio 历史结构承重（置换 bug 雷区）→ 是 port 不是 swap
- streaming-flow-policy（MIT CoRL'25 **无 LICENSE**）= 主要 state/keypoint，图像变体打不过 DP，无 license 不能 vendor，2/5
- HRI-EU flow_matching（Honda arXiv'24 BSD-3）= RGB-D+affordance 深度为主，zarr，TorchCFM 1-NFE，2/5

**结论（推荐路线）**：没有一个能 drop-in。**自己写最小 CFM head** = fork 现成 lerobot Diffusion-Policy 全套接线
（image encoder + FiLM-UNet1D + LeRobotDataset loader + serve/eval 适配器），**只把 DDPM 噪声预测换成直线
rectified-flow 速度场**（target = x1-x0，Euler 积分 1/2/4/8 步）。零新耦合、真 drop-in、与 DP 单变量对照。

**三方评审最强一致警告**：**DP 8.3% vs ACT 43.3% 的 5× 差距大概率是视觉分辨率问题非 head 问题**——
DP 把图像缩到 240×320 把橙子（10-40px）糊没了（呼应 [[vla-pickorange-vision-resolution-selection]]）。
换 FM head **不修视觉 grounding**。**实验阶梯必须先 E1=DP 高分辨率重测**，再 E2=CFM head（高清同 backbone），
否则是跟蒙眼对手比。Fair A/B：DP-lowres→DP-highres→CFM-highres(同backbone)→ACT。
π₀ 的 action head（CFM 与 VLM 解耦）是真正的设计参考，不是这 5 个学术 dump。
后续可选：ManiFlow / VFP(变分FM多模态) / RTC(real-time chunking) / consistency-FM(1步) / A2A 历史源（都留 v1）。

文档：`LeIsaac/docs/training/flow_matching_policy_survey.html`（含族谱 SVG + 实验阶梯 + 实现草图）。
关联：[[vla-pickorange-vision-resolution-selection]]、[[sonic-serve-state-permutation-bug]]、[[feedback-mimo-independent-review]]
