---
name: vla-pickorange-vision-resolution-selection
description: PickOrange VLA 选型铁律=vision 输入分辨率(橙子10-40px需≥448);Wall-X>StarVLA+Qwen3-VL>OpenVLA-OFT>multi_task_dit>LingBot 成功率预估 + 两份 plan 文档
metadata:
  type: project
---

LeIsaac SO-101 PickOrange(橙子在 640×480 帧里仅 10–40px)的 VLA 选型**铁律**:
**成败胜负手 = vision 输入分辨率,不是 action head 多先进。≥448 才进 work 区,固定 224 基本判死。** 8 个数据点验证:

- ✅ work 区:GR00T-N1.7 Eagle@448=**68%** · ACT ResNet 原生=43% · SmolVLA SigLIP@512=25% · DP ResNet 原生=概率0/3~3/3
- ❌ 224 死穴:π0.5 SigLIP@224=**1.7%** · OpenVLA 224=**0%**(根因见 [[pi05-pytorch-expertonly-phase15-negative]])

**2026-06-03 调研的新候选成功率预估**(同一铁律外推):

1. **Wall-X / WALL-OSS**(X Square Robot,`github.com/X-Square-Robot/wall-x`,Apache-2.0,LeRobot `--policy.type=wall_x`)= **估 40–65%,首选**。Qwen2.5-VL 动态分辨率根治 224 死穴 + **官方自带 SO-101 单臂示例**(dof_config 6 维)+ 预训练含单臂 + zero-shot Fruit Sorting 96。⚠️**唯一陷阱:默认部署配置 resolution=256,必须先调到 480+ 吃满原生帧否则退化成 π0.5**。推理官方背书单 4090 30ms;但无官方 LoRA,4090 单卡微调需实测(否则 bf16+grad-ckpt+bs=1 或多卡/云端全量)。
2. **StarVLA + Qwen3-VL-4B**(`github.com/starVLA/starVLA`,MIT,真实 ckpt `StarVLA/Qwen3-VL-4B-Instruct-Action`)= **估 30–60%,第二步**。Qwen3-VL DeepStack+3D-grounding 理论上对小物体最优(但**无量化证据**);四种 action head(PI/GR00T flow > OFT > FAST);LeRobot v3 + Bridge 单臂预训练。代价:**SO-101 在 README 是未勾选 TODO,单臂 config 要自补**。
3. **OpenVLA-OFT**(arXiv 2502.19645)= 估 15–40%。修好 action head(离散token→continuous+L1+并行解码,480ms→73ms 解 timeout)但 **vision 仍 224 没动** → 逃得出 0% 但天花板被 224 压死,打不过 GR00T。**仅"隔离验证 action head 增量"有科研价值,不是冲高分路线**。
4. **multi_task_dit (DiT)**(LeRobot 原生)= 估 5–30%,**比 DP 还弱**。实测源码 = CLIP ViT-B/16 @224 **+ 只取 CLS token**(整图压成单个 768 维全局向量,无 spatial map)。DiT head ≠ GR00T 成功来源(GR00T 靠 Eagle@448)。60 demo 落在其文档点名的 collapse 高风险区。roadmap 原"低风险 DiT"定位已修正。
5. **LingBot-VLA**(蚂蚁,Qwen2.5-VL-3B+flow)= 高成本/低把握,**仅双臂无单臂 ckpt,零先例**。等官方单臂 ckpt,别现在做。

**Qwen3-VL/3.5-VL 代 VLA 领域(2026-06)已不空白**但无 SO-101 开箱路径:StarVLA(可操作)、ABot-M0(Qwen3-VL-4B+DiT 单臂)、Qwen-VLA(Qwen3.5-4B,权重待证实)。

文档:`LeIsaac/docs/training/wallx_finetune_plan.html` + `starvla_qwen3vl_finetune_plan.html` + `training_roadmap.html`§3.5 成功率预估总表。
