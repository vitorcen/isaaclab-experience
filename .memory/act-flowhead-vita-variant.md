---
name: act-flowhead-vita-variant
description: 战略转向——flow 主线从 DP 骨架(E1 高清峰值仅~13%)挪到 ACT 骨架(43.3% 最强 baseline);ACT-FlowHead=VITA-lite,保留 ACT transformer enc-dec+ResNet,只把 CVAE 头换成 rectified-flow(decoder 当速度场,query=噪声 chunk x_τ+时间嵌入,Euler N 步采样);代码在 LeIsaac/FlowHeads(原 FlowDP 重命名为伞仓)的 flowact/ 包,已写+静态校验过(import/注册/继承),完整训练链未跑;两轮 codex+mimo 脑暴一致背书
metadata:
  type: project
---

# ACT-FlowHead（VITA-lite）= flow 实验主线（2026-06-13）

承接 [[flowdp-e1-e2-live-state]] 与 [[flow-matching-policy-survey]]。**战略转向**:把 flow 实验从
**DP conv-UNet 骨架**(弱 anchor)挪到 **ACT transformer 骨架**(强 anchor)。

## 为什么转向（关键判断）
- **E1 实测**:DP 高清(480×640,从零训 6ep)10 个 ckpt quick-eval **峰值仅 2/15=13.3%**(均值~5%),
  和原低清 DP baseline 8.3% 同档(噪声内)。**证伪"DP 8.3% 是纯视觉分辨率瓶颈"**——更像小数据(60 demo)+
  DP 从零 6ep 共同限制。**DP 骨架是输家,在其上换 flow 头=弱地基盖楼。**
  (注:首轮 eval 报"全 0%"是 `retracted_middle` 检测器假阳性,早期 DP ckpt 停 rest pose 被误判提前砍轮;
  `LEISAAC_DISABLE_RETRACT_DETECT=1` 重跑才得真曲线 13.3%。见 [[flowdp-e1-e2-live-state]]。)
- **ACT 在同一 60-demo 上 43.3%(最强)** → "transformer 撑不起小数据"在本任务**被 ACT 自身证伪**。
  ACT 能跑通靠 **CVAE 的 KL 正则**压 transformer。所以真问题不是"能不能用 transformer",
  而是 **"flow 头替掉 CVAE(去 KL 正则)在 60-demo 会不会过拟合"** —— 这正是 **VITA** 干的(ACT 骨架不动,CVAE→flow)。
- 锚定赢家才有机会**超过 ACT 而非打平 8%**。

## 两轮独立脑暴一致结论（codex gpt-5.5 + mimo-v2.5-pro，tmux）
- **同意**把 flow 主线挪到 ACT 骨架,VITA 式 ACT-flow 作主线,取代/优先于 conv-UNet flow。
- **过拟合 中等可控**:视觉 latent 当源(隐式 KL,VITA 核心)+ EMA + ACT 自带 dropout + early-stop(留 5 条 val);
  轻 KL(λ≈1e-4)只作 ablation。action chunk 维度小(chunk×6)本就比高维动作安全。
- **预期 30-50%,有机会超 43.3% 但不保证**;realistic 目标=打平 ACT+降方差;崩了可能掉 20-35%。
- **推理 Euler 4~6 步**(主推 5)守 <150ms:encoder 跑一次,只 decoder 迭代;ACT decoder 单次~10-15ms,flow 多个 t_emb 加法,每步≈不变。
- 分歧:codex 推 visual-source(改 x0),mimo 推 MeanFlow(改 loss)——都作 **E4 ablation**,E2′ 先用最干净的 Gaussian-source。

## 实现（已写，静态校验过；完整训练链未跑）
**LeIsaac/FlowHeads**(原 `FlowDP` 重命名为**伞仓**,内含两变体;inner commit `6b5433a` 未 push;
父仓 LeIsaac gitlink+`.gitmodules` 已 stage 未 commit;远端要新建 `git@github.com:vitorcen/FlowHeads.git`):
- `flowdp/` = **DP-FlowHead**(原样,policy type `flowdp`,DP 骨架,降级为对照,并行另跑)。
- `flowact/` = **ACT-FlowHead**(新,policy type `flowact`):
  - `FlowACTConfig(ACTConfig)`:`use_vae=False`(关 CVAE/KL)+`num_inference_steps=5`(Euler NFE)+`flow_source="gaussian"`。
  - `FlowACT(ACT)`:**只新增 2 个模块**——`flow_action_input_proj`(action_dim→dim_model)+`flow_time_mlp`(时间嵌入);
    拆 `encode()`(backbone+transformer encoder,latent token 恒 0=与 ACT 同布局,跑一次缓存)+`velocity()`(**decoder 当速度场**:query=`proj(x_τ)`+t_emb,cross-attend encoder,`action_head` 出速度);
    `forward`=rectified-flow loss(`x_τ=(1-τ)x0+τx1`,target `x1-x0`,τ~U(0,1));`generate`=Euler N 步。
  - `FlowACTPolicy(ACTPolicy)`:覆写 `forward`(masked MSE 速度,pad mask 同 ACT)+`predict_action_chunk`(调 generate)。
- **lerobot 机制坑**:`make_pre_post_processors` 按 `isinstance(cfg, ACTConfig)` 路由 → `FlowACTConfig` 子类**自动复用 ACT processors,无需 patch**(processor_flowact.py 仅 0.5.x 命名约定用,0.4.4 冗余);
  但 `get_policy_class` 无 registry 回退 → **必须 train.py monkeypatch**(`make_policy_config` 有 registry 回退,config 能解析)。
- **0.4.x vs 0.5.x import 适配**(已做 try/except):`PreTrainedConfig` 在 0.4.x=`lerobot.configs.policies`(0.5.x=`lerobot.configs`);
  train 入口 0.4.x=`lerobot.scripts.lerobot_train`(0.5.x=`lerobot.scripts.train`)。
- 启动:`FlowHeads/scripts/train_flowact.sh`(ACT 原生分辨率,无 resize_shape;`pip install -e . --no-deps`)。
- **静态校验已过**:py_compile + 包 import + `flowact` 注册进 PreTrainedConfig registry + 继承链 + 时间嵌入形状。
  **未跑**:完整 forward/训练(encode/velocity/decoder 张量形状的运行时验证=smoke pending)。

## 下一步（待 user 说跑）
1. **smoke**:`flowact.train` 跑 2-3 步验 forward/loss/save 不崩(本机 lerobot-v044 或对齐 ACT baseline 的 0.4.0)。
2. **训练预算对齐 ACT 43.3% baseline 的步数/chunk**(不是套 DP 的;ACT 从零收敛比 DP 快但仍非 6ep VLA-微调那套)——起训前查 baseline config。
3. E2′ 全量 → quick-eval 曲线 → 与 ACT/DP-flow 三方对比 → 赢家排 20-round strict。
4. E3 Euler 步 sweep 1/2/4/8;E4 visual-source / MeanFlow ablation(仅 E2′ 过线后)。

关联:[[flowdp-e1-e2-live-state]]、[[flow-matching-policy-survey]]、[[act-framework-drift-root-cause]]、[[feedback-incremental-eval-during-training]]、[[feedback-mimo-independent-review]]
