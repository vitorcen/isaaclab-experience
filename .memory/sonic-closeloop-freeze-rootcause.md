---
name: sonic-closeloop-freeze-rootcause
description: SONIC闭环kick/jump/walk站立不动根因=reset proprio OOD→冻结chunk(非动量/重力/模型/prompt);鲁棒重训方案Phase0先零重训(训练前必读)
metadata:
  type: project
---

# 🧊 SONIC闭环 kick/jump/walk 站立不动 = 根因 + 鲁棒重训方案(2026-06-24,训练前必读)

**现象**:GR00T-N1.7-SONIC头(`gear_sonic_live_demo.sh`,checkpoint-8000)闭环里 **kick/jump/walk 站立不动**,squat/dance/lunge/macarena 正常。开环 token-MSE 极好(0.0011)。

## 根因(多轮单变量受控实测,逐一证伪)
- ❌ **动量/速度**:帧间关节速度两组相同(WORK 4.98 vs FAIL 5.56),dance 最动态还 work。
- ❌ **模型能力**:喂 GT 观测 pred[0]≈GT[0]、40步 chunk drift 2.81(演化)。开环 dump npz = stride-40 的 40-chunk 预测。
- ❌ **重力 OOD**:`VLA_ZERO_GRAVITY=1` 置零无改善(已证伪);`projected_gravity` 数据恒零是真 bug 但**良性**(训练常数→头学成忽略)。
- ❌ **图像/渲染**:固定 GT 图、只把 proprio GT→Isaac → drift 2.81→0.21 翻平 → 图像非因。
- ❌ **单关节组 bug**:逐组消融——换任一单组到 Isaac 都不冻、留任一单组 GT 都救不回 → **0.54rad 全关节累加致命,无单点 bug**(腿/腰系统性偏移、臂/手 per-motion;`_fit` 不重排但组没串)。
- ✅ **prompt 有效**:prompt×proprio 矩阵——kick姿+"squat"→squat(3.0)、squat姿+"kick"→kick(2.8) → **prompt 选动作,非"按起始姿续演"**。
- ✅ **真因 = 单帧无历史策略对脆弱动作"训练轨迹细管子"零容差**:部署 reset proprio 整体偏 ~0.54rad(分散)+ 闭环 WBC 跟踪持续漂移 → 累计 OOD → 模型吐"保持当前姿"冻结 chunk(drift 0.31 vs GT 2.81)→ WBC 保持 → 站立不动点(无记忆每帧重判又冻)。脆弱 = 数据少(kick 165帧)+ 平衡/接触敏感(细管子);粗管子(squat/dance)容差大照常。
- 因果闭合证:首帧喂 GT proprio → drift 0.31→2.76、dpose 0.54→3.12 **真踢**;但**只修首帧→第2 chunk 用 Isaac proprio 又冻**(第二层 = 闭环持续漂移)。观测=43关节角+恒零重力,**无 base 位姿/速度/历史**。

## 鲁棒重训方案(`LeSONIC/doc/sonic_robustness_retrain.html`,codex gpt-5.5 + glm-5.2 双评审版)
- **Phase 0 先零重训(两评审一致最高优先,多半到这就 close ticket)**:①**WBC 隔离 ablation**——冻策略喂 GT token 看 WBC 跟不跟得住(跟不住=重训策略白搭,改 WBC/reset);②**复位锁回训练标称姿**(把 0.54 压到 <0.1)+ 扫 bootstrap 长度/execution horizon/RTC。
- **Phase 1 才重训**:**输入 proprio 噪声增广**(从实测 reset/WBC-drift 协方差采,非独立均匀±0.15)+ **LoRA 冻骨干**(rank 8–16)+ **零初始化 history residual adapter**(`delta_indices=[-1,0]`,-1帧走小MLP末层置0→step-0≡旧ckpt,**非插值**)+ history-dropout。
- **删 RSI 切片**(扩"沿轨迹相位"=错维度,0.54 在正交方向);**删 DART/DAgger**(off-tube 无 ground-truth token、恢复≠重放、WBC 在 action 空间不可反解=不可行)。velocity/扩数据 = P3。
- **判据换掉 `drift>1.5`**(摔=高 drift 假阳性、无方向)→ **任务级闭环判据**(kick 脚高/jump 质心升/walk 位移)+ 轨迹方向对齐(DTW/PCA),按扰动桶(标称/轻/重)分级、每动作 ≥100;**kick/walk basin 可能物理不可达,别强求 80%**。
- **删开环 MSE 护栏**(0.0011 时闭环已败=无意义,闭环 basin 是唯一验收门);有 **ablation 计划**(Phase0 / +噪声LoRA / +history 逐项加)。
- 风险:拆掉冻结=学到的安全 fallback → 可能自信错动作摔;prompt 选择性随 proprio 方差↑退化。

## 诊断脚手架(可复用,本机 4090,headless 加 DISPLAY=:0 渲染相机)
- injector 自带 `query #N: dpose/tok_std/root-z` 日志(dpose~0=冻、root z 0.79=站立)。
- prompt×proprio 矩阵 + 逐组消融:`Gr00tPolicy` 直接 load + `LeRobotEpisodeLoader`(dp.states/dp.images/dp.text),`parse_observation_gr00t`→`policy.get_action`→`parse_action_gr00t`。
- injector env 开关:`VLA_GT_PROPRIO_NPZ`(首帧覆盖 proprio,诊断)/`VLA_ZERO_GRAVITY`(已证无效)。
- ⚠️ headless eval 退出常卡 teardown → 按 PID 杀 `eval_agent_trl.py`+`run_gr00t_server`(pgrep 自匹配假阳性,见 [[feedback-shared-gpu-eval-queue-orphan-discipline]])。

关联 [[sonic-vla-critique-roadmap]]([已证]history 是必选非可选)、[[sonic-masked-ce-p0-result]]、[[sonic-live-switch-idle]]、[[feedback-lesonic-similarity-leaderboard-standard]]。
