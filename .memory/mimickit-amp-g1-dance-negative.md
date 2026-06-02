---
name: mimickit-amp-g1-dance-negative
description: AMP on G1 full 131s dance 失败（保真度）— Disc_Agent_Acc 平台 0.98、跟不上节奏；dance 保真要 DeepMimic phase-tracking 不要 AMP。含 AMP 启动方式 + 指标解读
metadata:
  type: project
---

2026-06-02 实验：用 MimicKit AMP 训 G1 整段 131s dance（`lafan_dance1_s2`，dance1_subject2 full），
默认 `amp_g1_agent` + `amp_g1_env`，4096 envs，停在 iter 2200。**保真度失败。**

**结果 / 数字：**
- `Disc_Agent_Acc` 从 iter ~800 起死平在 **0.98**（判别器 98% 一眼认出 agent 是假 → agent 只骗过 ~1.9%）
- 视觉：跟不上 dance 节奏，不如 DeepMimic 那段精准
- `Disc_Reward_Mean` 稳 ~0.95-1.0（policy 一直拿到风格奖励，run 没坏，但匹配不上精细分布）

**⚠️ 两个 "98%" 含义相反，不可比：**
- DeepMimic dance_15s "98%" = **Test_Ep_Length 跟踪覆盖率**（高 = 好，逐帧贴合参考）
- AMP "98%" = **Disc_Agent_Acc 判别力**（高 = 坏，动作 98% 可被认出非真）
- AMP 根本没有"逐帧匹配度%"——它 distribution-match 不 track

**Why 失败：** AMP 要匹配 131s 高方差全身舞蹈的**整体分布**，无 phase 时钟；默认 `amp_g1` 超参是为 walk/简单 motion 调的，吃不下 dance（最难的家族）。这跟 ASE 的 skill latent 要解决的问题同源，但 **MimicKit ASE 只有 humanoid 配置，无 G1**。

**结论：dance 保真度用 DeepMimic phase-tracking，不用 AMP。** "更长的 dance" → DeepMimic 30s 切段（见下），不是 AMP。AMP 适合"连续/可循环、不要求精确编舞"的目标。

---

## AMP 启动 + 指标解读（reusable）
- 启动：`BASE_ENV=amp_g1_env AGENT=amp_g1_agent bash scripts/mimickit_train_one.sh <motion> <iters>`（train_one.sh 已参数化 BASE_ENV/AGENT，sed 换 motion_file）
- `Return=0` 是**纯 AMP 正常**（`task_reward_weight=0`，模仿信号全在 disc reward）
- 收敛信号看 `Disc_Agent_Acc`：健康应**降向 0.5**（agent 学会骗判别器）；卡在 0.9+ 平台 = 判别器碾压 = 该 motion 在这套超参下学不动
- AMP 收敛快（~800 iter 到平台），**别傻跑到 4000**——平台后续训纯浪费（这次实测 iter 800 后 1400 iter 零进步）
- 负面家族按 cleanup 纪律只留 final（`model_0000002200.pt`）存档

## 下一步（compact 后做）
**DeepMimic 30s dance 切段**（我最早的性价比建议）：
```bash
python scripts/lafan_g1_npz_to_mimickit.py \
  --input  dependencies/MimicKit/data/motions/g1_extra/ember_lab/LAFAN_dance1_subject1_0_-1.npz \
  --output dependencies/MimicKit/data/motions/g1/lafan_dance_30s.pkl \
  --start_frame 1521 --end_frame 2421   # 以已验证的 15s 中段[1746:2196]为中心扩到 900f
bash scripts/mimickit_train_one.sh lafan_dance_30s 2500 \
  dependencies/MimicKit/output/train_lafan_dance_15s/int_models/model_0000001500.pt  # 热启动
```
预期：dance 连续，phase tracking 耐拉，30s 大概率仍 ≥90% 覆盖率。

关联：
- [[mimickit-lafan-fight-training-plan]] — 4 个 15s DeepMimic ship（dance_15s=98%）
- [[mimickit-g1-usd-material-fix]] — eval 用 g1_textured.usd（引擎默认已自挑）
