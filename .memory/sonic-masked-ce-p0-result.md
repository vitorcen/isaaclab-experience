---
name: sonic-masked-ce-p0-result
description: "Masked CE + proprio-history 双负结果实测 — 都比常数模板基线还差,架构死因已鉴定"
metadata:
  node_type: memory
  type: project
  originSessionId: 8c3a8a62-0a28-45de-95f0-59311cc264ba
---

## P0 Masked CE + P0+History — 双负结果(2026-06-11 离线开环实测)

之前写「改善不明显」是低估了。**离线 teacher-forcing dump(绕过 serving 栈)实测:两个都比 CE v1 差 2.5×,
连「每窗输出常数均值」的模板基线(0.0367)都打不过。** 8 窗全部一致回归,无一例外。

| 模型(同 flow3 / 24k 样本 / frozen Qwen3.5-4B / 开环) | MSE64 | bin_acc | token std(GT 0.216) |
|---|---|---|---|
| GR00T N1.7(锚点) | 0.0026 | — | — |
| **CE v1**(per-dim 独立,106M) | **0.0174** | **41.6%** | 0.219(=GT) |
| Masked CE(MaskGIT,58.8M) | 0.0437(差19%于模板) | 17.7% | 0.147(68%GT) |
| Masked CE + history K=3 | 0.0472(差29%) | 16.7% | 0.151 |

**死因分析(主结论 + mimo 复审修正,dump 走训练 dataloader 不过 serve,非实施 bug):**
1. **主结论:加 head 表达力(layerwise/masked/history 三连)在 5777 帧上一律开环退步**,最可能瓶颈是数据规模非 head 表达力。
2. **幅度塌缩 = confidence-ranking 揭示调度引起,非范式死刑(两个控制实验隔离)**:K=10 confidence 揭示 token std 0.147(68%GT)→ `SONIC_CE_MASK_STEPS=1` 单次前向恢复 0.19(88%GT)→ `SONIC_CE_REVEAL=random` 随机揭示恢复 0.18(83%GT)。**塌缩是调度产物可缓解,我先前判「固有病」过早(mimo 纠)。** 但三种解码 MSE64 全在 0.044-0.048,一律远差 CE v1 0.0174 → 修塌缩不修保真劣势。
3. **history 文本通路稀释(否定通路非概念)**:K=3 帧 state 过 256-bin digitize 塞 instruction(`[STATE_HIST]…[ACTION]`);~48% 相邻(维,帧)动不到 1 bin=半冗余,加长 ~138 token 稀释 cross-attn,且破坏 state 维间物理耦合。**连续 state encoder 未测,history 概念未否定。**

**⚠️ 「架构死刑」未坐实 —— mimo 指出两个未排 confound:**
- **d_model 不公(必须重训排)**:CE v1=1024 vs masked=768,每位置表达力差 (1024/768)²≈1.78×,被当成常量。
- **train/inference gap**:训练 revealed 喂 36% GT token,推理喂自己预测误差累积。
**待办两个 ~1 天消融:A=masked+d_model=1024+随机揭示+K=20(仍输 CE v1→死刑坐实);B=CE v1+数值 state encoder(有提升→history 概念有效)。两者都负再投 5-10 天 P2 扩数据。**

**Why:** 与 [[frozen-vlm-head-extraction-sweep]] layerwise 负结果同构,但下结论前要排便宜 confound,别直接跳最贵的扩数据。

**How to apply:** 先跑 A/B 消融(便宜),再决定 P2。两个 masked ckpt 留作负面存档。诊断开关 `SONIC_CE_MASK_STEPS`/`SONIC_CE_REVEAL` 已加进 QwenPI_CE._masked_decode(默认 off)。

文档: `LeSONIC/doc/sonic_starvla_swap_brainstorm.html` §11.5(双负结果)+§11.6(serve bug)。
脚手架: `starvla_dump_pred_tokens.py`(已支持 state_history 透传);pred 在 `datasets/sonic_vla_pred_starvla_ce_masked{,_hist}/`。

**相关**: [[starvla-sonic-ab-baseline]] [[sonic-vla-critique-roadmap]] [[sonic-p0-history-result]] [[sonic-serve-state-permutation-bug]]
