---
name: sonic-p0-history-result
description: P0+History (Masked CE + proprio K=3) = 负结果，开环 MSE64 0.0472 比模板还差，死因见 sonic-masked-ce-p0-result
metadata:
  type: project
---

**⚠️ 2026-06-11 已评估 = 负结果。** 完整死因鉴定 + 对比表 + 架构裁决见 [[sonic-masked-ce-p0-result]]。
摘要:开环 MSE64 **0.0472**,比 CE v1(0.0174)差 2.7×,比常数模板(0.0367)差 29%,8 窗全退步。
根因=小数据撑不起联合分布 + 迭代解码幅度塌缩 + history 文本通路稀释。下一步=转 P2 扩数据。

---

P0+History 训练完成：`sonic_qwen3_5_4b_ce_masked_hist`，6000 steps，checkpoints at 5000/5500/6000。

**改动：**
- `starVLA/dataloader/gr00t_lerobot/datasets.py` `_pack_sample` 加 proprio history（K 帧从 trajectory data 取）
- `starVLA/model/framework/VLM4A/QwenPI_CE.py` 加 `self.proprio_history`，override `add_discretized_state_to_instruction` 支持 `[STATE_HIST]` token
- `LeSONIC/scripts/serve_starvla_sonic.py` 加 rolling buffer + `--proprio-history` 参数
- `LeSONIC/scripts/starvla_sonic_live_demo.sh` CKPT 改指向 `_hist`，auto-detect 传 `--proprio-history 3`

**对比（均 6000 steps）：**
- CE v1 (baseline): MSE64 = 0.0125
- Masked CE (P0): 无明显改善（5777 帧撑不起联合分布）
- Masked CE + History (P0+H): 待 live demo 评估

**Why:** 原始 CE 只用单帧 state，缺少时序信息。K=3 proprio history 让模型看到前 3 帧的关节状态变化趋势。

**How to apply:** `bash scripts/starvla_sonic_live_demo.sh @flow3` 评估。如果效果好，可尝试更大 K 或加 action history。如果效果差，考虑增大训练数据或换 backbone。

**训练过程：** kernel 6.17 内存腐蚀极其严重，6000 步训练崩了 ~15 次 SIGSEGV/各种 Python 内部错误，每次都从最新 ckpt resume。关键：ckpt-2000 到 ckpt-3500 之间崩了多次但都没存新 ckpt（save_interval=500 但跑到 ~350 才崩），只能反复从 ckpt-2000/2500/3500 resume。
