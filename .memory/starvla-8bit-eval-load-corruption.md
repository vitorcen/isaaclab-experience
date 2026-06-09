---
name: starvla-8bit-eval-load-corruption
description: 大 VLM 本地24G eval 必须8bit + 8bit≈bf16实锤 + 大ckpt载入间歇堆腐蚀重试修法 + 单卡GPU争用坑
metadata:
  type: feedback
---

**8B(及更大)VLA 在本机 24G 卡 eval 必须量化**:serve + Isaac sim **同卡共存**,8B bf16 权重
~16G + Isaac ~7G = 必 OOM。8bit(LLM.int8)后 8B VLM ~11G、+ Isaac ~18G **挤得下 24G**(实测)。

**Why:** VLM 是冻结特征提取器,int8 近无损;DiT head 小且精度敏感,保 bf16。
**How:** `serve_starvla.py` 加了 `STARVLA_VLM_8BIT=1`/`STARVLA_VLM_4BIT=1` env 开关
(也有 `--vlm_8bit`)。实现 = **先 CPU bf16 load 完整 ckpt(device="cpu"),再就地把 VLM 的
nn.Linear 换 bnb int8 层,最后 `.cuda()`**——绕开"ckpt 自带 bf16 VLM 权重 vs int8 层"的
load_state_dict 冲突(不能用 from_pretrained-time 量化)。本机 `starvla_eval` env 已装
bitsandbytes 0.49.2。VRAM(4B实测):bf16 9.3G / 8bit 5.5G / 4bit 3.5G。

**8bit ≈ bf16 实锤**(4B steps_18000 strict 20-round 同协议对照):
橙子率 **21/60=35.0% = 35.0%**(完全相同),P(3) 5% vs 10%(n=20 噪声内)。
→ 8bit 放心用,不是"更好"(早期看着更好是 n=2/3 的小样本噪声)。

**🔴 头号坑:大 ckpt(8B=17.9G .pt)torch load 间歇(~40%)堆腐蚀** —— 表现**或**段错误(核心转储)
**或**诡异 `AttributeError: 'int' object has no attribute 'stale_possible_simple_keys'`(yaml/sre 报错)。
同一个 C 栈堆腐蚀根因(和 [[wallx-env-py310-torch-segfault]] 同源,py3.11 也中)。
**修法 = serve load 重试**(3-4 次,重启就过)。已加进 `starvla_8b_sweep_watcher.sh` 的 eval_one
+ `starvla_strict_eval.sh` 的 serve 段。单独脚本若没重试,失败就手动重跑一次即过。

**🔴 单卡 GPU 争用坑**:本机 24G 卡**不能同时跑两个 Isaac+serve eval**。`wallx_sweep_supervisor.sh`
有 `while true` 每 30s 自动重启 wallx watcher → 杀了 watcher 它又复活,周期性偷 GPU →
我的 8B eval 间歇拿不到显存 → **假 0/9**(serve 崩或 Isaac 中途缺显存)。教训:跑 sweep 前
`pgrep -f wallx_sweep_supervisor` 先杀 **supervisor**(不是只杀 watcher);多 sweep 串行别并行。
被污染的结果要在干净 GPU 下重测。关联 [[starvla-vlm-variant-2b-4b-8b]] [[wallx-eval-serving-adapter]]。
