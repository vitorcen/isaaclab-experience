---
name: per-model-action-horizon
description: 每个 VLA 模型有自己训练时的 chunk size = best inference action_horizon；eval 必须按模型分别查找配置，硬编码 16 是错的
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f29b6eda-afc7-4618-b533-2c8e72f8ad99
---

# 每个模型必须用自己训练时的 action_horizon

eval 时 `ACTION_HORIZON` 不应统一写 16。N1.7 实测 h=40 (其训练值) >> h=16；N1.6 训练用 50，被我们用 16 跑了一整轮全部低估。

**Why**: 每个策略训练时见到固定 chunk 长度（数据载入器决定）的动作序列；推理时 chunk 短于训练 = 截断 = 只看到 attention 中前几步的策略，policy 学到的"长程"决策全丢。

**已知 action_horizon 表（来自各模型 config.json `action_horizon` 字段）:**

| 模型 | action_horizon | 来源 |
|---|---|---|
| hi-space/GR00T-N1.7-3B-Pick-Orange | **40** | README + config.json |
| hi-space/GR00T-N1.6-3B-Pick-Orange | **50** | config.json |
| wsagi/GR00T-N1.6-PickOrange (自训 ckpt-6500) | **50** | config.json (复刻 hi-space 配方) |
| GR00T N1.5 (LightwheelAI) | 16 | 旧默认 |
| ACT (shadowHokage / 自训) | 100 | training chunk=100 |
| SmolVLA | 50 | smolvla default |
| X-VLA (自训 weakaug) | 32 (best @ 17k ckpt) | 之前 sweep 过 |
| Diffusion Policy | 16 (DDIM 32-step) | DDIM 不同 |
| π0.5 | 50 | pi05 default |

**How to apply**:
- 每跑新模型先 `python3 -c "import json; print(json.load(open(<cfg_path>))['action_horizon'])"` 或读 config.json `action_horizon` 字段
- 或写脚本自动 hf_hub_download config.json → 读 → 设环境变量
- 不要再用 ACTION_HORIZON=16 这种"通用默认"

## 建议落地位置

1. **TSV lookup table** `scripts/benchmark/baselines_action_horizon.tsv`，benchmark/eval scripts 读它
2. **eval_*.sh 内联**：每个 server eval 脚本头部贴自己的 `ACTION_HORIZON=<trained>` 默认（替代旧统一 16）
3. **auto-detect helper** `scripts/benchmark/get_action_horizon.py model_id` → 输出 N，shell `$()` 即用

最稳：1 + 3 组合，benchmark 脚本不再 hardcode。

## 关联

- [[eval-5round-mandatory]] N=5 round 必须；horizon 错了直接重测
- [[xvla-best-inference-cfg]] X-VLA h=32 是 sweep 出来的，符合训练 chunk
- [[gr00t-n17-hi-space]] N1.7 用 h=40 才发挥水平
