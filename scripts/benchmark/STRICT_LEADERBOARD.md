# Strict ≥20-round Leaderboard — PickOrange Probability Distribution

自动生成。**排名规则**: E(🍊)/ep DESC → P(3) DESC → env_success DESC → σ ASC.

E(🍊)/ep = total_oranges / N_episodes，即每 episode 平均放置橙子数（满分 3）。

默认最低 20-round 入榜（strict 标准）；σ(5-round) 跨 5-round sub-sample 计算。See `feedback-20round-strict-benchmark` memory.


## 主表 — Main leaderboard

| Rank | Model | N | **🔑⬇️ E(🍊)/ep** | σ(5-rd) | Worst-case (mean−1σ)/15 | P(3) | P(≥2) | env_success |
|---|---|---|---|---|---|---|---|---|
| 1🥇 | wsagi/GR00T-N1.7-PickOrange | 20 | **68.3%** (2.05/3) | 0.96 (6.4%) | 9.29 | 50% | 70% | 45.0% (9/20) |
| 2🥈 | hi-space/GR00T-N1.7-3B-Pick-Orange | 20 | **66.7%** (2.00/3) | 3.46 (23.1%) | 6.54 | 45% | 70% | 50.0% (10/20) |
| 3🥉 | GR00T N1.5 — LightwheelAI | 20 | **58.3%** (1.75/3) | 5.12 (34.2%) | 3.63 | 40% | 65% | 55.0% (11/20) |
| 4 | StarVLA-8B 自训 (Qwen3-VL-8B QwenGR00T, step-30k, 8bit eval) | 20 | **53.3%** (1.60/3) | 3.37 (22.4%) | 4.63 | 35% | 45% | 35.0% (7/20) |
| 5 | GR00T N1.6 (hi-space, h=40) | 20 | **48.3%** (1.45/3) | 2.63 (17.5%) | 4.62 | 25% | 40% | 30.0% (6/20) |
| 6 | GR00T N1.6 自训 ckpt-6500 (h=40) | 20 | **46.7%** (1.40/3) | 1.63 (10.9%) | 5.37 | 20% | 45% | 30.0% (6/20) |
| 7 | ACT (self) h=70 (h=16/30/50/100 全 wall_cap) | 20 | **43.3%** (1.30/3) | 2.08 (13.9%) | 4.42 | 30% | 40% | 35.0% (7/20) |
| 8 | StarVLA-4B 自训 (Qwen3-VL-4B QwenGR00T, step-18k) | 20 | **35.0%** (1.05/3) | 2.50 (16.7%) | 2.75 | 10% | 35% | — |
| 9 | ACT (other) — shadowHokage/act_policy | 20 | **28.3%** (0.85/3) | 2.22 (14.8%) | 2.03 | 10% | 20% | 10.0% (2/20) |
| 10 | SmolVLA (other) — edge-inference | 20 | **25.0%** (0.75/3) | 1.26 (8.4%) | 2.49 | 0% | 20% | 5.0% (1/20) |
| 11 | SmolVLA 自训 main=15k (sweep best) | 20 | **25.0%** (0.75/3) | 1.26 (8.4%) | 2.49 | 0% | 15% | 5.0% (1/20) |
| 12 | X-VLA 自训 weakaug 17k (h=32) | 20 | **6.7%** (0.20/3) | 0.00 (0.0%) | 1.00 | 0% | 0% | 0.0% (0/20) |
| 13 | DP (self) — wsagi/DiffusionPolicy-PickOrange | 20 | **0.0%** (0.00/3) | 0.00 (0.0%) | 0.00 | 0% | 0% | 0.0% (0/20) |
| 14 | OpenVLA-7B 自训 ckpt-5700 (vanilla, 8bit r64) | 20 | **0.0%** (0.00/3) | 0.00 (0.0%) | 0.00 | 0% | 0% | 0.0% (0/20) |
| 15 | π0.5 (self) — pt-v3 final_lora.npz (3.36B + 5M LoRA) | 20 | **0.0%** (0.00/3) | 0.00 (0.0%) | 0.00 | 0% | 0% | 0.0% (0/20) |

## 分布表 — P(placed=k) per model

| Model | N | P(0) | P(1) | P(2) | P(3) |
|---|---|---|---|---|---|
| wsagi/GR00T-N1.7-PickOrange | 20 | 15% | 15% | 20% | **50%** |
| hi-space/GR00T-N1.7-3B-Pick-Orange | 20 | 15% | 15% | 25% | **45%** |
| GR00T N1.5 — LightwheelAI | 20 | 30% | 5% | 25% | **40%** |
| StarVLA-8B 自训 (step-30k, 8bit) | 20 | 20% | 35% | 10% | **35%** |
| GR00T N1.6 (hi-space, h=40) | 20 | 20% | 40% | 15% | **25%** |
| GR00T N1.6 自训 ckpt-6500 (h=40) | 20 | 25% | 30% | 25% | **20%** |
| ACT (self) h=70 (h=16/30/50/100 全 wall_cap) | 20 | 40% | 20% | 10% | **30%** |
| StarVLA-4B 自训 (step-18k) | 20 | 40% | 25% | 25% | **10%** |
| ACT (other) — shadowHokage/act_policy | 20 | 45% | 35% | 10% | **10%** |
| SmolVLA (other) — edge-inference | 20 | 45% | 35% | 20% | **0%** |
| SmolVLA 自训 main=15k (sweep best) | 20 | 40% | 45% | 15% | **0%** |
| X-VLA 自训 weakaug 17k (h=32) | 20 | 80% | 20% | 0% | **0%** |
| DP (self) — wsagi/DiffusionPolicy-PickOrange | 20 | 100% | 0% | 0% | **0%** |
| OpenVLA-7B 自训 ckpt-5700 (vanilla, 8bit r64) | 20 | 100% | 0% | 0% | **0%** |
| π0.5 (self) — pt-v3 final_lora.npz (3.36B + 5M LoRA) | 20 | 100% | 0% | 0% | **0%** |

## Per-episode raw oranges
- `wsagi-n17-ckpt6000-20round` (20 eps): `[2, 0, 3, 3, 3, 2, 2, 3, 3, 1, 0, 3, 3, 0, 3, 1, 3, 3, 2, 1]`
- `hispace-n17-20round` (20 eps): `[1, 0, 3, 2, 1, 3, 3, 1, 0, 0, 3, 3, 3, 2, 2, 3, 2, 3, 3, 2]`
- `gr00t-n15` (20 eps): `[0, 2, 0, 0, 0, 2, 3, 2, 3, 1, 3, 3, 3, 2, 3, 3, 2, 3, 0, 0]`
- `starvla-8b-step30k-8bit` (20 eps): `[1, 3, 2, 3, 1, 1, 2, 3, 0, 3, 3, 1, 3, 0, 3, 0, 0, 1, 1, 1]`
- `gr00t-n16-hispace` (20 eps): `[3, 3, 1, 2, 2, 1, 2, 0, 1, 3, 3, 1, 1, 1, 0, 0, 1, 0, 1, 3]`
- `gr00t-n16-self` (20 eps): `[3, 1, 0, 3, 0, 3, 2, 0, 2, 2, 0, 3, 1, 2, 1, 1, 2, 1, 0, 1]`
- `act-self` (20 eps): `[1, 3, 3, 0, 2, 0, 0, 3, 3, 0, 3, 1, 1, 0, 2, 0, 0, 3, 1, 0]`
- `starvla-4b-step18k` (20 eps): `[1, 3, 0, 1, 0, 0, 2, 3, 2, 1, 0, 2, 1, 1, 2, 0, 2, 0, 0, 0]`
- `act-other` (20 eps): `[0, 1, 0, 3, 1, 2, 2, 3, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]`
- `smolvla-other` (20 eps): `[2, 0, 2, 0, 1, 1, 1, 2, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0]`
- `smolvla-self` (20 eps): `[0, 1, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 0, 2, 1]`
- `xvla-self` (20 eps): `[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]`
- `dp-self` (20 eps): `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- `openvla-self` (20 eps): `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- `pi05-self` (20 eps): `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

---
**指标说明**:
- **🔑⬇️ E(🍊)/ep**: 每 episode 期望放置橙子数（满分 3）— **主排序键**
- **σ(5-rd)**: 跨 5-round sub-sample 的标准差（用于估单次 5-round noise）
- **Worst-case (mean−1σ)/15**: 参考指标 — ~68% 概率任意一次 5-round 不低于此值；衡量 reliability
- **P(3)**: 单 episode 全 3 颗成功的概率
- **P(≥2)**: 单 episode 至少 2 颗成功的概率 (useful threshold)
- **env_success**: 环境 `task_done` fire 的 episode 比例（含 arm-rest 要求）— 通常 ≤ P(3) 因为 placement 后 arm 没收回会卡 wall_cap
