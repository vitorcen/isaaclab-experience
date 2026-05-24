# Strict ≥20-round Leaderboard — PickOrange Probability Distribution

自动生成。**排名规则**: worst-case (5-round mean − 1σ) DESC →
E(oranges) DESC → σ ASC → env_success DESC.

Worst-case 指标惩罚高 variance — 高均值但 σ 大 = 实际单次 deploy 不可靠。

默认最低 20-round 入榜（strict 标准）；σ(5-round) 跨 5-round sub-sample 计算。See `feedback-20round-strict-benchmark` memory.


## 主表 — Main leaderboard

| Rank | Model | N | E(🍊)/ep | σ(5-rd) | **🔑⬇️ Worst-case (mean−1σ)/15** | P(3) | P(≥2) | env_success |
|---|---|---|---|---|---|---|---|---|
| 1🥇 | wsagi/GR00T-N1.7-PickOrange | 20 | **68.3%** (2.05/3) | 0.96 (6.4%) | **9.29** | 50% | 70% | 45.0% (9/20) |
| 2🥈 | hi-space/GR00T-N1.7-3B-Pick-Orange | 20 | **66.7%** (2.00/3) | 3.46 (23.1%) | **6.54** | 45% | 70% | 50.0% (10/20) |
| 3🥉 | GR00T N1.5 — LightwheelAI | 20 | **58.3%** (1.75/3) | 5.12 (34.2%) | **3.63** | 40% | 65% | 55.0% (11/20) |
| 4 | SmolVLA (other) — edge-inference | 20 | **25.0%** (0.75/3) | 1.26 (8.4%) | **2.49** | 0% | 20% | 5.0% (1/20) |
| 5 | SmolVLA 自训 main=15k (sweep best) | 20 | **25.0%** (0.75/3) | 1.26 (8.4%) | **2.49** | 0% | 15% | 5.0% (1/20) |
| 6 | ACT (other) — shadowHokage/act_policy | 20 | **28.3%** (0.85/3) | 2.22 (14.8%) | **2.03** | 10% | 20% | 10.0% (2/20) |
| 7 | π0.5 PyTorch expert-FT ckpt-2000 (4B base + 693M expert, freeze VLM) | 20 | **1.7%** (0.05/3) | 0.50 (3.3%) | **-0.25** | 0% | 0% | 0.0% (0/20) |

## 分布表 — P(placed=k) per model

| Model | N | P(0) | P(1) | P(2) | P(3) |
|---|---|---|---|---|---|
| wsagi/GR00T-N1.7-PickOrange | 20 | 15% | 15% | 20% | **50%** |
| hi-space/GR00T-N1.7-3B-Pick-Orange | 20 | 15% | 15% | 25% | **45%** |
| GR00T N1.5 — LightwheelAI | 20 | 30% | 5% | 25% | **40%** |
| SmolVLA (other) — edge-inference | 20 | 45% | 35% | 20% | **0%** |
| SmolVLA 自训 main=15k (sweep best) | 20 | 40% | 45% | 15% | **0%** |
| ACT (other) — shadowHokage/act_policy | 20 | 45% | 35% | 10% | **10%** |
| π0.5 PyTorch expert-FT ckpt-2000 (4B base + 693M expert, freeze VLM) | 20 | 95% | 5% | 0% | **0%** |

## Per-episode raw oranges
- `wsagi-n17-ckpt6000-20round` (20 eps): `[2, 0, 3, 3, 3, 2, 2, 3, 3, 1, 0, 3, 3, 0, 3, 1, 3, 3, 2, 1]`
- `hispace-n17-20round` (20 eps): `[1, 0, 3, 2, 1, 3, 3, 1, 0, 0, 3, 3, 3, 2, 2, 3, 2, 3, 3, 2]`
- `gr00t-n15` (20 eps): `[0, 2, 0, 0, 0, 2, 3, 2, 3, 1, 3, 3, 3, 2, 3, 3, 2, 3, 0, 0]`
- `smolvla-other` (20 eps): `[2, 0, 2, 0, 1, 1, 1, 2, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0]`
- `smolvla-self` (20 eps): `[0, 1, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 0, 2, 1]`
- `act-other` (20 eps): `[0, 1, 0, 3, 1, 2, 2, 3, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]`
- `pi05-expert-2k` (20 eps): `[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

---
**指标说明**:
- **E(🍊)/ep**: 每 episode 期望放置橙子数（满分 3）
- **σ(5-rd)**: 跨 5-round sub-sample 的标准差（用于估单次 5-round noise）
- **🔑⬇️ Worst-case**: 5-round mean − 1σ — 即 ~68% 概率任意一次 5-round 不低于此值；惩罚 unreliable 模型；**🔑 = sort key, ⬇️ = DESC**
- **P(3)**: 单 episode 全 3 颗成功的概率
- **P(≥2)**: 单 episode 至少 2 颗成功的概率 (useful threshold)
- **env_success**: 环境 `task_done` fire 的 episode 比例（含 arm-rest 要求）— 通常 ≤ P(3) 因为 placement 后 arm 没收回会卡 wall_cap