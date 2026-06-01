---
name: feedback-20round-strict-benchmark
description: "单 5-round 有 ±11% noise → 任何\"SOTA\"声称必须用 20-round 概率分布 + 5-round sub-sample σ 才严格"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 362665d2-8c80-4478-b57a-c15a8d3667c4
---

# 20-Round Strict Statistical Benchmark（强制升级 5-round）

发到 leaderboard / model card 的"SOTA"声称必须基于 ≥ 20-round 概率分布，**不能拿单次 5-round = 14/15 这种 single-shot 当结论**。

**Why:** 2026-05-23 自训 GR00T-N1.7 ckpt-6000 单 5-round 跑出 14/15 (4/5 envs)，看似平 hi-space SOTA。20-round 后真实分布：P(3)=50%, P(2)=20%, P(1)=15%, P(0)=15%, **mean = 41/60 = 68.3%**, 5-round sub-sample σ = 0.96 (6.4%)。14/15 是 **~4σ 上偏 lucky outlier**，真实 mean 10.25/15。如果只信单次 5-round 就发"SOTA"会被现实打脸。

**How to apply:** 任何要写 README leaderboard / HF model card / 横评结论的 ckpt，**必须**：
1. 跑 20-round (EVAL_ROUNDS=20，~50min wall on RTX 4090) BENCH 标准 (EPISODE_LENGTH_S=120, MAX_ROUND_WALL_S=180, STEP_HZ=60 for GR00T)
2. 用 `scripts/benchmark/aggregate_distribution.py` 出 P(placed=k) 表 + E(oranges) + 5-round sub-sample σ
3. 在 leaderboard 同时显示 mean ± σ，而不仅是 total oranges

## 关键事实（2026-05-23 实测，5090 + ckpt-6000）

- 60 episodes total, **σ(5-round) = 0.96 oranges = 6.4%**
- 单 5-round 95% CI = mean ± 2σ ≈ ±13%
- 14/15 出现概率（mean=10.25, σ=0.96 假设正态）≈ 0.04% — 极罕见
- 所以 hi-space 公开 14/15 大概率也是 single-run lucky，需要他们自己重新 20-round 才能严格 confirm

## 脚手架 (2026-05-23 已建)

- `scripts/benchmark/aggregate_distribution.py` — 读 metrics.json → md + svg
- `scripts/benchmark/run_one_strict.sh <slug>` — 一键 20-round + 自动 aggregate
- 输出 → `results/benchmark/<slug>.distribution.md` + `.distribution.svg`

## 模板 (md 输出格式)

```markdown
## <model> — strict 20-round distribution

| Placed per episode | Count | P(placed=k) |
|---|---|---|
| **3** | **10** | **50.0%** |
| 2 | 4 | 20.0% |
| 1 | 3 | 15.0% |
| 0 | 3 | 15.0% |

- E(oranges/ep) = 2.05 / 3 = 68.3% (41/60 placed)
- env_success rate: 9/20 = 45.0%
- all-3-placed rate: 10/20 = 50.0%
  - gap = 1 ep where 3/3 placed but env didn't fire task_done (model didn't return arm to rest)
- 5-round sub-sample σ = 0.96 (6.4%)
```

## 何时可以接受 single 5-round

只能在以下场景用 single 5-round：
- 早期 ckpt sweep / qualitative debug（"会不会动"）
- ckpt 间相对 sanity check（A 11/15 vs B 0/15 这种 gap 大的）
- 不发 leaderboard / 不写 model card

任何上 leaderboard 第一行的"新 SOTA"声称、HF model card 主表格、对外宣传 — **强制 20-round + distribution**。

## 关联

- [[feedback-5round-benchmark-standard]] — config 标准 (180s wall_cap, 120s ep, step_hz=60 N1.7)
- [[eval-5round-mandatory]] — 5-round 是最低门槛（不是 3-round）；本文升级到 20-round 是更严格 layer
- [[leisaac-5round-leaderboard-2026-05-21]] — 当前 leaderboard 多数 single 5-round，需要后续重测 20-round 才严格
- 脚手架：`scripts/benchmark/aggregate_distribution.py` + `run_one_strict.sh`
- 实证：`results/benchmark/wsagi-n17-ckpt6000-20round.{metrics.json,distribution.md,.svg}` (2026-05-23)
