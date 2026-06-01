---
name: eval-5round-mandatory
description: "单 3-round closed-loop eval variance ±20-40%（同模型同配跑 4 次出 1/3-3/3 env, 6/9-8/9 oranges, 51-133s）；leaderboard 必须 ≥5 round"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f29b6eda-afc7-4618-b533-2c8e72f8ad99
---

# 单 3-round eval 不可靠，最少 5 round

2026-05-21 hi-space N1.6 用同样 model + 同样 fix 后代码跑了 **4 次 3-round**：

| run | env | oranges | avg |
|---|---|---|---|
| 1 | 2/3 | 8/9 | 123s |
| 2 | 3/3 | 6/9 | 51s |
| 3 | 1/3 | 7/9 | 133s |
| 4 | 2/3 | 6/9 | 123s |

variance 大到无法做模型横评。同一模型 env 1/3-3/3、oranges 67-89%、time 51-133s 全跨过。

**Why:** PickOrange 任务对 3 颗橙子的随机初始 xy + 视觉 lighting jitter 高度敏感；3 ep 不够采样满整个 task distribution。

**How to apply:**
- leaderboard 所有 baseline 默认 ≥ **5 round**（最佳 6 round = `vla-eval-sweep` skill 已建议）
- 复现 eval 用 `EVAL_ROUNDS=5`
- 报告时给 `oranges_total / (rounds × 3)` 而非整 round 分数
- N=5 round = 15 ep；variance 经验上能降到 ±10%

旧 3-round 历史数据（hi-space N1.6 = 6/9 之类）**所有 < 5 round 的 oranges 数都不可直接比较**。

## 关联

- [[gr00t-placement-bug-fix]] placement bug 修了，但 variance 不是 bug 是任务本质特征
- [[xvla-best-inference-cfg]] X-VLA 之前用 6 round 是对的
