---
name: eval-20round-still-noisy-combine-runs
description: 单次20-round对高方差策略仍是点估计(±~9%);PI_v3-8B同字节ckpt两次20-round=63.3% vs 41.7%,差异全在wall_cap轮数非推理速度;诊断"掉分"要拆avg_round_s分capped/非capped区分管线退化vs方差;高方差策略多轮合并(40-round)
metadata:
  type: feedback
---

# 单次 20-round 仍是点估计 → 高方差策略要多轮合并（2026-06-13 实锤）

**案例**：StarVLA-Qwen3-VL-8B-PI_v3（step-78000）**同一字节 ckpt、同配置、同机**两次 strict 20-round：
- 2026-06-08 = **63.3%**（38/60，P(3)=40%）—— 进了榜单 rank 🥉
- 2026-06-13 复测 = **41.7%**（25/60，P(3)=15%）
- 合并 **40-round = 63/120 = 52.5%, P(3)=27.5%, P(≥2)=50%**（最大样本=最严谨点估计）

**21.6 点差距 = 高方差噪声，不是管线退化**。诊断方法（可复用）：把 `avg_round_s` 按 wall_cap 拆开——
- 非截断轮推理速度**一致**：复测 66.8s ≈ 原 67.7s/round → **推理速度没退化**。
- 分差全在 **wall_cap 轮数 12 vs 5**：即随机场景 + flow-matching 随机噪声采样下，策略"180s 内没完成任务"的轮数随抽样浮动。
- 该策略 **5-round σ=18.4%**（极大），推回单次 **20-round 仍带 ±~9%**。

**Why**：闭环 sim eval 每轮随机化橙子位置 + flow-matching 推理每次随机初始噪声 = 两层随机源。20-round 把 5-round 的 ±20-40% 压到 ±~9%，但对高方差策略**还不够**——63.3% 是乐观抽样，真值 ≈ 52.5%，与同骨干 GR00T head 53.3% **打平**（原 README 宣称的"PI_v3 head +10 点 > GR00T head"被证伪，已修正 rank 🥉→5）。

**How to apply**：
1. **"掉分了"先别下结论是退化**：拆 `avg_round_s` 成 capped/非capped。非capped 轮速度一致 = 推理没问题，分差是方差或场景难度；非capped 轮变慢 = 真·推理退化（查 transformers/torch/flash-attn/dtype/GPU降频）。先排除字节级 ckpt 差异（merge 重建要核 key 数 + dtype + project_layers 等全模块齐全，不只 action_model）。
2. **高方差策略（5-round σ ≳15%）多轮合并**：跑 ≥2 次 20-round，pool 成 40-round 报点估计 + 注明两次 raw。单次 20-round 进榜=点估计，乐观/悲观抽样都可能。
3. **榜单口径**：N17 13-15% 这种**远低于所有抽样带**的负面不受此影响（方向稳）；但 50-65% 区间的名次对 ±9% 敏感，别用单次 20-round 定生死。

关联：[[feedback-20round-strict-benchmark]]（20-round 是底线但非终点）、[[eval-5round-mandatory]]、[[feedback-benchmark-table-sort]]、[[starvla-gr00t-v2-n17-head]]（同期 N17 评审，那里推理延迟是真截断、这里是方差）
