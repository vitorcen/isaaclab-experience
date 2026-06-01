---
name: gr00t-placement-bug-fix
description: policy_inference.py 严格 placement 计数 bug：env.step() 触发 termination 时返回 post-auto-reset obs，placed_flags 读到 reset 后的随机橙位 → 全 False 假阴。修法：读 PRE-step obs
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f29b6eda-afc7-4618-b533-2c8e72f8ad99
---

# 严格 placement 计数的 reset 假阴 bug

`LeIsaac/scripts/evaluation/policy_inference.py` 单 env 模式下，旧代码：

```python
obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
f1, f2, f3 = _read_subtask_flags(obs_dict)  # ← 假阴源
placed_flags[0] = bool(f1)
...
if reset_terminated[0]:
    success = True; break
```

**症状**：env 报 success 但严格 placed_flags 全 False。h=40 N1.7 跑出 env 2/3, strict 1/9 的离谱差距，眼看进了 6+ 颗。

**Why**：IsaacLab 单 env 模式下 `env.step()` 触发 termination 时**内部 auto-reset**，返回的 obs 已是新一轮的随机初始态 → 橙子全在桌面随机位置 → loose check 全 fail。wall_cap 路径不触发 reset，所以那一路严格数是真的（h=40 ep1 wall_cap 抓到 1/3）。

**How to apply**：
- 计算 placed_flags 必须用 `env.step()` **之前**的 obs（== 上一步的 tail，保证 pre-reset）。
- 见 commit：`LeIsaac/scripts/evaluation/policy_inference.py` 把 `_read_subtask_flags(obs_dict)` 挪到 `env.step` 之前。
- 不改 non-sticky 语义（用户要求 "夹起来又飞出去不算"，所以仍是 snapshot 不是 OR）。

**后果**：所有 2026-05-21 之前的严格 oranges 计数都可能**低估**（特别是 env_success 路径）。包括：
- 自训 N1.6 ckpt-6500 (2/3, 8/9 → 真实可能更高)
- 全部 X-VLA / SmolVLA / π0.5 / ACT / DP 历史 eval
- hi-space N1.6 baseline 也低估

修复后第一组真实数：
- hi-space N1.7 h=40: **2/3, 8/9, 104s**
- 自训 N1.6 ckpt-6500: **未重测**（还在 HF readme 上的 8/9 是旧 buggy 计数）

## 关联

- [[gr00t-n17-hi-space]] hi-space N1.7 验证记录
- [[feedback-benchmark-table-sort]] leaderboard 排序规则
