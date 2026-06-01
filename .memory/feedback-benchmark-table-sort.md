---
name: feedback-benchmark-table-sort
description: LeIsaac benchmark README 表格排序规则：strict Rounds DESC → oranges DESC → time ASC
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

LeIsaac 多策略 benchmark 表格（README.md 简表 + LeIsaac/README.md 详表）统一按以下顺序排序：

1. **strict Rounds 完成数 DESC**（主键，3/3 > 1/3 > 0/3）
2. **strict oranges 数 DESC**（次键，5/9 > 4/9 > 2/9）
3. **平均完成时间 ASC**（第三键，越快越好）

**Why:** 用户明确要求按业界 benchmark leaderboard 惯例展示，rounds 是主指标（任务级），oranges 是细化指标（subtask 级），时间是 tiebreaker（效率）。不按 env_success 排，按 strict 排（避免假成功污染榜单）。

**How to apply:** 每次更新 benchmark 表格（rerun 任意一个 baseline / 加新 baseline / 修订 sticky 判定）后，重排两个 README 的 leaderboard，并同步 [[leisaac-policy-comparison]] 记忆。新加 baseline 默认插入正确位置而不是末尾。
