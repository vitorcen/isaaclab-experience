---
name: feedback-training-output-cleanup
description: "After a training run is benchmarked / published, prune its outputs to family-winner + 3-6 ckpts; keep one failed variant per family as negative archive"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 30989355-7486-4921-8ab5-fa2239a26187
---

训练完 + eval/leaderboard 落地后，强制清理 `outputs/<run>/`：

- **每个模型家族留 1 个目录**（winner 优先，但负面家族也留 1 个做存档 — OpenVLA/π0.5 这种全废的也留，证明"试过且崩"）
- **每个保留目录留 3-6 ckpts**（best ckpt + 2-5 邻居 + LeRobot 的 `last`）
- **整目录删**：同家族重复试验、`*-cont`/`*-phased`/`*-smoke`/`*-patched-v2`、wire-debug 残骸（ckpt 缺权重不可 inference 的）、被新分支取代的旧版

**Why**: 2026-05-29 清理 LeIsaac/outputs 实测 1 TB → 196 GB（释放 ~800 GB，34 dirs → 14 dirs，118 ckpts 砍掉）。X-VLA 一个 run 跑 weakaug 留 44 ckpts ≈ 121 GB；同一家族还有 8 个失败分支各 40-56 GB。不清理 → disk full → 下次训练 `torch.save` SIGSEGV，看起来像 flash-attn 崩，实际是 ENOSPC。

**How to apply**:
1. Trigger: 模型卡 / leaderboard 行已发布（eval 结果变 external truth）
2. `pgrep` 确认无活跃训练进程在写该 dir
3. Dry-run 出精确字节给用户看，确认后才 `rm -rf`
4. 规则同样写入 `LeIsaac/CLAUDE.md`，每个 VLA 训练 repo 都该有
5. 注意：strict leaderboard winner 的 best ckpt 永远不动

参考 [[feedback-training-save-policy]]（训练时双层 prune，留最近 3 + 完成时 collapse 到 final-only）和这条配合 — 训练时 watchdog 控规模，训练完 benchmark 后再二次砍到家族存档。
