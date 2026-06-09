---
name: auto-eval-watcher-standard
description: "训练每阶段自动 eval 是新标准 — `lerobot_finetune.sh AUTO_EVAL=1` 默认 spawn `eval_watcher.sh`，3 连 0-orange/stuck 自动 abort 训练"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1e92da1e-d4b0-4cfa-b922-39f17de1ca36
---

# 训练 = 训练 + 自动每阶段 eval (新标准)

## Rule

`scripts/training/lerobot_finetune.sh` 默认 `AUTO_EVAL=1`：训练启动同时后台 spawn `scripts/training/eval_watcher.sh`，它会 poll `$OUTPUT_DIR/checkpoints/` 监听新落地的 ckpt，逐个跑 3-round 60s quick eval (`EVAL_ROUNDS=3 EPISODE_LENGTH_S=60 MAX_ROUND_WALL_S=90`)。结果写入 `$OUTPUT_DIR/auto_eval.csv`。

连续 3 个 ckpt 0 orange / stuck → watcher 写 `$OUTPUT_DIR/.eval_abort`，wrapper SIGTERM 训练进程 → 不再白训 N 小时。

**Why**: DP v0.4.0 跑了 10h 100k step (loss 0.011 看着完美)，但实际两个 bug 叠加：crop_shape=(84,84) 切 1.7% 视野 + lerobot async server n_obs_steps>1 stack 空错（见 [[lerobot-dp-async-server-bug]]）→ 0/15 every eval。**任何 1 个 10k 步 quick eval 都能秒发现**。把这套写进 launcher 而不是靠人盯。

**How to apply**:
- 默认行为，啥都不用做
- 想关掉：`AUTO_EVAL=0 bash scripts/training/lerobot_finetune.sh ...`
- 自定 horizon：`EVAL_HORIZON=70` (ACT) / `EVAL_HORIZON=8` (DP) / `EVAL_HORIZON=50` (SmolVLA)
- **想全程看完不 abort：`STOP_AFTER_FAIL=999`** — user 偏好 "全程及时观察，免得到时候哭"，3 连 0 不代表后面也 0（loss 可能继续降，60k+ 才出 motion）。重要训练默认 999，让 abort 是 opt-in 不是 opt-out
- Resume 安全：CSV 持久 + 内存 dedup，已 eval 的 step 不重跑

## 去重机制（answers "只会 eval 未 eval 的节点对吧？"）

`eval_watcher.sh` 主循环每 `POLL_S` 秒扫一遍 `checkpoints/`：
```bash
if grep -qE "^${step}," "$RESULTS_CSV"; then
    seen[$step]=1
    continue
fi
```
两层 skip：in-memory + CSV 持久。重启 watcher / 训练 crash resume，已 eval 的 step 都不重跑；想强制重 eval = 删 CSV 对应行。

## 关联

- [[lerobot-dp-async-server-bug]] DP 不动的真根因（必须先 patch async server）
- [[feedback-incremental-eval-during-training]] 10-slice 规则 (CLAUDE.md)
- [[compact-resume-2026-05-22]] DP-v040-fullres 真正配置
- `LeIsaac/scripts/training/eval_watcher.sh` + `lerobot_finetune.sh` AUTO_EVAL hook
- `LeIsaac/scripts/auto_sweep_xvla_ckpts.sh` X-VLA 同款 sweep 但 post-hoc
