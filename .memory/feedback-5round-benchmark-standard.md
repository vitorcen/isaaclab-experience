---
name: feedback-5round-benchmark-standard
description: 5-round leaderboard eval 的唯一权威 config 在 LeIsaac/scripts/benchmark/，不是 LeIsaac/server/eval_*.sh，不是临时改环境变量
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 362665d2-8c80-4478-b57a-c15a8d3667c4
---

# 5-Round Benchmark 标准 (强制)

任何 ckpt 跑 5-round leaderboard eval 必须用 `LeIsaac/scripts/benchmark/run_one.sh` (或 `run_all_baselines.sh`)，**不要用** `LeIsaac/server/eval_<model>.sh`。后者是 N1.6 时代的 ad-hoc 调试脚本，default config 跟 leaderboard 不一致。

**Why:** 2026-05-23 我自训 N1.7 ckpts (1200/2400/3600/4800/6000/7200/8400/9600) 跑了 11 次 5-round，全用 `LeIsaac/server/eval_gr00t.sh` defaults，结果发现 `EPISODE_LENGTH=60` (benchmark 是 120) + `MAX_ROUND_WALL_S=120` (benchmark 是 180) — episode 只给一半时间 + wall_cap 砍掉 hi-space episode 3 那种 "180s 跑到 2/3" 的合法 case。所有数据**不能与 README leaderboard 14/15 直接对比**，浪费 ~2.5h × ¥4 GPU 钱重新打分。

**How to apply:** 任何"5-round eval" / "横评" / "对比 hi-space" 任务，**第一件事**先看 `LeIsaac/scripts/benchmark/run_one.sh` 默认 + `baselines.tsv` 对应 row 的 `extra_env` override，按那个跑。

## Unified 标准（2026-05-23 当前）

| 参数 | 值 | 来源 |
|------|-----|------|
| `EVAL_ROUNDS` | **5** (老 default 3，[[eval-5round-mandatory]] 强制升到 5) | run_one.sh + memory |
| `EPISODE_LENGTH_S` | **120** | run_one.sh default |
| `MAX_ROUND_WALL_S` | **180** | run_one.sh default |
| `STEP_HZ` | **30** (default) / **60** (GR00T family override) | run_one.sh + baselines.tsv col7 |
| `ACTION_HORIZON` | per-model 查 baselines.tsv col3 (N1.7=40 / N1.6=50 / ACT=70 / DP=16 / SmolVLA=50 / X-VLA=32 / π0.5=35 / N1.5=16) | baselines.tsv + [[per-model-action-horizon]] |
| `STUCK_WINDOW_S` | 30 default / **99999 关掉** for `lerobot-act` & `lerobot-diffusion` (run_one.sh 自动 + baselines.tsv col7 显式) | run_one.sh switch + [[feedback-stuck-detector-off-act-dp]] |
| `STUCK_EPS_RAD` | 0.05 default / 0 for chunked policies | 同上 |
| `POLICY_TIMEOUT_MS` | 10000 (server.start) | start_server.sh |

总 wall budget per 5-round = `5 × max(EPISODE_LENGTH_S × 60/STEP_HZ, MAX_ROUND_WALL_S) + cleanup` ≈ **5 × 180s ≈ 15-20 min**

## 不要做的事

- ❌ 用 `LeIsaac/server/eval_gr00t.sh` / `LeIsaac/server/eval_pi05.sh` / `LeIsaac/server/eval_smolvla.sh` 跑横评 — 这些是 ad-hoc debug 脚本
- ❌ 临时 `EVAL_ROUNDS=5 EPISODE_LENGTH=60 MAX_ROUND_WALL_S=120 ACTION_HORIZON=40 POLICY_TYPE=gr00tn1.5 bash LeIsaac/server/eval_gr00t.sh` — 这就是 2026-05-23 翻车配方
- ❌ 改一两个 env var 就以为是 benchmark 标准 — 必须 **全部 4 项** 对齐

## 正确做法

```bash
# 1. 自训 ckpt 不在 HF？先加 baselines.tsv local-path entry 或临时 export 路径
# 2. 直接调 run_one.sh
EVAL_ROUNDS=5 bash LeIsaac/scripts/benchmark/run_one.sh <slug>

# 或者全跑：
EVAL_ROUNDS=5 bash LeIsaac/scripts/benchmark/run_all_baselines.sh
```

baselines.tsv N1.7 row 给的 N1.7 family eval cfg:
```
gr00t-n17  gr00tn1.6  40  hi-space/GR00T-N1.7-3B-Pick-Orange  gr00t-n16  GR00T N1.7 (hi-space, h=40)  STEP_HZ=60
```
- policy_type=gr00tn1.6 (wire 共用 N1.6 client — 但我 2026-05-23 验证 N1.7 server `--use-sim-policy-wrapper` 实际需要 N1.5 wire client 见 [[gr00t-n17-leisaac-wire-debug]]，这一条 baselines.tsv 标的 policy_type 可能 stale)
- action_horizon=40
- STEP_HZ=60 (N1.7 override)

## 关联

- [[eval-5round-mandatory]] — ≥ 5 round 强制 (variance ±10%)
- [[per-model-action-horizon]] — 各模型 horizon 不同，TSV lookup
- [[feedback-stuck-detector-off-act-dp]] — chunked policy 关 stuck detector
- [[leisaac-5round-leaderboard-2026-05-21]] — 当前 leaderboard (hi-space N1.7=14/15 SOTA)
- [[gr00t-n17-leisaac-wire-debug]] — N1.7 wire 协议 (baselines.tsv 上的 policy_type=gr00tn1.6 已经 stale，sim_wrapper 实际要走 gr00tn1.5 wire + observation envelope)
- `LeIsaac/scripts/benchmark/run_one.sh` — 权威 launcher
- `LeIsaac/scripts/benchmark/baselines.tsv` — per-model override
- `LeIsaac/scripts/benchmark/README.html` — 完整 baseline 文档
