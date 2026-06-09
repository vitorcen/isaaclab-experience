---
name: feedback-stuck-detector-off-act-dp
description: ACT/DP 默认关 stuck-detector — episode_length_s 自然 cap 即可；stuck false-trip 会把 recoverable 的 episode 砍掉
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1e92da1e-d4b0-4cfa-b922-39f17de1ca36
---

# ACT/DP eval 默认 STUCK_WINDOW_S=99999

## Rule

`LeIsaac/scripts/benchmark/run_one.sh` 已按 `POLICY_TYPE` 设默认：
```bash
case "${POLICY_TYPE}" in
    lerobot-act|lerobot-diffusion)
        STUCK_WINDOW_S="${STUCK_WINDOW_S:-99999}"
        STUCK_EPS_RAD="${STUCK_EPS_RAD:-0}"
        ;;
    *)  STUCK_WINDOW_S="${STUCK_WINDOW_S:-30}"; STUCK_EPS_RAD="${STUCK_EPS_RAD:-0.05}" ;;
esac
```

**Why**: ACT chunk=100 / DP n_action_steps=8 都会有「短暂 pause / 重 plan」phase，关节微动 < 0.05 rad / 30s 是正常 plan-execute cycle，不是 dead policy。dp-v040-fullres ckpt-60000 ep2 实测 stuck 44.9s 假停，把可能放橙的 episode 砍了；ep3 不被砍直接 placed=[F,F,T]。GR00T 系列纯连续 action stream 才需 stuck，VLA chunked policy 不用。

**How to apply**:
- Default 自动按 policy 切换，啥都不用做
- 想强行打开：`STUCK_WINDOW_S=30 bash LeIsaac/scripts/benchmark/run_one.sh ...`
- `episode_length_s` 是真上限（60/120s），cap 内跑满即可

## 关联

- [[auto-eval-watcher-standard]] watcher 会复用 run_one.sh 的 default
- [[gr00t-placement-bug-fix]] placement 计数 reset 假阴 bug — 与 stuck 是两种不同的假阴
