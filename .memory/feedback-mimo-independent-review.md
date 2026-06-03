---
name: feedback-mimo-independent-review
description: 习惯性按需用 opencode run -m xiaomi/mimo-v2.5-pro 做独立审查 — 关键设计/取舍决策前后开第二意见
metadata:
  type: feedback
---

做关键技术决策（设计取舍、参数选择、方案对比、代码改动）时，**习惯性按需**通过
`tmux new-session -d -s <name> "opencode run -m xiaomi/mimo-v2.5-pro \"$(cat /tmp/prompt.txt)\" 2>&1 | tee /tmp/out.log"`
开一个独立审查，拿第二意见。不是每件小事都问，是"有真取舍/我也拿不准"时主动开。

**Why:** mimo 是独立模型，能戳破我的盲点。2026-06-03 DART 实验里它一次纠正我两处错误选择
（prev_action 该记 executed 不是 clean 的因果一致性论证；σ 该上调到 0.10 不是我保守的 0.05），
都被后续探针/推理证实它对。独立视角的 ROI 很高，尤其是我自己有倾向性的设计决策。

**How to apply:**
- 时机：关键设计/取舍**前**（拿方案建议）或**后**（审查已做的改动），不阻塞自己——它在 CPU/API 跑，
  GPU 任务可并行；正好等它出结果时我推进别的。
- 形式：prompt 写清 context + 具体问题（逐条编号），结尾要"tight bulleted verdict per question, no preamble"。
- tmux 跑（`opencode run` 是一次性命令），`tee` 到 /tmp 日志，`sed 's/\x1b\[[0-9;]*m//g'` 去色再读。
- 用完收 tmux session（`tmux kill-session`），别留残留进程（见 [[feedback-pre-run-gpu-check]] 的清理纪律）。
- 把它的结论如实纳入决策记录（哪些采纳、哪些反驳），别假装是自己想的。

关联：[[mimickit-to-vla-distill-plan]]（DART 审查实例）、[[feedback-style]]（协作风格：澄清优先、负面如实写）。
