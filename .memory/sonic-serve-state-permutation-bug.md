---
name: sonic-serve-state-permutation-bug
description: "serve_starvla_sonic.py state 归一化置换 bug — right_arm proprio 恒为常数,已修"
metadata:
  type: project
---

## serve state 归一化的 concat-then-normalize 置换 bug(2026-06-11 修)

`LeSONIC/scripts/serve_starvla_sonic.py` 的 `MinMax.__call__` 原本把 injector 送来的
「按 `_STATE_GROUPS` 顺序拼接的 state43」**直接当 raw `observation.state` 列**做 min_max 归一化。

**根因 = `_STATE_SLICES` 是个置换**:raw 列里 `left_hand=22:29` 排在 `right_arm=29:36` **之前**,
与 `_STATE_GROUPS` 列举顺序(...left_arm, right_arm, left_hand, right_hand)相反。concat-then-normalize 导致:
- `right_arm` 通道被喂进 `left_hand` 的 min/max
- `left_hand` 被喂进 `right_arm` 的**退化(全零)统计** → **right_arm proprio 在 serve 时恒为常数**

实测逐帧 max|serve−train|:**修复前 1.0573**(right_arm 整段常数)→ **修复后 0.000242**(纯 fp16 噪声)。

**修法(数据结构正确,非补丁):先把每组散射回 raw 列位置 → 归一化整条 43-vec → 再按训练 key 顺序重切。**

**Why:** G1 大量动作(fight 挥拳/dance 拍手)右臂是主动肢;serve 右臂 proprio 恒定 = VLA 闭环里瞎一只手,
独立压低 live 幅度。**与开环负结果无关**(dump 走训练 dataloader 不过 serve),但污染了**所有 live demo 的幅度判断** ——
CE v1 的「动作幅度小」里有一部分是这个 bug。

**How to apply:** 修复后需用权威 CE v1(expected-T0.5)重测 live demo,把「serve bug 致幅度小」与
「CE v1 head 本身保真不足」两因素拆开。任何 state 按 group 拼接 + 按 raw 列 stats 归一化的 serve 适配器都要查置换。

**相关**: [[starvla-sonic-ab-baseline]] [[sonic-masked-ce-p0-result]]
