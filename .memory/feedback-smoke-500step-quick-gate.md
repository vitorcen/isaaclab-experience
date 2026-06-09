---
name: feedback-smoke-500step-quick-gate
description: 云端起新 VLA 训练前的 smoke 用 ~500 步(控制在 ~20min)只快速验"能训不崩+VRAM+臂会动",不必 1000 步
metadata:
  type: feedback
---

新 backbone/配方起训前的 **smoke 用 ~500 步**(而非 1000),目标墙钟 **~20min**,只做"快速能动判断"。

**Why**: smoke 的全部价值 = ① 训练管线不崩(env/config/dataset 通)② VRAM 装得下 ③ 抽 ckpt 一眼看臂会不会动。这三点 500 步足够暴露;1000 步(4B 在 A800 是 ~57min)是浪费。早 30min 拿到 go/no-go = 早 30min 决定是否投全量。

**How to apply**: smoke 脚本 `MAX_STEPS=500 SAVE_INTERVAL=500`(run_train.sh 已支持这两个 env)。smoke 跑完拉 500-step ckpt 本机 merge+serve 快速 eval(3-round 甚至 1-round)只看臂朝橙子去/会抓 vs 乱飞/不动。

**Caveat(实事求是)**: 500 步冻结-VLM + 从头 head 可能只够显示"动起来了"而非"有目的地朝橙子",别用 smoke 的 eval 数当能力判断——能力看全量从 5k 起的正式 sweep。smoke 只是"不是死的"门禁。

关联 [[starvla-so101-cloud-training]] [[three-box-sweep-live-state]] [[feedback-incremental-eval-during-training]]。
