---
name: feedback-resume-anchor-ckpt
description: 长训练留一个滚动resume锚点(model+optimizer)以便续训而非重跑;SAVE_ONLY_MODEL省盘但丢续训能力
metadata:
  type: feedback
---

长训练应保留**一个滚动更新的可续训锚点**(model+optimizer+scheduler+rng),而不是全部 ckpt 都
`SAVE_ONLY_MODEL=1`(只存权重)。

**Why:** GR00T/HF Trainer 的 `resume_from_checkpoint` 必须有 optimizer 动量+scheduler 位置才能真续训;
只存 model 的 ckpt 无法 resume → 要延长训练只能从头重跑(浪费 GPU+ENOSPC 风险)。2026-06-16 v6 吃了亏:
全程 `SAVE_ONLY_MODEL=1`,想从 6000 续到 7200 时发现没 optimizer,只能重跑(还触发了 ENOSPC,6600 存到一半损坏)。

**实测大小(GR00T-N1.7 冻VLM,本地量 hi-space ckpt + 我们 v6):**
- model-only ckpt = **8.2G**(fp32 backbone,含冻结但照存的 VLM 4.7G + action_head 3.5G)
- **optimizer.pt = 13G**(Adam 两个动量 state + fp32 master,只覆盖可训参数 action_head)
- 完整可续训 ckpt ≈ **21G**;比 model-only 多 **+13G**

**How to apply:** 密集 eval ckpt 存 model-only(8.2G/个,给开环 MSE 筛选,便宜);**另加一个固定路径
的滚动 resume 锚点**(每次 save 后把完整态写/覆盖到 `OUTPUT_DIR/resume_anchor/`,只占 ~21G 固定不增长)。
HF Trainer `save_only_model` 是全局开关(没法 per-ckpt),所以靠自定义 callback 实现双层。13G 固定成本
远小于重跑 40min GPU + ENOSPC 风险。天花板步数已定的一次性 run(如 v6e→7200 到顶)不需要锚点,只
对"可能要延长"的训练有意义。见 [[feedback-training-save-policy]] [[starvla-checkpoint-resume-migration]]
[[feedback-cloud-env-reuse-disk-cleanup]]。
