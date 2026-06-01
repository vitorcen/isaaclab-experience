---
name: feedback-training-save-policy
description: Long-training ckpt 保留策略：当前 phase 留最近 3 个 + 完成 phase collapse 到 final-only，watchdog 还要 prune 写坏的 partial ckpt
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

长跑训练（watchdog auto-resume 模式）的存盘策略，**必须**在 watchdog 层和 orchestrator 层各做一次裁剪，否则 4.5GB/ckpt × 200 saves = 900GB 直接把磁盘吃满，看起来像 flash-attn 随机崩，实际是 `No space left on device`。

## 双层 prune

**watchdog.sh（每次 attempt 前跑 prune_old_ckpts）：**
1. 扫所有 `step_*/model.safetensors`，size < 4GB（正常 4.6GB）= SIGSEGV 写到一半的，直接 `rm -rf`
2. 剩下的按 step 数排序，保留最近 `KEEP_LAST_N=3` 个，老的删
3. 理由：3 = 最新（resume target）+ 2 个 fallback（防最新这次写坏）

**run_all_phases.sh（每个 phase 完成后跑 collapse_phase_to_final）：**
1. 当前 phase 只保留 `step_<TARGET_STEPS>`（即下一 phase 的 resume 起点）
2. 其它全删
3. 理由：phase 完成后中间 step 没用了，只有 final 是跨 phase resume 必需

## 磁盘账

单 ckpt 4.5GB（FastWAM 5B 模型 + optimizer state，accelerate full-state save）。

- 训练中 peak：当前 phase 3 + 完成 phase × N（final-only）= 3 × 4.5 + 4 × 4.5 = 32GB
- 训练完：5 个 final = 23GB（可再删中间 phase，只留 final phase5/step_010000）

## 排查口诀

`No space left on device` → 看 `df -h /home`。watchdog 在跑但全是 exit=1 / exit=139 反复 crash → 99% 是磁盘满。disk-full SIGSEGV 后下次 resume 一定挂在 `safetensors_rust.SafetensorError: MetadataIncompleteBuffer`（partial write 头部不完整）。

**Why:** 2026-05-17 → 2026-05-18 跨夜训 FastWAM QLoRA，phase 2 从 step 2200 跑到 3150 后 30 attempt 全 fail，看 watchdog log 误以为是 stochastic SIGSEGV，实际全是 disk 100% 满了导致 ckpt 写一半就被 ENOSPC 砍掉。

**How to apply:** 任何 watchdog + auto-resume + frequent-save 的 setup（smolvla / π0.5 / FastWAM / 未来 OpenVLA 都是这种）都套这个双层策略。代码模板已落地在 [[fastwam-qlora-finetune]] 的 watchdog.sh / run_all_phases.sh，复用就行。
