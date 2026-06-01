---
name: feedback-training-resume-chunks
description: "Long training runs (hours-scale) should be split into N≈5 equal chunks via save_every + resume, not a single monolithic run"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

长训练默认分 ~5 段 resume 跑，**不要**一把梭哈一个 N 万步的单进程。

**Why:** smolvla / π0.5 / FastWAM QLoRA 都遵循这种模式。好处：
- 每段结束有明确的存档点（loss 曲线 / 显存 / 中间 ckpt 都能看），可中途叫停/调整
- 进程崩了不会丢全部进度（QLoRA 6B 模型一次启停 ~30s 加载，重启廉价）
- 用户能在两段之间介入决策（比如发现 loss 不降就提前停掉省 GPU 时间）

**How to apply:** 一次性 10k step 的训练 → 拆成 5 × 2000 step：
1. Phase 1: `max_steps=2000 save_every=1000 output_dir=runs/.../phase1`
2. Phase 2: `max_steps=4000 save_every=1000 resume=runs/.../phase1/checkpoints/state/step_002000`
3. ...
4. Phase 5: `max_steps=10000 save_every=1000 resume=runs/.../phase4/checkpoints/state/step_008000`

`max_steps` 是**累计**目标（不是本段步数）。`resume=<dir>` 触发 `load_training_state` 完整恢复 step + optimizer + scheduler；`resume=<file.pt>` 只恢复权重（不推荐用于分段续训）。

适用范围：单机长训（>2h）、QLoRA / LoRA 微调、任何上游 trainer 自带 resume 机制的训练脚本。短任务（<30min smoke）不必拆。

## 关联

- [[fastwam-qlora-finetune]] 当前主项目，5 phase × 2000 step
- [[pi05-pytorch-training]] π0.5 也是这种方式
- [[leisaac-smolvla-debug]] smolvla 同模式
