---
name: starvla-checkpoint-resume-migration
description: StarVLA ckpt 只存模型权重(无 optimizer);续训=重建optim+快进scheduler;迁移到别的机只需 head+vlm_base 合并;max_train_steps 必须取 save_interval 整数倍否则最后一个 epoch 不存
metadata:
  type: reference
---

StarVLA(QwenGR00T / QwenPI_v3 冻结-VLM)的 checkpoint / resume 机制,决定了如何把一个云端训练迁到**另一台机续训**,以及 TARGET 该怎么设。

## checkpoint 只存模型权重,不存 optimizer/scheduler

`train_starvla.py:_save_checkpoint`(~245)只 `torch.save(accelerator.get_state_dict(model))` →
`steps_N_pytorch_model.pt`(全模型=冻结VLM+head)。**没有 optimizer.bin / scheduler 状态**。
resume(`--trainer.is_resume True`)= 载入模型权重 → **重建 optimizer**(`setup_optimizer_and_scheduler`,Adam 矩从零)→
把 scheduler **快进 N 步**(`for _ in range(completed_steps): scheduler.step()`,行 232-237)。
⇒ Adam 矩在 resume 时丢失,但冻结-VLM 只训 head,几百步就 re-warm,代价极小。

## 迁移到别的机续训 = 只需 head + vlm_base,不必搬 11.4G 全 ckpt

因为"完整可续训权重"就是模型权重,而模型 = 冻结VLM(可重下/已 dump 成 vlm_base) + 训练的 head。
所以一个 run 的**完整 resume 料 = `steps_N_*_head.pt`(~1.1G,action_model.*+project_layers.*) + `vlm_base_<backbone>.pt` + base VLM(HF 可重下)**。
重建 + 续训步骤:
1. `merge_head.py vlm_base_<bb>.pt steps_N_head.pt steps_N_pytorch_model.pt`(本地即可,得 11.4G 全 ckpt)
2. 传到新机 `<run>/checkpoints/steps_N_pytorch_model.pt`
3. `RESUME=1 run_so101_train.sh`(`--trainer.is_resume True` 认最新 ckpt,重建 optimizer 接着练)
puller 拉的 head + 本地 `_head_sweep_tools/vlm_base_*.pt` 就已经是完整 resume 料 → **云端 box 可直接关机/释放,不丢续训能力**(2026-06-08 4B@23000 实证:head+vlm_base_qwen35_4b 全在本地)。

## TARGET 必须取 save_interval 整数倍(否则最后 epoch 白训)

`train_starvla.py:361-365`:**只在 `step % save_interval == 0` 存**,到 `step >= max_train_steps` 直接 `break`(**不存 final**)。
所以 max_train_steps 若非 save_interval 整数倍,最后那段(到 max 之间)训了不产 ckpt = 白烧卡,且拿不到那个 epoch 的权重。
**Why**:9B 设 max=13610(精确 6ep)、save_interval=1000 → 13000(5.73ep)是最后存的,**13610(6ep)根本不存**,13000→13610 的 610 步白训。
**How to apply**:max_train_steps 设成 save_interval 的整数倍,略微跨过目标 epoch 取整即可(见 [[feedback-vla-epoch-budget-6ep]]);反正 sweep 峰在 3-4ep,精确到某个 epoch 没意义。挂"到整千 ckpt 出现即停训"的 stopper 省尾段。
