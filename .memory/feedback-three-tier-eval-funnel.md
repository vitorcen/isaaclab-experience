---
name: feedback-three-tier-eval-funnel
description: ckpt选择三段式漏斗=开环MSE粗筛(box零拉取秒级)→闭环quick 5-round(拉top3-4)→闭环strict 3×20/5×20(top1-2定榜);开环MSE单调降陷阱别取全局最低
metadata:
  type: feedback
---

# ckpt 选择三段式 eval 漏斗 / Three-tier eval funnel

**Why**：闭环 sim eval 慢 + 云端 **box→本机拉取 ~5MB/s 是瓶颈**（10 个 GR00T full ≈ 2.8h）。对每个 ckpt 都闭环全量评 = 浪费。业界标准 = **逐级收窄漏斗**：便宜的开环先粗筛淘汰大多数，只对少数 finalist 花闭环。

**三段式（从便宜到贵）**：
1. **开环 MSE 粗筛**（cheap，**box 上跑、零拉取、秒级/ckpt**）：`gr00t/eval/open_loop_eval.py` 在 held-out 轨迹上跑 `policy.get_action()` 算 **Unnormalized Action MSE**（无 sim/rollout）。扫**全部** N 个 ckpt → MSE 曲线。脚手架 `/root/openloop_sweep_v4.sh`（`uv run --no-sync python -m gr00t.eval.open_loop_eval --model_path <ckpt> --dataset_path <ds> --embodiment_tag new_embodiment --traj_ids 0..9 --action_horizon 40 --denoising_steps 4`，grep `Average MSE across all trajs:`）。**GPU 与训练串行**（别和训练同跑抢卡）。
2. **闭环 quick 5-round**（mid）：**只拉 MSE 选出的 top 3-4** 个 → closed-loop 5-round（`run_one.sh`，EPISODE_LENGTH_S=120 MAX_ROUND_WALL_S=150 STEP_HZ=60/N1.7 h=40）→ 收窄到 top 1-2。5-round σ 大(±18%) 仅作排序非定值。
3. **闭环 strict 3×20 / 5×20**（expensive）：top 1-2 → 3×20=60round（或 top-3 升 5×20=100round）→ **最终榜单值**（见 [[feedback-20round-strict-benchmark]]、[[eval-20round-still-noisy-combine-runs]]）。

**收窄数量**：N 个 ckpt → 开环全扫 → 闭环 quick **3-4** 个 → strict **1-2** 个。

## ⚠️ 开环 MSE 的两个坑（别盲信）
- **MSE 通常单调下降**（越训越拟合训练集），但**闭环成功率在 4-6ep 见顶后过拟合下降** → **别取 MSE 全局最低**（=最晚 ckpt=很可能过训，wsagi 21ep 即此）。正解 = **MSE 拐点(elbow，曲线变平处) ∩ 先验甜点带(GR00T 4-6ep)**，故 quick 取 3-4 个 hedge。
- **膝点定 best 的概率规则（2026-06-17 v10+v11 双案例实证）**：闭环 best ≈ **开环 MSE 膝点(陡降转平处)那个 ckpt 或紧邻后 1 个（拐点±1）**，高概率落在"膝"上；**膝之前 MSE 仍高=欠训**，**膝之后趋向全局最低及以后=过训**。
  - v10(batch48冻视觉)：MSE 3500=75.7(欠训)→**4000=35.3 陡降(膝)**→4500=36.6→5000=24.9(全局最低)。闭环 best=**4500=81.1%@60**（膝的**下一个**），5000全局最低却 quick 73.3%(过训)。
  - v11(batch48解冻)：MSE 5500=27(欠训)→**6000=13 陡降(膝)**→6500=11.5→7500=5.9(全局最低)。闭环 best=**6000**（膝**本身**，quick 93.3%；strict 待 6500 落地钉），7500全局最低却 quick 73.3%(过训)。
  - **关键**：①best 是"膝±1"不是死板"下一个"（v10是下一个、v11是膝本身）；②**膝由曲线形状定、非固定 epoch**（v10 膝~5.3-6ep，v11 解冻膝~7.9ep，解冻视觉膝更晚）；③"越低越好"是反的，**全局最低=过训陷阱**。
- **开环 MSE ≠ 闭环成功率**（模仿学习误差累积，低 MSE 可任务失败，见 [[sonic-vla-critique-roadmap]]）→ 开环只**淘汰明显差的 + 缩 finalist**，**最终排名必须闭环**。训练 loss 更不可信（wsagi 训练 loss 更低却闭环更差 60%<67%）。

**How to apply**：长 sweep（≥6 ckpt）默认走三段漏斗，别一上来闭环全评。开环在 box 跑完→报 MSE 曲线+标拐点/甜点带→只拉 3-4 闭环 quick→1-2 闭环 strict。early ckpt(<3ep) 开环 MSE 会正确判高(差)，与"GR00T 开关式曲线<3ep≈0%"一致，是粗筛有效的自检。
