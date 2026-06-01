---
name: leisaac-eval-timeout
description: LeIsaac eval shell timeout 要按推理速度动态算，不能硬编码 600s — sim time ≠ wall-clock
metadata: 
  node_type: memory
  type: project
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

## 现象

DP 100k 在 PickOrange eval 第三颗后机械臂"完全定格不动"，等了好几分钟也没 Isaac sim episode timeout。

## 根因

`--episode_length_s=120` 是 **sim time**，不是 wall-clock。当 inference 慢时：

```
wall_clock_per_sim_step ≈ inference_ms / effective_chunk
slowdown = wall_clock_per_sim_step / (1000 / step_hz)
```

DP 实测：inference 393.6 ms/chunk，effective_chunk=8 (因 DP `n_action_steps=8` < client `horizon=16`)，step_hz=60 → 每 sim step 真实耗时 49ms vs 理想 16.67ms → **slowdown 2.95x**。

3 round × 120s sim time × 2.95x = ~1060s wall-clock + 60s startup ≈ **1100s wall-clock 才能跑完**。之前 shell `timeout 600` 永远在第一个 episode 就被杀。

## 三个模型实测 inference 速度

| 模型 | inference_ms/chunk | n_action_steps | slowdown @ 60Hz | 3-round timeout |
|---|---|---|---|---|
| ACT 80M | ~10 | 100 (client horizon=16) | <1x（无瓶颈）| 60+360×1=420s |
| SmolVLA 450M | ~22 | 50 (client horizon=16) | <1x | 60+360×1=420s |
| DP 267M | **393** | 8 | **2.95x** | 60+360×2.95×1.5≈**1700s** |

DP 慢是因为 **diffusion 100-step DDPM 采样**。可改 DDIM 10-step 提速 10x，但模型重训成本高。

## 修复 — `run_eval.sh` wrapper

`~/work/isaaclab-experience/LeIsaac/scripts/evaluation/run_eval.sh`：
- probe server log 提取最近 50 chunks 的 inference_ms 平均
- 从 ckpt `config.json` 自动读 `n_action_steps` 算 effective_chunk
- 公式：`timeout = startup + n_rounds × episode_s × slowdown × safety`
- 默认 SAFETY=1.5、STARTUP=60，可用 `TIMEOUT_SAFETY=2.0 bash run_eval.sh ...` 调

**用法**：替代直接调 policy_inference.py 的命令，把所有原 args 接在 `--` 后传过去。

## 算法 v2 (最终版，2026-05-17 晚)

之前公式 `slowdown × episode_s × n_rounds × safety` 是**错的方向** — 给慢模型放水。用户耐心 3-5 min，**应该 user-patience cap**：

```bash
STARTUP=60            # Isaac sim 冷启动
TIMEOUT_PER_ROUND=90  # GR00T 30s baseline × 3 tolerance
total = STARTUP + n_rounds × TIMEOUT_PER_ROUND  # 3 round = 330s = 5.5 min
```

timeout 是**判别器**不是**包容器**。慢模型跑不完一个 round = 判定不适合实时部署，**不要再等**。

run_eval.sh 现在的实现：probe inference_ms 只作为 informational warning（slowdown >2x 时提示"model too slow for 60Hz control"），实际 timeout 用上面 user-patience 公式。

## DP DDIM scheduler swap （本轮新发现）

DP 慢根因 = **DDPM 100-step 串行 sampling**（不是模型大）。改 ckpt config.json：

```
"noise_scheduler_type": "DDPM" → "DDIM"
"num_inference_steps": null → 32  # 4090 sweet spot
```

不重训直接 swap OK（DDIM 是 DDPM 子集，社区标准用法）。重启 server load 新 config。

inference 393ms → 147ms (DDIM 32-step)。slowdown 2.96x → 1.1x。

## DDIM 步数按 GPU 算力反推

拟合公式（4090 + Isaac sim 并行）：
```
inference_ms ≈ 36 + step × 3.3
# overhead 36ms = ResNet18 encode + ZMQ RTT
# per_step 3.3ms = UNet single denoising on 4090

target_inference_ms = effective_chunk × (1000/step_hz) × safety
                    = 8 × 16.67 × 0.85 = 113ms
max_steps = (target - overhead) / per_step ≈ 23 step  # 安全档
        = (133 - 36) / 3.3 ≈ 29 step                  # 临界档
```

4090 实测：30 → 2 颗，32 → 完成动作回 rest_pose（1 颗被推歪 plate 外），50 → 爆 3D。

弱卡 per_step 更高（3060 ~10ms），sweet spot 反推应是 ~7 step。建议未来在 run_eval.sh 启动前 calibration probe。

## 三模型 fair eval 对比

| 模型 | inference | sweet | 第一颗 | 第二颗 | 第三颗 | 瓶颈 |
|---|---|---|---|---|---|---|
| GR00T N1.5 | 50ms | — | ✅ | ✅ | ✅ | 1/1 success 30s 完成 |
| ACT 80M | 10ms | horizon=16 | ✅ | ✅ | 磨叽 | dataset OOD |
| SmolVLA 450M | 22ms | horizon=16 | 难 | — | — | VLM 60 ep 适配不足 |
| DP 267M | 147ms | DDIM 32-step | ✅ | 被推歪 | — | dataset OOD + 物理边界 |

三个独立架构都卡第三颗 / 第二颗后期 → **dataset 60 ep × 每集 1 次"放最后一颗"经验不够**，三个模型共同 OOD bottleneck。不是单一模型问题。

## 关联

- [[act-eval-debug-roundN]] eval 配方调试主线（horizon=16 / sim_warmup）
- [[feedback-pre-run-gpu-check]] 启动前先 check GPU + 残留进程
- 设计文档：`LeIsaac/docs/training/dp_inference_speedup_and_dynamic_timeout.html`（完整 HTML postmortem，含 SVG 拟合曲线 + 中英对照）
- 设计文档：`LeIsaac/docs/training/act_eval_debug_postmortem.html`（上一篇）
