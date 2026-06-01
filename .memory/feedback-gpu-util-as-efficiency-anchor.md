---
name: feedback-gpu-util-as-efficiency-anchor
description: 训练性能优化的判断锚：GPU util mid-window % 而不是 it/s，结合 1Hz 双采样定位 CPU/H2D/GPU 三阶段瓶颈
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 39899a1b-ef9c-4403-93fc-2e3491bfb440
---

# GPU util mid-window % 是训练效率优化的判断锚

任何 VLA / 大模型微调跑得慢时，**第一性判断锚是 GPU util 时间占比**，不是 step/s，不是 loss 曲线。

**Why:** 2026-05-23 在 LeIsaac GR00T-N1.7 训练上花了一上午做 5 组对照实验，发现单看 step/s 会误导。memmap + worker tuning + pipeline patch 加完后 step/s 只涨 22%（0.86 → 1.05），但 **GPU util mid-window 从 50% → 61.3% 是 +11 pp 的真涨**。说明 H2D 重叠生效了，只是 steady-state 被别的 CPU 重活卡住。如果只看 step/s 会以为优化失败，丢掉关键证据。

**How to apply:** 任何新训练任务（DreamZero / π0.5 / X-VLA / SmolVLA / GR00T 等）启动后第一件事先建立 GPU util 基线，然后按瓶颈层 A/B 攻。

## 测量工具组（最低集）

1. **1Hz GPU+CPU 双采样脚本** `/tmp/sample_gpu.sh`（30 行 bash + nvidia-smi + top）：
   ```bash
   /tmp/sample_gpu.sh /tmp/gpu_<exp>.csv 1500 &
   ```
   关键指标：**gpu util mid-window mean**（去掉 startup + 收尾 5% 后的中段平均）。

2. **Per-phase profile patch**（env-gated `PROFILE_PHASES=1`）：插 monkey-patch 到 collator + image transform + VLM tokenize，每 20 步打印 mean/p90 ms。判断 worker vs 主线程谁慢。

3. **Micro-bench 隔离**：不要直接跑端到端，先用纯 dataloader bench 拿单点数据（如 `bench_dataloader.py`）。我用 baseline torchcodec 167 fps → memmap 2862 fps 证明 decode 单点 17× 提升后，知道端到端 17% wall-clock 涨幅意味着 decode 在 step 总时间里只占小头。

## 瓶颈分层判断表（GR00T 实测 → 通用模板）

| GPU util mid | CPU | 含义 | 攻击方向 |
|---|---|---|---|
| < 30% | 100% | CPU 完全堵 GPU | precache + non_blocking + 移 aug 到 GPU |
| 30-50% | 100% | CPU/GPU 串行，H2D 重叠不够 | non_blocking pin_memory + cuda.Stream prefetcher |
| 50-70% | 100% | H2D 已重叠，CPU 主线程瓶颈 | **移 collator 到 GPU 或缓存 tokenize 结果** |
| 70-90% | 50-80% | 接近理想，调 batch / grad_ckpt | bf16 / mixed precision / grad_accum 调整 |
| > 90% | 任意 | GPU bound，瓶颈在算力 | 模型变小 / 更快 GPU |

**陷阱**：CPU 100% 不一定全在 worker — 主线程 collator 单线程做 VLM tokenize 时 CPU 也 100% 但加 worker 完全没用。要 profile 才能区分。

## 已验证的优化顺序（按 ROI）

0. **🏆 减 grad_accum、增 micro_batch（保 effective batch 不变）**：steady it/s +178%（1.99→5.54）。这是单点最大优化。原理：每 step 少 N-1 个 micro-batch 同步点 + 大 matmul 更充分用 Tensor Core。GR00T-N1.7 4090 24GB 上 mb=2 ga=4 → mb=8 ga=1：VRAM 15→19.7 GB peak（几乎不涨，optimizer state 占主导），step time 502→180ms。**首先调这个，比 dataloader 优化收益大 10×**。
1. precache video → npy memmap（CPU decode → memmap read，17× 单点提升；解决 startup 但 steady 不动）
2. monkey-patch `Trainer._prepare_input` 加 `non_blocking=True` 配合 HF 默认 `pin_memory=True`（H2D 重叠，GPU util +11pp）
3. num_workers 别超过物理 P-core 数（i9-13900KF 是 8，超过反而 -5.8%）
4. prefetch_factor 不要超过 4（pf=8 反慢 2.9%）
5. torch.compile(action_head.forward) 仅 +3-5% 但 JIT warmup ~18s — **短跑反而慢，6000 步以上才回本**
6. ⏳ 未验证：flash-attn（装不上 — CN→GitHub release 网络太烂；预期 +5-12%）

## 反陷阱

- **不要只看 train_runtime**：80 step 短跑 40s startup 会污染数据；steady-state 看 logger 的 it/s 收敛值。
- **不要相信 8w > 4w**：i9-13900KF 8 P-core / 16 E-core，超 8 worker 抢资源反而降。
- **不要先动框架**：WebDataset/DataLoader2 是伪优化（Opus + Codex + Opencode 三模型一致反对）。
- **不要忘了 PROFILE_PHASES patch**：盲改 1 小时不如 profile 10 分钟。
- **🚨 CPU 100% 饱和时，不要把工作从主线程移到 worker**：本机 4090 上把 Qwen3VLProcessor.image_processor 从 collator 主线程拆到 4 worker 反而 GPU util 50→43%、wall +7%。主线程 collator 和 GPU forward 通过 non_blocking H2D 已是天然重叠；移到 worker = 让 workers 抢同一批 P-core，喂数据更慢 → GPU 等更多。这条陷阱只有在 worker 仍有 CPU slack（CPU 总 util &lt; 70%）+ 主线程是单点瓶颈（profile 显示主线程 &gt; 40% step）+ batch_per_call ≥ 16 imgs 时才不踩。
- **不要用单 call 大 batch 测 估算总收益**：本来想 `image_processor(16 imgs) = 43.8 ms`，以为拆 worker 省 175 ms / step；实际 collator 每次只 4 imgs（micro_batch=2 × n_cam=2），单 call ~10 ms，节省被 worker 阻塞吞没。**估算前先量真实 batch size**。
- **🚨 不要只看 GPU util mid_window**：mb=8 ga=1 比 mb=2 ga=4 快 178% 但 mid_util 反而从 61% 跌到 21%。**原因**：step time 缩到 180 ms 后 1Hz sampler 大概率抓到 step 之间的 idle 间隙。p90 才是真值（86% vs 90%）。**步速跌不跌看 train_steps_per_second 收敛值，不看 mid util**。
- **不要假设 effective batch 变了就改了配方**：mb=4 ga=2 与 mb=2 ga=4 数学上等价（同 effective batch 8），但实测前者 +76% 步速 + loss 同收敛。"hi-space SOTA 配方 ga=4" 这种"等价"配置改了不会破坏收敛，只改速度。
- **CN→GitHub release 网络靠不住**：flash-attn 290MB wheel 直 wget 反复 EOF / corrupted，ghfast.top 也失败。**别死磕，跳过；ROI 又只 +5-12%**。如真要装：HF Hub 镜像 / autodl-tmp 上预下好后 rsync。

## 关联

- HTML 实战记录：`LeIsaac/docs/training/gpu_dataloader_zero_copy.html`（5 组对照 + micro-bench + 时间预算）
- 待建的通用脚手架：`scripts/training/perf/`（precache / bench / patches / sampler / profiler，DreamZero 等可复用）
- [[gr00t-n17-leisaac-wire-debug]] 同期工作 — 别和 wire 协议混
- [[feedback-pre-run-gpu-check]] — 跑实验前先 nvidia-smi 清残留
