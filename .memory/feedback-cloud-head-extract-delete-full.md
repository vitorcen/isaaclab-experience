---
name: feedback-cloud-head-extract-delete-full
description: 云端冻-VLM训练边训边在box抽head+GOLD验证重建后删full→省box盘+拉取快3×,与本地拉取并行不阻塞;附pgrep自匹配假阳性坑
metadata:
  type: feedback
---

# 云端冻-VLM 训练：box 侧抽 head + 验证后删 full（省盘 + 加速拉取）

**Why**：GR00T-N1.7 / StarVLA 等冻-VLM 训练，每个 full ckpt 很大（GR00T bf16≈4.6G，含 2.2B；冻的 Cosmos VLM 部分每 step 字节相同），但可训只有 ~877M head。云端训练时 **box→本机 PULL 是 ~5MB/s 瓶颈**（[[autodl-hf-download-speed]]），拉 10 个 full ≈ 2.8h。在 **box 侧**抽 head-delta（~1.7G）+ 共享一份 vlm_base → 只拉 head = **快 ~3×**，同时省 box 盘。区别于 [[feedback-vla-ckpt-best-only-head-rest]]（那条讲**本地**存储纪律），这条强调 **云端侧、训练中、为拉取效率**。

**How to apply**（用户 2026-06-16 定的工作流）：
1. **训练 prune 关**（`LOSS_PRUNE_TOP_K` 设大如 20，默认 top_k=2 会删早期 epoch 毁曲线——v4 踩过，见 [[gr00t-n17-retrain-v3-curve]]），box 盘够就全留 full。
2. **box 侧抽 head**：每存一个 full，抽出 head-delta（对 vlm_base 逐张量 diff，工具 `scripts/ckpt/prune_ckpts.py`）。
3. **确认没问题再删 full**（`确认没问题就删`）：GOLD 验证 = `merge_ckpt.py vlm_base + head` 重建出的 full 与原 full **字节一致**，验证通过**才** `rm` 原 full → 省 box 盘。验证不过绝不删（删除不可逆）。
4. **与本地拉取并行、不互相阻塞**：box 抽head/删full 是一条独立轨；本地 watchdog 拉 head（小、快）+ 本地重建 + eval 是另一条轨。两轨解耦——box 删 full 只在「head 已安全（抽出+验证，或已被本地拉走）」后触发，不会删掉本地还没拿到的东西。
5. **最简安全变体（零重建风险）**：GR00T ckpt 是 2-shard safetensors，若 shard-1=冻结 Cosmos VLM 在所有 ckpt 间**字节相同**（`md5sum` 核对），则本地只拉一次 shard-1 + 每 ckpt 只拉差异 shard-2 = 无损去重、免 torch 重建。先验证 shard 是否干净分离再用。

**判据**：full 是否能由 head+base 字节级重建？能→抽 head 后可删 full（省盘/加速）；不能→留 full。删前必 GOLD 验证 + 确认无进程在用。

## 🩸 附:pgrep -f 自匹配假阳性坑（2026-06-16 血泪，浪费 ~1h）
`pgrep -f launch_finetune_ckpt_n17`（或任何 `-f <pattern>`）在 **SSH 命令行本身包含该 pattern 字符串**时，会**匹配到自己这条命令**→ 报 "procs: 2" 假阳性，让我误判"有 2 个卡死训练进程"→连环 kill/重启 → 把 box 搞到 SSH 风暴 + 死锁（见 [[feedback-shared-gpu-eval-queue-orphan-discipline]]）。
**正解**：① 判训练是否在跑用 **GPU util% + log 步数推进 + cgroup 是否在爬**，**别用 pgrep launcher 名**（命令里常含该名）；② 真要 pgrep，`ps -o pid,stat,args -p <pid>` 核对匹配到的是不是自己；③ 用不含 pattern 的精确路径如 `pgrep -af "bash outputs/.../watchdog.sh"`；④ **GPU 0% + log 冻住 + cgroup 不爬 = 真卡死**（非 warmup，warmup 时 cgroup 会从盘 page 爬向缓存大小）；⑤ 反复 kill 前先只读探活确认真有问题，**别 hammering**——重启大法（用户重启云主机）一键清干净比连环 kill 强。
