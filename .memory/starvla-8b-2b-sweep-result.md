---
name: starvla-8b-2b-sweep-result
description: 8B sweep 干净结果(30k峰7/9后悬崖塌)+ 2B 训练 live state + 云端下载 finalize-hang 修法
metadata:
  type: project
---

**8B(Qwen3-VL-8B,bs=4,60k步)干净 sweep curve**(3-round quick,本机 8bit eval):
`6k:0 / 12k:0 / 18k:1 / 24k:4 / 30k:7(峰) / 36k:0 / 42k:0 / 60k:0`(48k/54k 漏测,在塌陷区无所谓)。
**30k=7/9 见顶后悬崖式过拟合塌陷**,符合 StarVLA 特性 + bs=4 样本右移预测(30k≈4B 15k 峰的 120k 样本)。
30k 是赢家。**strict 20-round(8bit)真值大胜 4B**:E(🍊)/ep **53.3%(32/60)** vs 4B 35%(21/60);
P(3) **35%(7/20)** vs 4B 10%(2/20);分布 P3=35/P2=10/P1=35/P0=20,5-round σ=22.4%。
→ **8B 的 vision dividend 实打实兑现**,橙子率 +18 点、完整成功率 3.5×。本机 24G 8bit 测(≈bf16 可信)。
⚠️ 36k-60k 第一遍是 wallx 抢卡期跑的**假 0**,在干净 GPU 下重测确认才是真塌(见
[[starvla-8bit-eval-load-corruption]] 的 GPU 争用坑)。

**Sweep watcher 基建**(`LeIsaac/scripts/evaluation/starvla_8b_sweep_watcher.sh`,2B 复用):
poll 云端 ckpt → 拉 → **拉完即从云端 rm**(100G 盘装不下多个大 ckpt,ENOSPC 会崩训练,见
[[feedback-training-save-policy]])→ 8bit serve(带重试)→ GUI 3-round eval → CSV。
**self-heal**:每轮扫本地无 CSV 行的 ckpt 自动补 eval(救回 serve 崩跳过的)。
`MIN_STEP=3000` 过滤欠训早期。本地 base 复用(每 ckpt 只拉一次)。

**🔧 云端 HF 下载 finalize-hang 坑**:hf-mirror 把大 LFS 文件 302 重定向到 xet
(`cas-bridge.xethub.hf.co`);要 `HF_HUB_DISABLE_XET=1` + 用 **wallx env 的 hf**(装了
hf_transfer)+ `HF_HUB_ENABLE_HF_TRANSFER=1`。但 **hf_transfer 会在传完全部字节后、finalize
前挂住(0 MB/s,进程活着,.incomplete 大小=Content-Length 却不 rename)** → 看着像没流量。
修法:杀掉 → 用**标准下载器**(`HF_HUB_ENABLE_HF_TRANSFER=0`)重跑,它一看字节齐了**秒
finalize** 成正式文件。学术代理 network_turbo 对 xet 会 401,故走 hf-mirror。

**📦 已发布 + README(2026-06-07)**:8B 进 HF `wsagi/StarVLA-Qwen3-VL-8B-PickOrange`
(model card + 60s demo mp4 `starvla-8b-pickorange.mp4`(webm 截前 60s,**conda ffmpeg libx264 坏→用
`/usr/bin/ffmpeg`**)+ config.yaml(base_vlm 改成 `Qwen/Qwen3-VL-8B-Instruct`)+ 训练配方;
ckpt `steps_30000_pytorch_model.pt` 17.9G hf_transfer 后台传)。三处 leaderboard 已加 8B **rank 4**:
root `README.md` / `LeIsaac/README.md` / `scripts/benchmark/STRICT_LEADERBOARD.md`。video 嵌 model card 用
HF resolve full URL `<video src=.../resolve/main/starvla-8b-pickorange.mp4>`。
⚠️ **footgun**:`starvla_strict_eval.sh` METRICS 文件名只按 ckpt+quant → 用同 ckpt 跑 3-round demo
**会覆盖正式 strict json**(实测被 1/9 demo 覆盖,已从 README 里的 raw 重建)。已防呆:非 20-round 写
`_r<N>demo` 独立文件;且 backbone 现按路径 tag 自动解析 base(见 [[starvla-vlm-variant-2b-4b-8b]])。
⚠️ 3-round GUI demo 实测仅 **1/9(11%)**——n=3 巨大方差 + GUI 渲染拖慢推理;**strict 20-round 53.3% 才是真值**。

**Live state(2026-06-07,compact 时)**:
- westd:15528(4090-48G)8B 训完(60k),10 ckpt 全在本地 `LeIsaac/outputs/starvla-8b-run/run/checkpoints/`
  (各 17.9G),云端 checkpoints 已空、**可关机**。8B 30k strict 20-round 本机在跑(harness task)。
- westc:31709(4080S-32G)2B 训练中 ~step 1200/30000,save_interval=1000(30 ckpt),bs=8。
- 待办:strict 真值出来报数;8B 缺的 48k/54k 可补(预计0);2B base(~4.5G)拉本地 + 起 2B watcher
  (`MIN_STEP=3000`,等本机 GPU 从 strict 腾出再开,**勿与 8B eval 抢卡**)。
关联 [[starvla-vlm-variant-2b-4b-8b]] [[starvla-so101-cloud-training]]。
