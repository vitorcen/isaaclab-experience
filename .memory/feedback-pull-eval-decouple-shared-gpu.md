---
name: feedback-pull-eval-decouple-shared-gpu
description: 云端训练的插空 eval sweep,在共享 GPU 上必须把"拉 ckpt/抽 head(纯网络CPU,不占卡)"与"eval(占卡,需排队)"解耦成两路——拉取持续不被 eval 门阻塞(抢救 head 防 box prune),eval 排队只在 GPU 真空隙跑(让位邻居任务)。
metadata:
  type: feedback
---

# 插空 eval sweep:拉取与 eval 双路解耦（2026-06-13 定）

**场景**：云端 box 持续训练出 ckpt，本机 GPU 跑 eval sweep，且本机 GPU 与**邻居任务共用**(如隔壁 FlowDP 也在跑 Isaac eval)。

**标准 = 两路解耦,别串成一条**：
1. **拉取路(纯网络/CPU,不占 GPU)** —— 每轮**先把所有缺失的 head 全拉下来**(盒上抽 `action_model.*` head → rsync 本地)。
   这一路**绝不被 eval 的 GPU 门阻塞**,因为它不占卡。目的:**趁 box `keep_last` 裁掉旧 ckpt 前把 head 抢救到本地**(head 一旦抽出/拉回就永久安全)。
2. **eval 路(占 GPU,排队)** —— 对未评的 head **GPU-gated 排队**,只在 GPU 真空隙跑,**让位邻居任务**。

**Why**：若写成串行"拉一个→等门→评一个"(单趟 for 循环),等门时**拉取也停了** → 邻居长时间占卡时,新 ckpt 的 head 拉不下来,可能被 box prune 丢失;且违背"插空"本意(eval 该让位,拉取不该停)。

**How to apply**：
- 主循环每轮两趟:`pass1: for step: [ -f head ] || extract_and_pull`(全拉,失败下轮重试);`pass2: for step: 已评跳过; until gpu_free; do wait; done; eval_step`。
- **GPU 门要按"邻居 eval + 真实余量"判,不是只看自己**:`无邻居的 eval 进程(按其端口/进程名)` **且** `nvidia-smi memory.free ≥ 我 serve+eval 峰值(实测,如 4B≈17G)`。只看"无 >2.5G 进程"会漏判——邻居 eval 中途 ramp 起来会撞满(实测两边双 Isaac → 97% 23.8/24.5G,差点 OOM)。
- 被要求"先清显存"时:杀自己的 serve+eval+watcher(**setsid serve 是独立 session,杀 watcher 进程组带不走它,要按 :端口/进程名 PID 单独 kill -9**),只留邻居;再用硬化门重启 watcher → 拉取继续、eval 自动排队等空隙。
- 本机 GPU 任务铁律:**setsid 完全 detach + Bash 工具 dangerouslyDisableSandbox**,否则在工具调用里同步跑会被 exit 144 杀(零输出);别前台同调用 `pkill -f`/`sleep`(也触发 144)。

脚手架实例:`LeIsaac/outputs/starvla-qwen35-4b-gr00t-v2-midlayer/e1_gap_sweep.sh`。
关联:[[e1-midlayer-sweep-live-state]]、[[starvla-av1-dav1d-thread-leak-enomem]]、[[feedback-vla-ckpt-best-only-head-rest]](抽 head/merge 机制)、[[eval-20round-still-noisy-combine-runs]]。
