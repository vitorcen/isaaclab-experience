---
name: wallx-env-py310-torch-segfault
description: 这台机器上 python 3.10.20 conda 构建会让 import torch 间歇 segfault；Wall-X env 必须用 python 3.11。含 wall-oss 搭建配方
metadata:
  type: reference
---

## 🔴🔴 本机头号坑：python **3.10.20** conda 构建 → `import torch` 间歇 segfault（2026-06-05 卡了 ~3h）
**症状**：`python -c "import torch"` ~40% 概率段错误（核心已转储）；**纯 Python / numpy `import` 0% 崩**。
gdb backtrace = `_PyEval_EvalFrameDefault` ceval.c:1866 深递归（torch `_jit_internal._overload` → inspect.getsourcelines → tokenize → sre_compile）= **C 栈溢出**。
还会以"换花样的诡异报错"出现（同一堆腐蚀的不同表象，**不是独立 bug**）：`posixpath UnboundLocalError 'b'`、`sre_compile/sympy "unpack expected N got M"`、`scipy docscrape 'str' not callable`、`import torchvision` exit 139。

**根因 = python 版本**（100% 相关，2026-06-05 实测同机多 env）：

| python | torch import |
|---|---|
| 3.12.12 (lerobot) / **3.11.14** (isaaclab) | ✅ 0/20 稳 |
| **3.10.20** (`h741d88c_0`；wallx/vla/gr00t-n17/torchtest) | ❌ ~40% 间歇 segfault |

Python 3.11 重构帧实现（帧不再吃 C 栈），深 import 递归不溢出；3.10.20 这个构建溢出。
**与 torch 版本/CUDA build 无关**：py3.11+torch2.6.0+cu124 = 0/20；py3.10.20+任何 torch(2.6cu124/2.7cu128/cpu) = 崩。

**真·根治 = env 用 python 3.11**（不是重启、不是改 pyc/torch/CUDA——那些全是 2026-06-05 走过的弯路，无效）。
排查口诀：本机 import torch 间歇 segfault + 纯 Python 稳 = python 3.10.20 的锅，env 换 3.11，别 surgical 修。

## 训练状态（2026-06-05 跑通）
SO-101 PickOrange 微调**已跑通并在训**：detached daemon（`setsid run_leisaac_retry.sh`，脱离 session、survive compact）跑 **4 epoch ≈ 13h**，loss 1.1–2.4 健康，0.357s/step，VRAM 14.3G/24G。ckpt 每 epoch 存 → `LeIsaac/outputs/wallx-oss05-leisaac-pick-orange/{0,1,2,3}/`（首个 ~3.2h）。
- **抗崩 harness（用这个，不是 run_leisaac_retry）**：`bash workspace/leisaac_pick_orange/train_watchdog.sh`。它崩了(启动崩/训练中崩)自动从 `outputs/wallx-oss05-leisaac-pick-orange/` 最新 ckpt **resume 权重**重启，直到最终 epoch ckpt(`…/3/`)出现；每次裁剪到最近 3 个 ckpt。**挂了/compact 后只需重跑 `train_watchdog.sh`**，自动续。⚠️ trainer resume 是权重-only 暖启动(epoch/LR 重置)→ 多次崩续会过训一点，eval 挑最佳即可。
- **2026-06-05 加固版 watchdog 配方（关键，否则卡死在 FRESH-forever）**：本机 startup 崩率会飙到 ~100%(残余 import 腐蚀坏 streak)，且**快速崩完立刻重启会让 CUDA context 反复拆建、加剧腐蚀**。四个加固缺一不可：① **attempt 间 `SETTLE=10` sleep**(让 CUDA/驱动沉淀；单发 idle 启动能成、连环快启全崩就是这个)② **thread 限制** `OMP/MKL/NUMEXPR/OPENBLAS_NUM_THREADS=1` + `MALLOC_ARENA_MAX=2`(减 import 期线程竞争 + glibc arena 碎片)③ **`save_every_steps=2`**(=global_step 2=iter 64≈24s 就落 ckpt，赶在 ~iter100 崩点前；否则永远到不了首个 ckpt→永远 FRESH→0 进度。注：显示的 `iter` 是逐样本数，global_step=iter/grad_accum(32))④ trainer `save_every_steps` 补丁 + `num_workers:0`(+补丁⑦ prefetch=None)。验证：加固后 attempt 1 打穿训练存 `0_2`，attempt 2 从 `0_2` RESUME，续训链路通。⚠️ `save_every_steps=2` 每存 8GB I/O 重，**确认 0_4/0_6 在增长(向前推进、非卡 0_2 死循环)后调回 ~20**。
- **✅ 2026-06-05 真·续训已实现（trainer surgery，user 钦点"先实现真·续训"）**：原 `resume_from_checkpoint` 是 weights-only(epoch/optimizer/数据位置全丢)→ 崩续从 epoch 0 重头遍历，频繁崩时只训前 ~940 样本=废。改法(`qwen_vl_act_trainer.py`)：① full 分支用 `accelerator.load_state(ckpt)` 恢复 model+optimizer+scheduler+RNG(save_checkpoint 本就 `save_state`，对称 round-trip)；② 从 `global_step.pth`/`current_epoch.pth`/新增 `within_epoch_step.pth` 恢复 `global_step`/`start_epoch`/`initial_step`(step-ckpt→同 epoch 续 batch 偏移；epoch-ckpt→下个 epoch 起)；③ `train_loop` 首个 resumed epoch 用 `skip_first_batches(dl, skip_n)` 真跳已训 batch，消费后 `initial_step=0`；④ 删掉 __init__ 里会覆盖 global_step 的旧 224-227；⑤ 带 try/except 回退 weights-only。**效果**：崩→续 skip 已训→继续前进,~N 次崩续也能真正跑完 4 epoch。配套 `save_every_steps=20`(global_step 20=iter640，须 < 典型崩点 global_step 29，否则崩时无 ckpt)。**✅ 2026-06-05 重启后真机验证通过**：手动 kill 训练→watchdog 从 `0_20` 续→日志确认 `[resume] accelerator.load_state OK` / `start_epoch=0 initial_step=640 global_step=20` / `skipping first 640 batches` / 训练从 **iter 640 继续(非从 0)** / lr=1e-5 延续 scheduler 没归零。startup 偶发腐蚀崩仍有(post-reboot 也有但稀少),watchdog 重试 2-3 次穿过。
- **下一步**：① 重启后起 `train_watchdog.sh` 验真·续训日志；② 写 wall-x eval server adapter（仿 `serve_<policy>.sh` + `run_one.sh` 的 PORT 模式，见 [[vla-eval-sweep]] skill），ckpt 出来跑闭环验证文档预估的 40–65%。
- **跑通踩的 7 个补丁**（都已落地代码/config）：① wandb no-op shim（train_qact setup_logging）② 加载跳过 shape 不符 key（modeling from_pretrained，action_preprocessor 26→6 保新初始化）③ freeze_vlm 额外放行 action_preprocessor 可训（trainer）④ 数据集用 v2.1 的 `leisaac-pick-orange_old`（lerobot 0.3.4 是 v2.1，读不了 v3.0 的 tasks.parquet）⑤ KEY_MAPPINGS/ACTION_DATASET_NAMES 注册 `leisaac/pick-orange` ⑥ num_epoch 100→4 / 每 epoch 存（num_training_steps 只管 LR 不是停止条件）⑦ **`num_workers:0` 必须配 `prefetch_factor=None`**：`load_lerobot_dataset.py` 两处 DataLoader 硬编码 `prefetch_factor=2`，num_workers=0 时 PyTorch 抛 `ValueError: prefetch_factor only specified in multiprocessing` → 改成 `2 if num_workers>0 else None`。
- ⚠️ **2026-06-05 误判纠正**：当时 watchdog "14/14 attempt 全崩、0 进度" 我误判成"本机启动腐蚀率飙到 ~100%"，**真因是补丁⑦这个确定性 bug**（num_workers=0 + prefetch=2 每次必崩在 ~40s model-load 后）。修复后单跑一次就到 iter 85 loss 1.8。**教训="100% 复现" 永远先怀疑确定性 bug，不是概率性腐蚀**（呼应用户铁律：怀疑硬件/腐蚀前先查 env/代码确定性问题）。残余 import 腐蚀（3s 快崩、概率性）确实还在，但只占少数 attempt，watchdog 重试 2-3 次就有一个穿过去开始训。
- ⚠️ **残余间歇腐蚀（py3.11 也没根除，只降频）**：import torch 0/15 稳，但更重的操作仍偶发堆腐蚀，**三处都中过**：① 启动 ast 解析/深模型构建 → IndexError/segfault（修：`ulimit -s 131072` + 重试器兜）；② `import datasets` 偶发 segfault；③ **DataLoader worker 子进程**训到 iter 674 崩 `TypeError: unhashable type: 'list'`（也是腐蚀签名）→ 修：config `num_workers: 0`（数据在主进程加载，主进程相对稳；消除 worker fork 腐蚀面）。**这些"unhashable/IndexError/unpack/segfault"全是同一堆腐蚀的不同脸，别当代码 bug 排查。**

## 🔴🔴🔴 2026-06-05 真·根因落定：**kernel 6.17 per-VMA-lock D 状态卡死**（不是 env/代码）
跑了 2.5h 后整机退化：startup 崩率飙到 ~100%、连 `pgrep`/`ps`/`pkill` 都卡住、`/proc/loadavg` load=22 但只有 1 个 running（`1/1598`）。
**根因**：一个崩了的 `python3.11` 卡在 **D 状态(不可中断睡眠)**，`cat /proc/<pid>/wchan` = **`__vma_start_write`**（内核 per-VMA 写锁，kernel `6.17.0-22-generic` 的新特性）。它持 VMA 锁不放 → 任何碰内存的操作(fork/mmap/`ps` 读 /proc maps)全阻塞成 D → 级联 → load 飙、命令全卡。**`SIGKILL 杀不掉 D 状态进程**（卡在内核里）。
- **这才是上面"残余腐蚀"的真身**：mmap 密集操作(safetensors mmap 加载 4.2B、fork worker)触发 kernel per-VMA-lock bug；那些 `range_iterator context manager`/sre unpack/段错误/数据 fetch 段错误，大概率**都是 VMA 子系统出问题后内存被踩的不同表象**，不是 conda python 构建的锅(那条之前的判断要降权——py3.10 import torch 段错误可能也是同一 kernel 问题的早期表现)。
- **诊断口诀**：本机 load 突然飙高 + `pgrep`/`ps` 卡住 + `for p in /proc/[0-9]*; do [ "$(cut -d' ' -f3 $p/stat)" = D ] && echo $p; done` 看到 python/CUDA 进程 D 状态 + wchan=`__vma_start_write` → **kernel 卡死，唯一解是重启**，别再 surgical 修 env/代码。
- **真·根治待办**：升/降 kernel 版本绕开 6.17 per-VMA-lock bug（[[wallx-env-py310-torch-segfault]] 本条）；或训练时 `vm` 调参。重启只是临时复位(~2.5h 又会退化)。
- **pipeline 本身已证明正确**：clean 时能训到 iter 940、loss 1.4-1.8 健康、14.4G/24G 进卡、0.39s/step。崩全是机器态问题，非 Wall-X 配置问题。

## Wall-X (wall-oss) env 配方（conda `wallx`, **python 3.11**）
仓库 `dependencies/wall-x`；底座用 **wall-oss-0.5**（同 Qwen2.5-VL 架构、更强下游先验，见 [[vla-pickorange-vision-resolution-selection]]）。
- `pip install -r requirements.txt`（torch 2.6.0+cu124；**requirements pin 的 transformers==4.49.0 是 stale，代码 `vla_mixin.py` import `AttentionInterface`，须 bump `transformers==4.51.3`**）
- flash-attn 预编译 wheel 省 30min，**py3.11 用 cp311**：`flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311`（abiFALSE，见 [[autodl-uv-sync-cn-strategy]]）
- lerobot 官方 pin commit `c66cd401`（→ lerobot 0.3.4）
- wall_x csrc 编译：**必须 `export TORCH_CUDA_ARCH_LIST="8.9"`**（4090），否则 setup.py 调 `torch.cuda.get_device_capability()` 失败 → `metadata-generation-failed`。`pip install ninja` 提速，`MAX_JOBS=4 pip install --no-build-isolation -e .`
- 训练入口 `accelerate launch train_qact.py --config ...`（非 lerobot CLI）；**freeze_vlm: true** 只训 0.47B action expert ≈19G 进 4090；resolution face_view=-1（原生 640×480，不缩橙子）。
- 脚手架：`workspace/leisaac_pick_orange/{config_qact_leisaac.yml, compute_norm_stats_leisaac.py, norm_stats.json, run_leisaac.sh}`；KEY_MAPPINGS/ACTION_DATASET_NAMES 已注册 `leisaac/pick-orange`。

## ⚠️ 2026-07-12 修正:"解释器胡话错"两个真根因,python 版本可能只是相关非因果
FlowDP 线在**全新 py3.11 env 复现同样胡话崩溃**('List' not callable/import torch 段错误/locale.locale)→ 单靠换 python 版本不是根治。实锤的两个独立根因:
1. **陈旧/半写 .pyc**:硬重启(整机冻死硬断电)会留下半写字节码 → 之后该 env 进程启动期随机胡话错。**修=`find <env> -name __pycache__ -exec rm -rf` 清光(当次 1568 个)+ 关键任务 `python -B`**。症状签名=错误发生在 import/解析层、每次死法不同、同命令偶尔能过。
2. **torchcodec 视频解码后端**(lerobot 0.4.x 装了就默认用)训练期进程内踩内存 → 每 500-1000 步随机崩;**同数据集同机器 pyav 后端零崩溃**(ACT 全程 pyav 稳,FlowDP 切 pyav 后 6000 步零崩)。**修=数据集/eval 全钉 `video_backend="pyav"`**(offline_action_mse.py 已钉,resume 时注意 ckpt train_config.json 里存的 backend)。
另:新建 env 记得移植 populate_queues 补丁(lerobot 0.4/0.5 的 predict_action_chunk bug,见 [[lerobot-dp-async-server-bug]]),否则 flowdp eval 报 stack expects non-empty TensorList。

## 🔴 2026-07-12 终审:第三个真根因 = aliyun 镜像 torch 2.7.1+cu126 wheel 的 CUDA 宿主侧有毒
二分定案(decode-only/forward-only/CPU/换env 四象限):**py311+torch2.7.1cu126(aliyun wheel)纯 GPU 前向(无视频无数据集)几分钟必炸**(CPython 帧损坏:Module._call_impl 里 self 未绑定/CUBLAS_STATUS_NOT_SUPPORTED/段错误);同 env 纯 CPU 前向 670 次零崩;同驱动同卡 py312+torch2.10.0cu128 前向 3831 次零崩。**换 torch==2.10.0(pypi 标准源,自带 cu128)后 8905 次零崩痊愈**。教训:aliyun pytorch-wheels 的 cu126 旧 wheel 别再用;装 torch 优先 pypi 标准源新版。eval 数字跨 torch 版本可比(同 fp32 权重,统计等价)。

## ✅ 2026-07-13 完结:第四因素 + 最终稳定配方(eval 侧全愈)
换 torch 2.10 后 eval 仍偶发腐蚀 → 最后一块拼图=**libav(pyav)与 torch CUDA 同进程交替仍互踩**(四象限:decode-only 稳/forward-only 稳/合并炸)。**终修=进程隔离**:`offline_action_mse.py` 把 CUDA 前向放 spawn worker 子进程(主进程只碰数据集+pyav 解码,永不碰 CUDA),①IPC 只传 **numpy**(torch 张量 pickle 走 /dev/shm 共享内存有竞态)②`recv` 必须带 worker 活性轮询(worker 段错误时裸 recv 永久阻塞,实测主进程挂 36 分钟)。终态=24 ckpt 双侧 sweep 一口气磨完。
**新 env 建成清单(lerobot 0.4.4 复现用)**:py3.11 + torch==2.10.0(pypi 标准源)+ **datasets==4.1.1 + pandas==2.3.3**(datasets 4.8.5 有 `Value.__call__` API break 载不了 v3.0 数据集)+ av 15.1 + torchcodec 不装(dual-ffmpeg 冲突源)+ populate_queues 补丁 + `python -B`。
