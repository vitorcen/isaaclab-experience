---
name: wallx-autodl-cloud-training
description: Wall-X PickOrange 微调搬到 AutoDL 云端（本地 kernel 6.17 腐蚀放弃）；env 配方 + 启动方式 + 一堆 SSH/conda 坑
metadata:
  type: project
---

## 为什么上云
本地机器 kernel 6.17 per-VMA-lock bug，训 ~40min 就腐蚀到 ~100% startup 段错误（见 [[wallx-env-py310-torch-segfault]]）。
2026-06-06 搬 AutoDL。**云端 kernel 5.15.0-117，稳定无腐蚀**，是正解。

## 实例 & 连接
- `ssh -p 12710 root@connect.westd.seetacloud.com`，密码在 **`pass autodl/westd`**（绝不写 memory）。
- **RTX 4080 NVL 32G**（AutoDL 的 4080 是 32G 变体；arch 8.9 同 4090，`TORCH_CUDA_ARCH_LIST=8.9` 不变），CUDA 12.4，GPU 模式 12 核。
- 系统盘 `/root` 30G（conda env + nvidia libs ~21G）；数据盘 `/root/autodl-tmp` 50G（项目/模型/数据/ckpt）。
- 关机/开机切卡保留两盘 + 进程被杀；切回要重启 mihomo。

## env 配方（conda `wallx`，py3.11，全在 `/root/miniconda3/envs/wallx`）
- **torch 2.5.1+cu124**（不是 2.6——AutoDL 没 2.6；2.5.1 仍 cxx11abi**FALSE**，配 flash-attn torch2.5 wheel）。装法见下"坑"。
- transformers **4.51.3**、accelerate 1.10.1、peft 0.17.1、torchdiffeq 0.2.5、qwen_vl_utils 0.0.11、diffusers 0.38.0、datasets 3.6.0、**triton 3.1.0**、numpy 等。
- **flash-attn 2.7.4.post1** wheel = `cu12torch2.5cxx11abiFALSE-cp311`（github release，pip+proxy 装，~23s）。
- **lerobot 0.3.4** editable `/root/autodl-tmp/lerobot-wallx`（`pip install --no-deps -e`）。
- **wall_x 1.0.1** editable `/root/autodl-tmp/wall-x`，csrc 已编译（`TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=12 pip install --no-build-isolation --no-deps -e .`）。
- 验证：`import torch,triton,transformers,flash_attn,lerobot,wall_x,accelerate` 全 OK，`cuda_avail True`。

## 🔴 云端搭建踩的坑（CN 网络 + 0.5核无卡 + SSH 怪）
1. **nvidia CUDA 库（cudnn 664MB）aliyun/代理都下不动** → **从 base env(py3.12) 拷贝**：`cp -rf /root/miniconda3/lib/python3.12/site-packages/{nvidia,nvidia_*.dist-info} 到 wallx 的 site-packages`。nvidia-*-cu12 是 `py3-none` 跨版本通用，**.so + dist-info 都要拷**（缺 dist-info 则 pip 不认为已装、又去下 cudnn）。
2. **triton 别从 base 拷**：triton 有 cp312 编译扩展，拷过来报 `module compiled for Python 3.12`。要 `pip uninstall triton; rm -rf site-packages/triton*; pip install --force-reinstall triton==3.1.0`（aliyun）。
3. **torch 装法**：aliyun pytorch-wheels 没 2.5.1；aliyun pypi 下 torch 大 wheel 会 stall。最终：torch 主 wheel `pip install --no-deps torch==2.5.1 --index-url download.pytorch.org/whl/cu124`（via proxy），nvidia deps 从 base 拷。
4. **filelock 等精确 pin freeze 版本 aliyun 没有** → 非关键包剥 `==版本`（`sed 's/==.*//'`），只硬 pin transformers/accelerate/peft 等关键。
5. **无卡模式 CPU 0.5 核**（cpu.max `50000 100000`）解包/编译极慢 → CPU 密集步（解包 wheel、wall_x csrc 编译）要切 GPU 模式（12 核）。纯下载在无卡做。

## 🔴🔴 启动方式（最关键，否则训练起不来）
- **`conda activate wallx` 在非交互 SSH 里会挂起** → 训练起不来、留僵尸 train_qact。**必须用 env 二进制全路径**：`/root/miniconda3/envs/wallx/bin/accelerate`，不要 conda activate。
- 启动脚本 **`/root/train_cloud.sh`**（已落地）：`exec > /root/tlog.txt 2>&1`（脚本内自重定向，不靠外部捕获）→ cd wall-x → export PATH=env/bin → `accelerate launch --num_processes=1 --main_process_port=295XX train_qact.py --config workspace/leisaac_pick_orange/config_qact_leisaac.yml --seed 42`。
- 启动命令：`nohup bash /root/train_cloud.sh >/dev/null 2>&1 &`（proven 形式）。**tmux send-keys 有 race 常不落地；nohup 偶尔 SSH 255（高负载）重试即可**。换端口避开孤儿占用。
- 看进度：`tail /root/tlog.txt`（找 `epoch N/4 | iter X | loss`）；GPU `nvidia-smi`（训练时 ~14.5G）。
- **SSH 输出常被吞** → 命令结果写文件 + 单独 ssh `cat` 文件读。

## 🔴 batch 提速是死路：640 分辨率下 compute-bound，batch 1 已最优（2026-06-06 实测）
误以为 batch_size_per_gpu=1 喂不饱 GPU（util 60%）→ 想提 batch 加速。**实测全错**：
- batch 8 OOM（单激活 22.86G）；batch 4 OOM（真实 24.27G+5.72G 瞬态>31.5G，连 expandable_segments 修完碎片也不够）；batch 2 能跑（20.8G/66%，util 88%）但 **~2.0s/iter = 1.0s/帧 vs batch1 的 0.85s/帧，反而慢 17%**。
- 根因：**640 分辨率 vision forward（~750 token/图 × 2 相机）是 per-frame 固定重活，每样本激活 ~2.86G，GPU 串行一帧已吃满算力**。批多帧只线性叠加耗时，零并行红利。util 60% 不是"等数据"，是 forward/backward 交替的结构性空隙，加 batch 填到 88% 也不变快。
- **结论：保持 batch_size_per_gpu=1 × grad_accum=32（有效 batch 32）。想保有效 batch 不变改 batch 必须保乘积=32。** num_workers 云端可设 4（kernel 5.15 安全，但数据非瓶颈、提速也有限）。
- **gs 记账（别再算错）**：`global_step` 跨 epoch **累积**，**32 iter/gs**（=accum），**1015 gs/epoch**，32478 iter/epoch；epoch0→gs1015、epoch1→gs2029、epoch2→gs3044、epoch3→gs4060（≈`num_training_steps:4000`，注释一直对）。epoch ~6.5h（32478×0.72s），4 epoch ~26h。step-ckpt 名 `{epoch}_{gs}`，save_every=500 → gs2500/3000/3500…
- 顺带验证：**从 epoch 边界 ckpt full-resume 改 batch 安全**（`within_epoch_step=0`→`skip_n=0` 不触发 skip_first_batches，optimizer/scheduler/gs 干净延续，line 1100-1108）；中途 step-ckpt 改 batch 会 skip 越界。改 batch 必从 epoch 锚点 resume。

## 配置（已改云端路径）
`/root/autodl-tmp/wall-x/workspace/leisaac_pick_orange/config_qact_leisaac.yml`：
pretrained_wallx_path=`/root/autodl-tmp/wall-oss-0.5/`，save_path=`/root/autodl-tmp/wallx-outputs/`，
data root=`/root/autodl-tmp/datasets/leisaac-pick-orange_old`（v2.1，meta/info.json+tasks.jsonl 齐全），
norm_stats=workspace 内。num_epoch=4(~13h)，save_every_steps=20，freeze_vlm=true(~14.5G/32G)。

## 🏁 终局（2026-06-07，4 epoch 训完 → 负面 → 停掉归档）
**训练完整跑完 4 epoch**(`EXIT rc=0` 09:35),最终 best ckpt `3`(gs4060)。**双 serving bug 修完(分辨率 + proprio,见 [[wallx-eval-serving-adapter]])仍 0/9**——最终 `3` retract-detect 关闭跑满 240s×3 也 0/9,行为从"乱挥"改善到"有目的伺服靠近橙子"但跨不过 grasp。**用户决定收尾停掉,Wall-X(freeze_vlm)PickOrange 归负面**,卡让给已 work 的 StarVLA(5/9)。
- **ckpt 已全部落地本地** `LeIsaac/outputs/wallx-sweep/{0,1,2,3,3_4000,+评过的step}`(7.8G model-only,无 optimizer.bin;够推理/归档,不够 resume 训练——resume 需云端 8.7G 全态)。
- **本地 sweep(supervisor+watcher)+ serve 全停**。**云端训练已 EXIT,需用户去 web UI 关机/释放停 GPU 计费**(¥6-8/h;权重已本地,关机保数据 or 释放彻底停费)。
- 待办(可选):写负面 HTML 归档 + 传 HF(参照 pi05 负面);本地 ckpt 按 CLAUDE.md 清理(负面家族留 1 个 `3` 存档即可,删冗余 step-ckpt,需用户确认 rm)。

## 历史状态（2026-06-06 20:12，batch 提速折腾后回退原配置，训练中）
**训练健康前进**：epoch **2/4**，从 epoch1 锚点 `1`(gs2029) full-resume 续上，loss ~0.16，time_per_step 0.71s，GPU 13.3G/57%，0 崩溃。epoch ~7.7h，epoch 2-3 还需 ~15h（约 2026-06-07 中午完成 4 epoch）。
- config 现值（**batch 提速失败已回退,见上方 §batch 提速是死路**）：`batch_size_per_gpu: 1` × `gradient_accumulation_steps: 32`(有效 batch 32,实测最优)、`num_workers: 4`、`keep_last_step_ckpts: 1`（**磁盘铁律**：60G 盘只够 keep_last=1）、`save_every_steps`、`freeze_vlm: true`、`num_epoch: 4`。
- `train_cloud.sh` export 行已加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（修碎片,无害）。
- config 末尾 `resume: {ckpt: /root/autodl-tmp/wallx-outputs/1, load_ckpt_only: False}`。✅ **resume.ckpt 现指向 epoch 锚点 `1`(永不被剪)→ 崩溃重启可直接 setsid 重跑,无需先改 ckpt**（之前指向已剪的 1_1500 的隐患已解除）。

**真·续训已实测 OK**：从 step-ckpt resume → `[resume] accelerator.load_state OK / start_epoch=1 initial_step=15522 global_step=1500 / skipping first 15522 batches`，继续前进不从头。

**compact 后续训查验**：
1. `grep -E "epoch +[0-9]+/" /root/tlog.txt | tail -1` 看在训；`grep -c "EXIT rc=1"` 看崩没崩。
2. 崩了：改 config resume.ckpt→最新 ckpt，`setsid bash /root/train_cloud.sh </dev/null >/dev/null 2>&1 &`（**setsid 偶尔 exit1,本地 nohup 也行；云端 ssh 内 setsid OK**）。
3. ckpt 落 `/root/autodl-tmp/wallx-outputs/{0,1,2,3}=epoch / {epoch}_{gs}=step`。

**sweep 边训边评**：见 [[wallx-eval-serving-adapter]] §sweep。已测 gs1.0k/1.2k=0%(早期正常)，supervisor 自愈中。
mihomo 代理 `/root/mihomo -f /root/mihomo-min.yaml` 切卡后重启。
