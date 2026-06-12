---
name: starvla-so101-cloud-training
description: StarVLA(Qwen3-VL-4B)在 westc 云端训 SO-101 PickOrange 全配方 + 三个真坑(224 死穴硬编码/worker RAM 爆/冻结 VLM 只训 head);跑通中
metadata:
  type: project
---

**2026-06-06 启动**。第二个 VLA 路线:StarVLA 框架 + Qwen3-VL-4B 训 LeIsaac SO-101 PickOrange,
换到**第二台云机 westc**(`connect.westc.seetacloud.com:31709`,密码同 westd 存 `pass autodl/westd`;
**别写密码进 memory**)。承接 [[vla-pickorange-vision-resolution-selection]] 的 StarVLA 选型、
[[wallx-autodl-cloud-training]] 的云端套路。

## 机器
- RTX **4080 SUPER 32G**(arch 8.9,`TORCH_CUDA_ARCH_LIST=8.9`)/ kernel **5.4(超稳)** / CUDA 12.4 toolkit 在 `/usr/local/cuda-12.4`(nvcc 不在 PATH 要加)。
- 系统盘 `/root` 仅 **8.6G 空** → conda env **必须建在 `/root/autodl-tmp`**(`--prefix`,52G)。
- 容器 **cgroup RAM 上限 = 62GB**(非主机 503G);GPU 直连 github **超时**(curl CDN 假 200),用 `/root/mihomo -f /root/mihomo-min.yaml`(hysteria2,mixed-port 7890)代理。

## Env 配方(`/root/starvla_env_build.log` 全自动脚本)
- `conda create --prefix /root/autodl-tmp/envs/starvla python=3.10`(repo 要 py3.10;wallx 的 py3.10 段错误是**本地 kernel 6.17 专属**,cloud 5.4 无虞)。
- **torch 2.6.0+cu124**:aliyun pytorch-wheels/cu124 **没有** → `download.pytorch.org/whl/cu124` **经 mihomo 代理**(同 wallx)。torchvision 0.21.0 配套。
- requirements.txt:transformers **4.57.0**(Qwen3-VL 必需)、deepspeed 0.16.9、numpy 1.26.4、av 12.3.0、decord;aliyun pypi。
- flash-attn:预编 wheel `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310` 经代理 `--no-deps`。
- `pip install -e . --no-deps`。
- 启动铁律(同 wallx):**全用 env 二进制全路径,别 `conda activate`**(非交互 SSH 挂)。
- 🔴🔴 **进程持久化铁律 = 必须 tmux,别 `setsid nohup … &`**(2026-06-07 westb:27361 实测血泪)。
  这台 AutoDL 4090 上 `setsid nohup bash train.sh &` 启动的进程**随那条 SSH 会话关闭就被杀**
  (设了 setsid+nohup 也没保住)→ 表现为 `pgrep` 总抓到一个 `etime≈0s` 的新 PID、log mtime 永远
  不推进、GPU 一直 0 = 看着"启动了却没在跑"。**根治 = `tmux new-session -d -s <name> "bash train.sh"`**,
  tmux 让进程脱离 SSH 会话生命周期,SSH 断开/抖动都不影响。检查用 `tmux ls` + log mtime 推进 + GPU。
- 🔴 **AutoDL SSH（走本机 mihomo 代理 FakeIP 198.18.x）极不稳 + 易触发 sshd 限流**:
  短时间几十次快速 SSH 重试会被云机限流 → 后续大面积 `Connection timed out/closed`(误判 box 挂)。
  且**大 HF 上传(hf_transfer 129+连接)会饿死同代理的 SSH**(见 [[hf-upload-tricks]] §0.3)。
  对策:① 重试间隔 ≥5s;② 重试函数**只在"输出为空"时重试,别在"退出码非零"时重试**——否则
  `pkill`/`tmux kill-server` 返回非零会让你把整条命令(含刚建的 tmux 会话)反复重跑、自己杀自己;
  ③ launch 这种一次性操作:**短命令 + tmux** 一次成功即可,不需要长会话保活。
- 🔴🔴 **训练在固定 step 崩 `av.error.MemoryError [Errno 12]` = torchvision_av codec-context 泄漏(2026-06-07 westb PI_v3 实测，已验证修复)**。
  症状:训练每次都**精确崩在同一 step（如 1654）**，traceback 在 `torchvision/io/video_reader.py → av/codec/context.pyx:238 CodecContext.open → MemoryError`；
  **host RAM 一堆空闲（407G free）但仍 ENOMEM** = 不是 RAM 不够，是 `torchvision.io.VideoReader` 每次读视频泄漏 FFmpeg codec context（native mmap），
  累积撞 `vm.max_map_count`（即便已是 655300=默认10×，~396 maps/read 也撑不住）→ `avcodec_open2` 失败。**误诊陷阱：别赖到 RAM/rsync/模型大小头上**（我先后错赖了并发 rsync、PI_v3 更占 RAM，都不对）。
  **根因修复（零开销，已验证 1.91 it/s 不降）**：`starVLA/dataloader/gr00t_lerobot/video.py` 的 `get_frames_by_timestamps` torchvision_av 分支 `finally` 里，
  在 close container 之后加 `reader=None; gc.collect(0)`（顶部 `import gc`）——**`gc.collect(0)` 只收第 0 代**（刚弃用的 reader 环就在 gen-0，便宜；别用全代 `gc.collect()`，`load_all_data` 下全代扫描会拖垮吞吐）。
  本数据集视频是 **AV1 编码 → 只能用 torchvision_av，不能换 decord**（decord 0.6 解不了 AV1），所以必须就地修泄漏。该泄漏修复已收编为 `LeIsaac/patches/starvla/0004-pyav-codec-context-leak-gc.patch`（LeSONIC 同源 0004），云端 apply 或 scp 覆盖 video.py 均可。
  **附带认知**:codec 泄漏是 step/读次数驱动 → bs 越小（步数越多）越早暴露；westd 的 8B GR00T(bs=4)没崩可能只是它的 box/epoch 边界凑巧没触发，不代表无泄漏。

## SO-101 适配(比 plan 预估小很多)
- **repo 自带 `SO101Config`**(`examples/Franka/train_files/data_registry/data_config.py`,6-DOF joint:shoulder_pan/lift,elbow_flex,wrist_flex/roll,gripper)。
- data_mix **自动发现** `examples/*/train_files/data_registry/data_config.py`(registry.py 扫描合并 `DATASET_NAMED_MIXTURES`/`ROBOT_TYPE_CONFIG_MAP`,embodiment_tag 读 classvar)。
- 我们的接线:`examples/SO101_PickOrange/train_files/`(自包含)= `data_registry/data_config.py`(robot_type+mixture `so101_pickorange`→`leisaac-pick-orange_old`)+ `modality.json`(state/action 6 维 start/end 0..6,video primary→`observation.images.front`/wrist→`observation.images.wrist`,annotation→task_index)+ config yaml + run script。modality.json **必须拷进数据集 `meta/`**。
- 数据集 LeRobot **v2.1**:loader `lerobot_version` 默认 `v2.0` 且支持 v2.0/v3.0 双路径 → v2.1 走 v2.0 文件布局直接读。视频 **av1 codec** → backend 必须 `torchvision_av`(video.py 专为 video.av1 写;decord 解不了 av1)。

## 🔴 三个真坑(冒烟抓到,长跑前修)
1. **224 死穴硬编码**:`starVLA/dataloader/gr00t_lerobot/datasets.py:1384` `_pack_sample` 把每帧 **`resize((224,224))` 写死** → 橙子 10-40px 在喂 Qwen3-VL 前就毁了(正中 [[vla-pickorange-vision-resolution-selection]] 铁律)。**sed 改 448**(GR00T@448=68% 验证档)。第二层旋钮 = config `datasets.vla_data.obs_image_size`(`build_qwenvl_inputs`→`resize_images`),不设则只用 _pack_sample 的值。
2. **worker RAM 爆**:`dataloader/__init__.py` build_dataloader **硬编码 `num_workers=16` + prefetch 4**,16 个 persistent worker 各 fork 一份 dataset + 并行 av1 解码 → RAM 57G 撞 62G cap → `[Errno 12] Cannot allocate memory ... Retrying`。**改 num_workers=4 + prefetch=2** → 37G,Errno 归 0。
3. **冒烟进程不退占显存**:persistent_workers+deepspeed teardown 卡住,打印 "and that's all folks" 后仍占 24G GPU → 下一个训练抢显存。冒烟后**逐 PID `kill -9`** 清干净再启正式跑。

## Run 1 配方(跑通中)
- `--framework.name QwenGR00T`(GR00T flow-matching head,与成功的 [[gr00t-n17-hi-space]] 同架构,Bridge 65.3)+ base `Qwen3-VL-4B-Instruct`(8.3G,HF 经代理)。
- **`--trainer.freeze_modules qwen_vl_interface`**=冻结整个 VLM 只训 action head(GR00T-N1.7 同款,32G 稳)。模型两个顶层模块:`qwen_vl_interface`(VLM)+ `action_model`。
- action_dim/state_dim=**6**,action_horizon=**16**(QwenGR00T 默认 7/7/8 要 override);bs=8,30k steps,save 3000。
- deepspeed `ds_config.yaml` 硬编码 `gradient_accumulation_steps=1` → config 必须也 1(否则冲突报错);bf16 已开。
- 实测:**GPU 25/32G · util 97% · ~1.0s/step · loss(action_dit_loss)~1.2 稳 · ETA ~8.5-9h ≈ 6.6 epoch**。`/root/starvla_train.log`;watchdog `/root/starvla_watchdog.sh` 崩了 `RESUME=1`(trainer `is_resume` 读最新 ckpt)。

## 🔴 第4个真坑 = 磁盘满 ENOSPC(根治补丁)
每个 ckpt **8-10GB**(冻 VLM 但 trainer 存整个 4B 模型),save 无裁剪 → 4 个填满 60G 盘 → 第 4 个写一半损坏崩溃("PytorchStreamWriter failed writing"/"failed finding central directory")。**还踩了恢复坑:留了编号最高但损坏的半成品、删了完整的 → 进度全丢**。**根治 = patch `train_starvla.py:_save_checkpoint`**:① 原子保存(写 `.tmp` 再 `os.replace`,正式名永不损坏)② keep-last-N 裁剪(`trainer.keep_last_checkpoints` 默认 1,配 2)。`.bak2` 备份在云端。脚手架 `scripts/patch_save_checkpoint.py`。同理 [[feedback-training-save-policy.md]]。

## ✅ Eval-sweep 方法论(已跑通,本地 4090 评云端 ckpt)
**铁律工作流**:云端训练时**先快速 eval 一次确认机械臂会动**(降 save_interval→首 ckpt ~step500 提前出),再开 **auto-pull-back + 每 ckpt GUI eval sweep**。验证看的是"**arm 动不动**"(motion/std>0,非 0% 成功率)——undertrained ckpt 成功率必 0 但能验证整条链路。
- **架构**:ckpt 留云端不行(用户要"拉回来")→ 每 ckpt rsync 10GB 回本地 4090,本地 serve + Isaac **轮流用同一张卡**(serve ~9G + Isaac ~6G < 24G),**GUI 模式 DISPLAY=:0**(用户要亲眼看),非 headless。
- **本地 env**:`conda create --clone wallx -n starvla_eval`(py3.11+torch2.6+flash-attn 现成)→ 升 transformers 4.57 + `pip install -e starVLA --no-deps`;**坑=`--no-deps` 漏装一堆**,serve import 链逐个炸:`rich`(overwatch dictConfig 'console' handler)→`pipablepytorch3d==0.7.6`(transform)→ 还有 diffusers/timm/decord/albumentations/tiktoken。
- **serve_starvla.py**(`LeIsaac/scripts/evaluation/`,starvla_eval env):`baseframework.from_pretrained(ckpt.pt)` 需 `<run_dir>/{config.yaml,dataset_statistics.json}`+config 的 `base_vlm` 重指本地 Qwen3-VL-4B(serve 启动时 sed 重写);内联 ~30 行 websocket server(标准 msgpack-numpy,同 [[wallx-eval-serving-adapter]] 协议);obs **stateless**(训练样本无 state)+ 2 cam **强制 448**(`--img_size 448`,不让 224 死穴在 eval 重现);action 出 motor-deg→弧度复用 wallx 转换。
- **StarVLAServicePolicyClient**(`service_policy_clients.py`,照抄 WallX)+ `policy_inference.py` 加 `--policy_type=starvla` 分支 + preprocess 列表。
- **starvla_sweep_watcher.sh**(适配 wallx_sweep_watcher):轮询 westc ckpt → rsync(.pt file,非 wallx 的 dir)→ serve → GUI eval(EVAL_ROUNDS=3/EP=60s)→ CSV → 本地裁剪留最新 3。`pass autodl/westd` 密码 westc 通用。CSV `LeIsaac/outputs/starvla-sweep/sweep.csv`。
- **2026-06-06 首验证**:`steps_500` GUI eval = arm 动(retracted_middle,142.9s,0/3 橙子)+ 服务端 smoke action(1,16,6) per-joint std>0 → **链路通,用户确认机械臂会动**。
- ⚠️ **节奏权衡**:10GB/ckpt pull ~12min + GUI eval ~5min = 17min/ckpt;save_interval=500 产 8min/ckpt → watcher 追不上会漏(被云端 keep_last 裁掉)。要"每个 ckpt 都评"则 save_interval≥1500(RESUME 续,原子 save 不丢)。
- 🔴 **watcher serve-就绪检查 bug（两层）**:① overwatch `dictConfig(disable_existing_loggers=True)` 禁用 serve_starvla 模块 logger → "serving StarVLA on ws" ready-line 永不打印 → watcher grep 不到误判 serve FAILED。**修=serve 用 `print("SERVE_READY",flush=True)`(print 不受 logging 配置影响)+ watcher 就绪检查改 grep SERVE_READY 或 `ss -tln | grep :$PORT`**。② 修完仍连续 FAILED 的真因 = **双 watcher 抢同一 8002 端口**(每次 nohup "重启" 旧 watcher 没真死)→ 第二个 `websockets.serve` 撞 address-in-use + 两者互写 `/tmp/sv_serve.log` 混乱。**根治=重启前必 `pkill -9 -f starvla_sweep_watcher` 再 `ps -eo pid,cmd|grep [s]tarvla_sweep` 确认 0 个真进程(pgrep 会自匹配命令行,要用 `[s]` 或 ps 排除自身)**。实测 serve 暖缓存 **~13s** SERVE_READY+端口就绪,远快于 150×2s 窗口,只要单实例就稳。

- 🔴🔴 **昂贵教训:双重裁剪夹死峰值 winner（永久丢失 15000/12000）**。两处**独立** prune 同时跑:① 云端训练 `keep_last_checkpoints=2`;② **我自己在 `starvla_sweep_watcher.sh` 加的"本地留最新 3"**（怕 14×10GB 撑爆盘 = 过度设计，本地实际 493GB 空闲）。后果:sweep 认定峰值是 15000，但等想回头 strict eval 时它**已被本地 newest-3 和云端 keep_last=2 两头删光**，两边都没了 = 不可恢复。**铁律:① sweep 期间本地永不自动删 ckpt（盘够就全留，清理只在 strict 定 winner 之后手动做）；② 真要省云端盘就 keep_last≥4 给 watcher 留追赶余量；③ 每出一个"当前最优"ckpt 立即 `cp` 到 rescued/ 锁住**。已在 watcher 里删掉本地 prune 段。

- 🔴 **sweep 评测参数偏离 leaderboard 标准（数值偏低不可比）**。`starvla_sweep_watcher.sh` 当时用 `episode_length_s=60 + max_round_wall_s=0(关) + 无 stuck`，但**权威 `LeIsaac/scripts/benchmark/run_one.sh` = `EPISODE_LENGTH_S=120 + MAX_ROUND_WALL_S=180 + STEP_HZ=30`**，且**两个超时一起用**（仿真 120s 或墙钟 180s 先到者结束）。后果:① 每轮只给 60 仿真秒 = 标准一半，橙子没放完就被终止 → sweep 成功率**系统性偏低**；② sweep 数不能直接进 leaderboard。**stuck 检测**:run_one.sh 只对 `lerobot-act|lerobot-diffusion` 关（99999/0），**其余 VLA（含 starvla/gr00t/wallx）开 `stuck_window_s=30 stuck_eps_rad=0.05`** → 过拟合"晃动"会在 30s 内判 stuck 提前结束（leaderboard 预期行为）。**铁律:任何要对外/进 model card 的 eval 必须用 run_one.sh 全套参数；watcher 快筛可以用宽松值但要标注"非标准口径"**。已用 120/180/30 + stuck30/0.05 重评 18000。

## 📊 Sweep 结果 = 倒 U 过拟合曲线(2026-06-06，3-round/ckpt，⚠️非标准口径 ep_len=60/无墙钟）
峰值在**中段 ~15k 步**，不在末尾。60 demo 冻 VLM 只训 head 训太久必过拟合：

| step | 整轮成功 | 橙子放置 | |
|------|------|------|--|
| 10500 | 0/3 | 3/9 | 欠训 |
| 12000 | 1/3 | 5/9 | ↗ |
| 13500 | 1/3 | 3/9 | 平台 |
| **15000** | 1/3 | **6/9** | **峰** |
| 16500 | 1/3 | 4/9 | 平台 |
| 21000 | 0/3 | 1/9 | 塌陷 |

- **过拟合表征**：手臂晃动、悬停、不果断闭夹爪（用户肉眼可见）= flow-matching head 背了训练轨迹，新状态只吐流形均值 + 每 16 步 chunk 互打架。
- **3-round 区分不开**（整轮全是 1/3），靠 oranges_placed 细分 → 15000 最优；**定胜负必须 strict 20-round**（见 [[feedback-20round-strict-benchmark]]）。
- 📤 **已发布（2026-06-06）**:18k ckpt → `https://huggingface.co/wsagi/StarVLA-PickOrange`;两个 README leaderboard 表（root `README.md` rank7 + `LeIsaac/README.md` rank7/14 行，含 `starvla` policy_type 备注 + finding + 截图 `doc/images/` 与 `LeIsaac/docs/assets/starvla-pickorange.jpg`）已加 35.0% 行。formal CSV 留底。
- 🏁 **最终战绩（2026-06-06，现存最优 = steps_18000，15000/12000 已丢）**:正式口径 **strict 20-round = 2/20 (10%) 整轮成功 / 21/60 (35%) 橙子**，avg 169.8s。**3-round 当时是 1/3 (33%) = 方差幻觉，真值 10%（3 倍偏差，铁证 20-round 必要性）**。19500 正式 3-round 已掉到 0/3、2/9（过拟合）。**判定:偏弱负面档**（leaderboard 弱档，Wall-X 官方 40-65%）；非硬失败——35% 橙子率说明在认真抓放，主死因是**慢**（17/20 轮 180s 墙钟超时放不完 3 个），根因 60 demo + 冻 VLM 只训 head 容量/数据双不足。formal CSV: `LeIsaac/outputs/starvla-sweep/formal_eval.csv`。
- **2026-06-06 18:30 云端按用户"下一个 ckpt 出来就停"在 step 25500 停训**（明显过拟合无意义再训，省 GPU 费）。云端留 [24000, 25500]。
- 🔴 **停云端训练的坑**：① 必须**先杀 watchdog 再杀训练**（否则自动 resume）；② `pkill -9 -f train_starvla` **杀不净 deepspeed worker**（accelerate launcher 死了子进程还在占 GPU 100%）→ 要按显式 PID kill + `pkill -9 -f deepspeed`；③ 训练满载时 **load avg 23-29 → ssh 间歇 255 连不上**，重试几次 + `ServerAliveInterval` 才进得去。
- ⚠️ 云端 box 别急着关机：本地 watcher 还要拉 24000/25500，拉完评完再释放实例彻底停费。

## 📍 当前状态(2026-06-06 收尾完成)
- ✅ **第一轮 QwenGR00T 全流程闭环 + 已发布**:云端 step 25500 停训 → 本地 strict 20-round 选定 winner=18000(35.0%/P(3)=10%) → HF `wsagi/StarVLA-PickOrange` 已传(9 文件 9.98GB,逐文件 commit) → root + LeIsaac README leaderboard rank7 已加。formal CSV `LeIsaac/outputs/starvla-sweep/formal_eval.csv`。
- **本地资产**:`LeIsaac/outputs/starvla-sweep/rescued/` 留 18000/19500/24000/25500 四个完整 ckpt(备份);HF staging `hf_upload/`(ckpt 硬链接,可删)。云端 westc 训练已停 GPU 0%,**用户随时可释放实例**(数据已全拉回)。`pass autodl/westd` 密码(别写 memory)。
- **未提交**(用户自己 commit):`serve_starvla.py`/`starvla_sweep_watcher.sh`(已关本地裁剪 + 对齐 run_one.sh 正式参数)/`smoke_starvla_client.py`/`service_policy_clients.py`(+StarVLAServicePolicyClient)/`policy_inference.py`(+starvla 分支)/root+LeIsaac `README.md`(+rank7 行+截图)/`doc/images/`+`LeIsaac/docs/assets/starvla-pickorange.jpg`/`workspace/starvla_so101/`/`.memory/`;云端 patch(datasets.py 224→448、__init__.py workers 16→4、train_starvla.py atomic+prune)有 `.orig`/`.bak2` 备份。

## 第二轮杠杆（compact 后做：Gemma4+PI_v3 + 提分）
- 🔑 **第一轮亏在跳过机器人预训练**:用裸 `Qwen3-VL-4B-Instruct`(冻)+ GR00T head **从随机初始化**硬学 60 demo（config 无 `pretrained_checkpoint`）。35% 偏弱一大主因。
- **base_model 现实**:StarVLA 官方在 Bridge+Fractal 预训练的 VLA base **只有 Qwen 系**有（model_zoo:`Qwen3VL-PI_v3-Bridge-RT_1`=69.8 / `Qwen3VL-GR00T-Bridge-RT-1`=65.3 / `Qwen-GR00T-Bridge`=71.4）。**Gemma4 / Cosmos / MiniCPM 骨干没有官方 Bridge base** → 那些路线只能裸骨干 + 头从零训，起点天然吃亏。
- ⚠️ **用户想训的 Gemma4+PI_v3 当前无现成框架文件**:仓库有 `Gemma4PI.py`(v2 PI) + `QwenPI_v3.py`(v3 绑 Qwen)，**无 `Gemma4PI_v3.py`**。要么用现成 **Gemma4+PI(v2)**（Gemma4 backbone LIBERO 96.0% 已验证，examples/Gemma4/README），要么把 PI_v3 改进（参数压缩 +(vl/dit)²、VLM attend state）移植到 Gemma 骨干（工作量大）。
- **最高 ROI 路线（建议）**:从 **`StarVLA/Qwen3VL-PI_v3-Bridge-RT_1`**(69.8,已下载) 做 `trainer.pretrained_checkpoint` init(`load_pretrained_backbones`)→ 拿机器人预训练 + PI_v3(π0.5 设计)起点。注意 action head 7-dim EEF vs 我们 6-dim joint 维度不匹配要 `reload_modules` 跳过。再叠 obs_image_size 升 480+ / 解冻 VLM 顶层 N 层 / 加 demo 60→100+。
