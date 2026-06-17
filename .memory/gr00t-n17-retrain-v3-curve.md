---
name: gr00t-n17-retrain-v3-curve
description: GR00T-N1.7 PickOrange 重训 v3 拿回 epoch 曲线 + 跨卡训练经验教训——真瓶颈=容器cgroup RAM非显存(63G帧缓存须装进cgroup,5090/90G失败 PRO6000/110G成功3.5s-it util100%);成功配方=ulimit1048576首行+load_bf16=True+frame cache+per_device32+workers16;原run只留2端点ckpt无delta曲线不可重建
metadata:
  type: project
---

# GR00T-N1.7 PickOrange 重训 v3 = 拿回 epoch 曲线（2026-06-15 起）

**为什么**：原 published `wsagi/GR00T-N1.7-PickOrange` 硬盘只剩 **2 个端点 ckpt**——`checkpoint-6000`(榜单口径 5.3ep,best,已发HF)+`checkpoint-9600`(8.5ep,**从没 eval 过**),中间 epoch 全删、**没抽 head delta**(GR00T 被 `scripts/ckpt/README.md` 标"⏸️跳过:已发HF+格式摩擦")→**epoch 曲线不可重建,"best=5.3ep"无 eval 证据**。用户要重训密存中间 ckpt 找真 best。关联 [[e3a-trunc-2x20-campaign]] [[feedback-incremental-eval-during-training]] [[feedback-vla-epoch-budget-6ep]]。

## 🎴🎴 GR00T-N1.7 跨卡训练经验 + 教训（2026-06-15 实战总结，权威）
**头号铁律:GR00T-N1.7 训练的真瓶颈是「容器 cgroup RAM」,不是 GPU 显存。** GR00T 用自家 sharded loader(`shard_size=1024` 批量解码屏障)解码极重 → 必须 precache 成 npy memmap 缓存才快(17×);但 **PickOrange 缓存=63G(36293帧×2相机×480×640×3 uint8)**,这个缓存必须能装进容器 cgroup RAM(`free` 看到的是 host 不是容器!查 `/sys/fs/cgroup/memory.max`)。

**选卡标准 = 容器 RAM ≥ ~110G(装 63G 缓存+15G 模型+缓冲),不是看显存。** 显存 24G 就够(bf16 micro-8≈15-20G);大显存(96G)的价值是上大 micro-batch 拉高 util。

| 卡 / box | 显存 | cgroup RAM | 结果 |
|---|---|---|---|
| RTX 5090 (westd/westc) | 32G | **90G** | ❌ **缓存 63G+模型 warmup 尖峰超 90G→静默 OOM-kill→worker死(ConnectionRefused)+sshd饿死(SSH不可达)**;无缓存则 shard_size=1024 批量解码 9min 没第一步=不可用。**90G 装不下这缓存** |
| **RTX PRO 6000 Blackwell (westd:31762)** | **96G** | **110G** | ✅ **成功**:缓存装下(cgroup 稳 92G/110G 不OOM)+ per_device=32 吃满算力 → **3.57s/it,util 峰 99-100%,~5h** |

**✅ 成功配方(PRO 6000,`/root/run_n17_pro6000.sh`)**:`ulimit -n 1048576`(**首行!** 1024→fd耗尽崩) + `load_bf16=True`(launch_finetune_ckpt_n17.py:442;默认False=fp32 VLM前向2×浪费+多占显存,flash-attn"current dtype float32"警告是铁证) + `LEISAAC_FRAME_CACHE_DIR=...pickorange_frames`(63G npy) + `DATALOADER_NUM_WORKERS=16` + `GLOBAL_BATCH=128 GRAD_ACCUM=4`(=**per_device 32**,big96 profile no-grad-ckpt+adamw,VRAM~31G) + `SAVE_TOTAL_LIMIT=11`(全留10ckpt,150G盘够) + `COMPILE_ACTION_HEAD_DISABLE=1`。effbatch 128 保持与原 published 可比。**效率三要素=大micro-batch(吃满显存)+缓存(数据不饿)+bf16**。
**坑/教训**:① **StarVLA 不折腾是因为它用标准 lerobot 逐样本解码(97%util),GR00T 用 sharded loader 必须 precache**。② warmup 时 63G 缓存从盘 page 进 RAM(cgroup 78→92G),前~15步慢(17s/it)是冷页缓存,warm 后 3.57s/it。③ **SSH 风暴**:训练起来+kill churn 时 sshd 被 I/O/内存压力饿死(尤其 cgroup 接近上限时)→**别反复 kill/重启火上浇油,用只读探活+box端 setsid fire-and-forget 启动**(我 SSH 断了 box 照跑),见 [[feedback-shared-gpu-eval-queue-orphan-discipline]]。④ `free -g` 754G 是 host 假象,容器真限额在 `/sys/fs/cgroup/memory.max`。⑤ workers=4 防 fd 崩但喂不动 accum→71s/it 饿;正解 workers=16+ulimit修好。

## 训练 LIVE（PRO 6000 westd:31762，2026-06-15 ~16:00 启动,成功）
- box: `ssh -p 31762 root@connect.westd.seetacloud.com`(pw 会话内,sshpass -e)。RTX PRO 6000 96G / cgroup 110G / 盘150G。venv/cosmos/dataset/63G缓存/HF token 全在(数据盘从旧box拷)。
- 启动器 `/root/run_n17_pro6000.sh`(成功配方见上)。崩了重跑它(train_n17.sh 自动 resume 最新 ckpt)。OUTPUT `LeIsaac/outputs/gr00t-n17-leisaac-pick-orange-v3`,save每500=10ckpt全留。
- **实测 3.57s/it / util峰100% / ~5h / cgroup 92G稳**。下游:拉10ckpt→quick 5-round sweep→epoch曲线→best→3×20→榜单。

## 📌 延训到 5500 衔接 published 6000(2026-06-15 用户要求)
用户:"把5500也训完,和之前的6k衔接上了"→v3 曲线延到 5500 步(4.8ep)桥接 published checkpoint-6000(5.3ep)。**已改 `/root/run_n17_pro6000.sh` MAX_STEPS=5000→5500 + watchdog 完成条件 5000→5500**。**流程**:当前 run 满 5000 自动停(max_steps当时是5000)→进程退后**重跑 `/root/run_n17_pro6000.sh`(现5500)→从 checkpoint-5000 resume +500步→存 checkpoint-5500**。⚠️**resume坑**:SAVE_ONLY_MODEL=1 无 optimizer.pt→HF resume 是 warm-restart(reinit optimizer+重建scheduler到5500 fast-forward到5000,LR平滑续;同 e3a-trunc 30k→54k 套路),500步 warm-restart 影响微;若 HF 硬报缺 optimizer 则加 resume model-only。**watchdog 若已按旧5000条件退出→改后的脚本 setsid nohup 重起补评5500**。**最终曲线=500..5500(11点)+ published 6000 衔接**。

## (历史)训练 LIVE（westd 5090，失败,见上跨卡表）
- box: `ssh -p 32455 root@connect.westd.seetacloud.com`（pw 在会话 security-constraints,`sshpass -e` SSHPASS,**绝不写文件**）。RTX 5090-32G,repo `/root/autodl-tmp/isaaclab-experience`,uv venv `dependencies/Isaac-GR00T/.venv`(torch2.7.1+cu128)。
- 启动器 `/root/run_n17_v3.sh`(我写的 heredoc 脚本,避引号地狱)→ `LeIsaac/scripts/finetune/gr00t/train_n17.sh`。**配置**:`GLOBAL_BATCH=128 GRAD_ACCUM=32`(有效 batch 128=匹配原 published run,per-step micro=4 才在 32G 放得下,VRAM 22.7G)/`MAX_STEPS=5000 SAVE_STEPS=500`→**10 ckpt**(500..5000)/`LOSS_PRUNE_TOP_K=20 SAVE_TOTAL_LIMIT=20`(=**保住全部 10 个,默认 top_k=2 会删早期 epoch 毁曲线**)/`SAVE_ONLY_MODEL=1`(8.8G/ckpt,10个=88G<115G free)/`COMPILE_ACTION_HEAD_DISABLE=1`。OUTPUT `LeIsaac/outputs/gr00t-n17-leisaac-pick-orange-v3`。Trainable 877M/39.9%(冻 Cosmos VLM+训 AlternateVLDiT头),= N1.7。
- **ETA ~15.6h**(5090 micro-4 ≈0.35s/micro→accum32 ≈11s/it×5000;smoke accum-4=1.4s/it)。崩了重跑 `/root/run_n17_v3.sh`(train_n17.sh 自动 resume 最新 ckpt)。
- **启动坑(全踩过已解)**:① `uv: not found`(非交互 SSH PATH)→`PATH=/root/.local/bin:$PATH`;② HF gated 401 + huggingface.co unreachable→`source /etc/network_turbo`(开代理) + `HF_TOKEN=$(cat ~/.cache/huggingface/token)`(我设 HF_HOME 后 token 丢要显式传) + `HF_HOME=/root/autodl-tmp/hf_cache`(cosmos 缓存在此);③ **`HF_HUB_OFFLINE=1` 会让 launcher 崩**(launch_finetune 对 base 硬调 `model_info()` HF API,offline 直接 raise)→不能用 offline,只能 network_turbo 联网。base=`/root/autodl-tmp/cosmos_raw`(4.6G本地)。dataset=`LeIsaac/datasets/v2-gr00t/leisaac-pick-orange`(已转换 v3.0 格式)。

## epoch 口径（关键，别再绕晕）
榜单 `epoch = step × 32 / 36293`(用 per_device=32,**忽略 grad_accum**)。原 best 6000步=5.3ep。**5000 步 ≈ 4.4 lb-ep**(≈用户要的"约4.6,整k步")。原真实 per_device=32×accum4=有效128(需~80G卡,非本5090)→v3 用 micro4×accum32=有效128 保 step↔frames 与原一致可比。

## 🔥 GPU 利用率低的三大根因 + 修复（2026-06-15 三模型 codex+mimo 批判后锁定）
**症状**:GPU 0% 冷启动长 + 训练中仅 ~37% util,~18h。**三模型一致=几个瓶颈叠乘,非单一**:
1. **fp32 加载冻结 VLM(头号,我漏判)**:`launch_finetune_ckpt_n17.py:442 config.model.load_bf16=False`(N1.7 config 默认 False)→冻结 2.2B VLM 全程 fp32 前向(5090 上 bf16 matmul 快~6×)=2×浪费+多占显存。那个 `flash-attn current dtype torch.float32` 警告是铁证非 benign。**修=`load_bf16=True`**(冻结无梯度,bf16 安全)→ VRAM 22.7→15.3G。
2. **"冻结=免费"错觉**:冻结 VLM 每 micro-batch 仍跑全量前向(占 per-step 60-70%),micro-batch=4 喂不满 5090 208SM。
3. **CPU mp4 冷解码 = dataloader 瓶颈**(GPU 饿):**StarVLA 不卡是因为用标准 lerobot 逐样本解码(97%util);原生 Isaac-GR00T 用自己的 sharded loader(shard_size=1024 批量屏障)→需 precache**。
**✅ precache 修复(决定性)**:`LeIsaac/scripts/training/perf/precache_videos.py` 把 mp4 预解码成 **.npy memmap(uint8)**,env `LEISAAC_FRAME_CACHE_DIR` 自动 fast-path,**17× fps**。实测 westd cache-smoke(per_device8/accum1/workers16)=**5.48 it/s vs 无缓存 1.4(7.7×)**,util 46-66%,VRAM 20G。**缓存尺寸=36293帧×2相机×480×640×3 uint8=63G**(我之前 du 估 200G 是块大小高估)。盘装不下 cache(63G)+10ckpt(58G)→ **cache 放盘+754G RAM 自动 page-cache 常驻=RAM 速**(/dev/shm 容器禁remount/默认45G<63G不够)。bf16 ckpt=5.8G(冻bf16+可训DiT fp32 混合)。
4. **🔴 ulimit -n=1024 → dataloader fd 耗尽崩**:workers=16+120 npy memmap+pin_memory fd传递+socket → 撑爆 1024 → resource_sharer `ConnectionRefusedError[Errno111]` worker死→主线程卡do_wait。**修=启动脚本首行 `ulimit -n 1048576`(hard已是1048576;放`set`前+回显验证;之前放set后/`||`链没生效)**。StarVLA 启动器早有这条,GR00T train_n17.sh 漏了。
**坑:workers=4 防崩但 accum=16 喂不动→71s/it 0%util(又饿)。正解=workers=16+ulimit修好(既快又不崩)。** 全量 config 见上(run_n17_v3.sh:bf16+cache+workers16+ulimit+per_device8/accum16/5000步)。

## 🔴🔴 真根因(2026-06-15 修正):frame cache 63G > 容器 cgroup RAM → OOM 打崩 box
用户揪出"显存瞬间超高打崩"记忆([[feedback-cloud-env-reuse-disk-cleanup]]:AutoDL 容器有 cgroup RAM 上限,`free` 的 754G 是 **host** 不是容器!)。**westc cgroup memory.max=96636764160=90G**(那台记忆里是62G,各box不同,必查`/sys/fs/cgroup/memory.max`)。**我的 frame cache=63G**,warmup 时 memmap paging 63G 进容器页缓存 + 模型加载~15G + 16worker缓冲 → **瞬间冲过 90G → 容器静默 OOM-kill**(worker死=ConnectionRefused;内存抖把 sshd 饿死=SSH不可达)。容器内 dmesg 看不到 host/cgroup OOM,只能"加载处静默死"判。**这才是反复崩+SSH死的元凶,非GPU显存、非ulimit(那是次因)。**
**✅ 决定=去掉 frame cache(稳>快)**:无缓存直读 mp4 → RAM 只~25G≪90G,不OOM/不storm SSH/可管理;代价 GPU~37%util/~18h(dataloader-bound),但在训不崩。保留真修复 bf16+ulimit1048576+workers16。**StarVLA 不崩正因没这63G缓存(标准lerobot逐样本解码RAM低)**。启动器 `/root/run_n17_v3_nocache.sh`(无 LEISAAC_FRAME_CACHE_DIR)。**若想要缓存提速又不OOM:得把缓存压到≪cgroup(如训练分辨率缓存/JPEG压缩~20G,需改 precache_videos.py),当前未做**。

## ⚠️ SSH-warmup 风暴坑(2026-06-15,根因见上=cgroup OOM)
westd→westc 两台都在**训练起来后 SSH 变不可达**(连接超时数分钟+)。疑=首个 epoch 把 63G memmap cache page-fault 进 RAM + 16 worker 多进程 = I/O/内存风暴**把 sshd 饿死**,warmup 后通常恢复。**纪律=别反复 kill/重启火上浇油(见 [[feedback-shared-gpu-eval-queue-orphan-discipline]]),挂 90s 温和探活(`/tmp/westc_probe.sh`)等恢复,或 AutoDL 控制台看 GPU 曲线**。westc 端口 36429(westd 32455 已弃)。

## 🔬 最终 eval LIVE-STATE + "呆"调查(2026-06-15 深夜,compact续点)
**全 10 个 v3 ckpt(500-5000)quick 5-round sweep 完**(本地 LeIsaac/outputs/gr00t-n17-v3-curve/sweep.csv):epoch曲线 500-3500=0-20%平台→**4000=53.3% 4500=66.7%峰 5000=53.3%**(4ep开关式转折)。5500放弃(resume data-skip OOM,SAVE_ONLY_MODEL无optimizer warm-restart)。
**但 4500 strict 20-round 真值低(~40%):** run1 = 1,0,2,1,3,1,0,1,0,3(10轮12/30=40%)。**quick 66.7% 是乐观抽样**(5-round σ大);seed默认`int(time.time())`(policy_inference.py:271)→quick/strict不同随机场景不可逐轮比。
**三方查bug(codex+mimo,已结)**:无灾难bug。两个真·小瑕疵(对所有ckpt一视同仁,不破坏对比):①**placed_flags非sticky**(每步覆盖非OR,wall_cap时刻采样,已放橙子被碰掉就抹0,codex);②**581-835ms推理→sim落后wall**(mimo,GR00T同步调用Isaac不前进=非stale-obs;180s wall时sim才~80-97s→失败轮早截)。mimo的GPU降频假设**证伪**(时钟满血2595/3135MHz)。PhysX"PxGeometry"警告benign。
**⚠️ 用户重启电脑后复测:4500 fresh ep1=0/3,published 6000(已知60.6%)fresh也"有些呆"(ep1/2=1/3,1/3 wall_cap)。**

### ✅✅ "呆"调查定论(compact续点,2026-06-16):**不是退化,是指标错觉+高方差**
**根因#1=榜单 E% 是橙子率(oranges_placed/3N),不是 round 成功率**。published wsagi 60.6% = 3×20 橙子率合并(run=41/60,34/60,34/60=68.3+56.7+56.7)。用户"呆"是在盯 **round 成功率**(全3个才算)——它永远比橙子率低十几二十点,因为策略**慢**,一半轮撞 180s wall_cap 时只放了 1-2 个橙子(橙子计数照拿,round 算败)。
**铁证#2=reboot前 self 两次复测 = 橙子率 56.7% + 56.7%,正好等于 published 的 run2/run3**(gr00t-n17-self.metrics.json/.run2 各 34/60)→ **完美复现,零退化**。那两次 round 成功率只 30%/45%(高方差,wall_cap 11/20、9/20 主导)。
**铁证#3=clean run(reboot后)橙子率复现**:ep6 时 9/18=50%,在 56.7% 噪声带内。GPU 时钟满血 **2595/3135MHz,throttle=0x0,温50°C**→reboot 无硬件/性能退化;推理延迟一贯(6000 fp32 835ms)。
**结论**:① 无退化、harness 没坏;② "呆"=看错指标(round-succ vs orange-rate)+ 单次 20-round round-succ 方差 ±15%(同 ckpt 30%↔45%↔60%);③ orange-rate 才是榜单口径,稳定复现 ~56.7%;④ **v3-4500 quick 66.7% 是乐观抽样,且 4500=3.97ep < published 6000=5.3ep 本就欠训**,不顶替 published。**剩:pub6000-clean 跑完20轮确认橙子率落 ~56%(收尾确认非翻案)**。
**当前**:`pub6000-clean` 20round在跑(/tmp/pub6000_clean.log,metrics results/benchmark/pub6000-clean.metrics.json),4500 fresh已杀。
**🩸 进程组kill纪律(血泪)**:本地eval杀不净反复出僵尸+for-loop wrapper杀子进程会respawn+run_one.sh的server/conda子树→**必须 `kill -9 -<PGID>` 整组**(pgrep|kill单PID会漏children/wrapper);`fuser /tmp/leisaac_gpu_eval.lock`看持锁者;清完验GPU compute=0;**别用respawn的for-loop跑3×20,单个一次起**;load>50才急(orphan wedge)。eval本地任务必 setsid+dangerouslyDisableSandbox。

## 下游流程（标准 pipeline,训练完做）
1. **拉 10 个 ckpt 回本地 4090**(box→本地 ~5MB/s 慢,8.8G/个;或 box 直接 serve 但 GR00T eval 要 Isaac 在本地)。
2. **每 ckpt quick 5-round eval**(标准:run_one.sh gr00t,STEP_HZ=60,h=40,ep120 wall180)→画 epoch 曲线找 best。
3. **best → 3×20=60-round**(对齐榜单 top-3 口径)→更新榜单(伞仓README/LeIsaac README/STRICT)。若 best≠原 6000,榜单 wsagi 行换 v3 best。
4. **这次密存全 ckpt + 抽 head delta**(GR00T N1.7=冻VLM,head-delta 可行,虽分片 safetensors diff 麻烦)→永不再丢曲线。
