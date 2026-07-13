---
name: so101-real-demo-recording-and-act
description: SO-101真机teleop示教录制(后台+手动/休息位停)+数据集预览+ACT套圈开环选ckpt结果(6k≈50%);脚手架run.py/dataset.sh/offline_action_mse
metadata:
  type: project
---

> ⚠️ 本文按时间线累积,中段的"候选/待测/待办"多已完结——**最终结论以文末「sim-warm-start 战役收官」+「真机终榜」为准**。

SO-101 真机示教数据采集 + ACT 训练/开环选 ckpt 全链路(2026-07-09 跑通)。见 [[so101-leader-follower-hardware-setup]]。
LeIsaac 已提交(`feat(so101,dataset,eval): ...`),outputs/datasets gitignored。

## 2026-07-10 FlowHeads-DP 真机 pick-orange:训练撞 AV1 墙 + 开环选 ckpt + 真机接线(未 commit)
**血泪真凶=真机数据集视频是 AV1 编码**(`meta/info.json` features video.codec=av1)。`torchcodec`/`torchvision-VideoReader` **多线程解 AV1 时 dav1d 解码线程堆腐蚀**→症状=随机**段错误**(rc=139)+ 诡异 `'_backward_hooks'`/`'NoneType' object is not callable`/`groups` 报错(caught 或直接崩),**py3.10 和 py3.11 都中招**(与 py 版本无关,见 [[starvla-av1-dav1d-thread-leak-enomem]])。这才是本机 flowdp **训练中途每 ~5-7k 步崩 + eval 崩 + GPU 攒僵尸占满**的共同主因。**修=torchcodec `VideoDecoder(..., num_ffmpeg_threads=1)`(单线程 dav1d 不腐蚀)+ `OMP_NUM_THREADS=1`**(减 torch CPU race)→段错误 22→2/轮,单线程 AV1 解码实测 8/8 绿。**另一独立小坑**:py3.10.20 conda 构建 `import torch` 本身 ~30-40% 段错误(隔离验证,连纯 import torch CPU 版都崩;[[wallx-env-py310-torch-segfault]])→根治 py3.11;但 AV1 才是工作负载崩溃主因(py3.11 照崩)。
**训练**:`FlowHeads/scripts/train_flowdp_real.sh`(watchdog+resume+挂起检测),lerobot-v044(py3.10) micro 全跑;batch16 原生 480×640 flowdp(conv-UNet×rectified-flow,horizon16/n_obs2/n_action8),实测 **4.75 step/s**(RAM 无忧,几个 G,非 GR00T 那种帧缓存)。**撞 AV1 墙止步 step 12000(7.6ep,12 ckpt 每 1000 存)**——本机继续训=和"崩溃→僵尸堵卡→重启"死磕不值;想要 19000(12ep)得先给训练也上单线程解码或重编码 H.264。
**开环 eval env=新建 `flowdp311`**(clone `lerobot-dp311` py3.11 + `pip install lerobot==0.4.4 --no-deps`;dp311 的 torch2.7.1+cu126 与 v044 同版直接复用不重下)。三处 patch:①`site-packages/lerobot/.../modeling_diffusion.py` predict_action_chunk 加 populate_queues(DP 那个 [[lerobot-dp-async-server-bug]],新装的 0.4.4 没带);②video_utils.py VideoDecoder 加 num_ffmpeg_threads=1;③`offline_action_mse.py` 适配 flowdp(`_policy_class`+`_register_flowdp` 注册 config、`chunk=chunk_size or horizon`、**增量可续 CSV**(每 ckpt 追加、重启跳过已完成,抗崩)、**NaN 预测跳过 + nanmean 聚合**(flow 偶发 NaN 别污染整段))。sweep 脚手架 `/tmp .../flowdp_openloop_sweep.sh`(flowdp311+OMP1,崩了 resume 直到 12 行齐)。
**开环曲线(train-set,k=1,30 集)**:tf_ns 单调降 156→100→84→76→68→58→51→46→39→37→29→**25.6**(step1000→12000);夹爪 ~7-8k 才 commit(grip_miss 0.11-0.14),var_grip 全程 ~1.0(无退化)。**全局最低 12000=train-set 记忆化=过训否掉**。**候选=step 8000(5.1ep,tf_ns45.8/miss0.11,4-6ep 甜点带)+ step 11000(7ep,tf_ns29.2/miss0.07,更训版对照)**,真机才是唯一判据。
**真机启动接线**:flowdp 跑在 **lerobot 0.5.2 臂 env**(py3.12)——实测 0.4.4 训练的 ckpt 在 0.5.2 config/policy 加载+`select_action` 都 OK(无 API 漂移),真机=实时相机无 AV1 解码坑、py3.12 无段错误(两坑全绕开)。新增 `FlowHeads/flowdp/rollout.py`(仿 train.py monkeypatch `factory.get_policy_class` 注册 flowdp→转 `lerobot-rollout` main)+ run.py `flowdp-start/stop`(`FLOWDP_CKPT` env 覆盖、`_bg_start` 加 env 参数注 PYTHONPATH=FlowHeads)+ **SO-101.ipynb §8**(候选 8000/11000 点启动格 + 成功率/耗时表)。用户自己真机点启动。**待办**:FlowHeads(rollout.py 新增 + train_flowdp_real.sh)、LeIsaac(run.py+offline_action_mse.py+SO-101.ipynb §8)未 commit;真机测 8000/11000 定 best 后 12 ckpt 收敛留 best+邻居。

## 2026-07-10 进展:30 集真机 pick-orange ACT + GR00T 本机训练调查(compact 断点,待续)
**数据集**:`datasets/raw/so101-real-pickorange` = **30 集真机 pick-orange(25091 帧,~28s/集)**(套圈那批已改名 ringstack)。**未提交**的改动一堆(见文末清单)。
**ACT 结果(已完成)**:`act/train.sh` 训 25000 步(save 1000→25 ckpt),开环 `offline_action_mse.py`(已适配 ACT)全扫。曲线:tf_ns 1000→6000 陡降(308→67)后缓降到 23(=背 30 集);夹爪 step3000 就 commit(var_grip~1.0)。**工具报的 best step22000 是过训尾巴否掉**。**"突然学会"候选 = step 6000 / 7000**(6000:tf_ns67/miss0.56;7000:tf_ns60/miss0.38 夹爪更准)。已加 **SO-101.ipynb §7**:候选 6000/7000 点启动格(`ACT_CKPT=$(pwd)/outputs/act-so101-real-pickorange/checkpoints/00X000/pretrained_model bash scripts/so101/so101.sh act-start`,run.py 加了 ACT_CKPT env 覆盖)+ 成功率/耗时记录表。用户下一步真机测 6000/7000。

**GR00T-N1.7 本机训练调查(关键结论:能训但 RAM 是墙)**:见 [[gr00t-n17-v4-v5-recipe-align-live]] 配方。
- **数据要转 v3.0→GR00T v2.1**:`scripts/finetune/gr00t/convert_v3_to_v2.py`。**两个真机专属坑(已修)**:①转换 ffmpeg 用系统 `/usr/bin/ffmpeg`(lerobot env 的 ffmpeg 缺 libx264.so.138)→ 起转换时 `PATH=/usr/bin:$PATH`;②**转换器 global-vs-local 索引 bug**:真机 v3.0 每集独立 data 文件(file_index 递增),但 `dataset_from_index/to_index` 是全局累积→拿全局索引切单集文件只有 ep0 对、其余 29 集写空→GR00T `generate_rel_stats` 空 df 崩(`iloc[0] out-of-bounds`)。修=按文件算 local 偏移(减 `min(from_i for that file)`),sim(全集一文件)和真机(每集一文件)都对。已修 convert_v3_to_v2.py,重转后 30 集全非空、视频帧数对齐。转好的在 `datasets/v2-gr00t/so101-real-pickorange`。
- **env**:用 `dependencies/Isaac-GR00T/.venv`(uv-synced,train_n17.sh 内 `uv run`)——避开 [[gr00t-4090-finetune]] skill 头号坑(cloned conda env 随机内存崩)。
- **batch 48**(SOTA 冻视觉 GNS 封板值);4090 上 = **micro-batch1 × grad-accum48**(GLOBAL_BATCH=48 GRAD_ACCUM=48;micro-batch2 必 OOM)。GPU_PROFILE=small24(grad-ckpt+adafactor)+ LOAD_BF16=1 + COMPILE_ACTION_HEAD_DISABLE=1。显存 ~16G(远低 24G)。
- **RAM 是硬墙(62G 机)**:实测 **2 workers ~48G 稳(~40s/it→6ep ~35h);3 workers 峰值 ~52G 安全(~31s/it→6ep ~27h);4 workers 直接把系统搞死机重启**(帧缓存~46G+worker 缓冲超 62G,裸机页缓存回收也扛不住 spike)。**RAM 守卫必备**:`avail<4500MB` 优雅杀训练(watchdog 从 ckpt 恢复)绝不硬死机。
- **真跑脚本已备好(未启动)**:`scripts/finetune/gr00t/train_real_3w_guarded.sh` = 3 workers + RAM 守卫 + watchdog 自恢复 + 每 250 步 ckpt + 无 eval,MAX_STEPS=3500(~27h)。用户先测 ACT,之后再启动。**注:27h>「若干小时」,上云 big-GPU 同样 6ep 只 ~3-4h;本机跑=拿时间换不上云。**
- **血泪教训(重复踩)**:`pkill -f <pattern>` 当 pattern 字串出现在我自己的命令行里 → 杀自己 shell → exit 1 无输出(编排全废)。**杀训练一律按 PID 或 pgrep 交给 for 循环 kill(pgrep 自排除)**,别用会自匹配的 pkill pattern。setsid 嵌在后台调用里也打架→用 Bash 工具 run_in_background 直接跑长任务别嵌 setsid。杀 CUDA 进程留 defunct 僵尸持 GPU 显存,要 reap 其 parent 才释放。

**协作规矩(2026-07-09 用户定)**:真机**录制/策略启动都由用户自己在 notebook 触发**,我不代启动(和"push 只用户来"同理——真机是用户手上的活,手放急停旁自己起)。我只准备工作区 + 只读探活(ls by-id / detect 只读不动臂)。真机数据集名后续由用户统一重命名(套圈数据 2026-07-09 已从 so101-real-pickorange 改名 so101-real-ringstack,腾出 pickorange 名录真 pick-orange)。
**USB 掉线判据**:`/dev/serial/by-id/` 空 + 无 `/dev/ttyACM*` + `lsusb` 无 CH340(1a86)= 真·USB 脱落/掉电(非急停,急停 CH340 靠 USB 供电会留);要用户物理插线/上电,软件救不了。SIGINT 停不会打掉 CH340,只有 kill -9 会。

## 录制(`scripts/so101/run.py` + `SO-101.ipynb` §6,env=lerobot 0.5.2)
- 命令:`record-{start,stop,abort,drop,clear,info}`(经 `so101.sh` 转发);`record-run` 是内部 worker。
- **后台录**(不占 jupyter kernel):`record-start` 后台拉起 → 拖主臂做任务 → 三选一结束本集。
  一格一 kernel 的约束下,前台会占住 kernel 没法再点 stop,所以必须后台。
- **停/存三态**:①`record-stop`=建哨兵文件 `logs/so101_record.stop` → 钩子下一 tick 置 `exit_early` → 存盘(Ctrl-C 安全,后台照存);②主臂**回休息位自动停**(见下);③`record-abort`=SIGINT 丢弃当前集(不 save_episode)。
- **休息位自动停判的是主臂(leader)不是从臂**:主臂无 `max_relative_target` 限速无跟随滞后,回 home 是真实意图;从臂被限速滞后可能永远差几度不触发。钩子包 `teleop.get_action`(度制,`use_degrees=True`;夹爪是 RANGE_0_100 归一化,ignore)。阈值:离开>25°、回位<12°(8°太苛人手摆不回)、停留 0.5s;基准=开头 0.3s 中位数(抗 feetech 首读跳变)。`SO101_REST_DEBUG=1` 打 dev/armed/dwell。
- **技术实现**:monkeypatch `lerobot_record.record_loop` 包一层(数据写入零改动,避免 framework drift),只借每 tick 的 obs/action。防重入:`record-start` 探活 pidfile 活着就拒;崩后陈旧 pidfile 自动清。首连串口抖动(`no data`/`multiple access`/`status packet`)重试 3×。可选流式编码 `SO101_REC_STREAMING=1`(默认关,保持已录集编码路径一致)。
- **删集坑(已修)**:`lerobot-edit-dataset delete_episodes` 默认输出写到 `$HF_LEROBOT_HOME/<repo_id>`(HF cache)不是原地→我们目录看着"没删掉"+ cache 残留挡下次(FileExistsError)。**修=传 `--new_root`=`--root` 触发 in-place**(备份 `_old`,删后清 `_old`+cache 残留)。删到 0 集会崩→直接删目录。

## 数据集预览(`scripts/dataset/dataset.sh` + `Dataset.ipynb`)
- `dataset.sh list`=扫 `datasets/raw/*` 打印 名称/版本/**集数**/帧数;`dataset.sh viz [EP]`=起 `lerobot-dataset-viz`(rerun GUI,`mode=local` spawn 独立窗口喂完即退窗口留);`DATASET=<短名>` / `DATASET_ROOT=<路径>` 选数据集。

## ACT 套圈结果(任务=抓圈圈套橙子,数据集名暂留 so101-real-pickorange 待后续重命名)
- **10 集真机(4615 帧,577 步/epoch)** → `scripts/training/act/train.sh` shadowHokage 配方(chunk100/batch8/lr1e-5/MEAN_STD/ImageNet,env lerobot 0.5.2 读 v3.0)训 **10000 步 ~17ep,SAVE_FREQ=500=20 ckpt**。GPU 98% ~13.7步/s ~12min。
- **开环选 ckpt**=`scripts/evaluation/offline_action_mse.py`(已适配 ACT:`_load_policy` 按 cfg.type 分发,保留 XVLA)。跑法:`--val-episodes 0-9 --rename-map "" --gripper-idx 5 --k-samples 1`(ACT 确定性)。**GPU 空是正常的**——开环是 CPU/视频解码为主(~55s/ckpt,ACT 推理极小),别以为卡住。
- **best 判读**(见 `outputs/act-so101-real-ringstack/openloop_mse.csv`):train-set 开环 MSE 单调降,**全局最低=过训别用**(工具报的 step10000 要否掉)。手臂 tf_ns 快降拐点 ~step2500-3500,**但套圈靠夹爪**——只看手臂 MSE 会偏欠拟合均值预测;夹爪真动起来要 `var_grip≈1.0` + `grip_miss` 低,约 **step 5500-6500**。**best 候选=5500/6500**(手臂收敛 ∩ 夹爪 commit ∩ 未过训),合"刚快速收敛后 1-2 ckpt"直觉。
- **真机试(唯一定 best 途径,套圈无 sim)**:`ACT_CKPT=<...>/checkpoints/006000/pretrained_model bash scripts/so101/so101.sh act-start`(run.py 加了 `ACT_CKPT` env 覆盖)。**step 6000 真机 ≈ 50% 完成率**——10 集+开环选就到一半,链路成立。下一步 ROI:补数据(10 集太少)>试邻居 5500/6500。
- 收尾:20 ckpt 占盘,定 best 后按 `LeIsaac/CLAUDE.md` 收敛到 best+邻居几个删其余。

## ACT sim-warm-start(2026-07-11 开环全面正收益;2026-07-12 真机实测更强=ACT 新 best)
- **动机**:ACT 无机器人预训练,sim best 头=唯一可得运动先验(GR00T 相反:原生头已是官方预训练,sha256 实锤 init 从来不是 sim 头,"换 init"伪变量)。
- **配方**:init=`act-v040-baseline/checkpoints/020000`(sim 43.3% best),`lerobot-train --policy.path=...` 其余与从零版逐字段同(lr1e-5/batch8/30集 pickorange);**0.5.2 warm-start 自动用新数据集 stats 重建归一化处理器**(`lerobot_train.py` preprocessor_overrides),sim stats 不残留。12k 步 ~14min(4090 14.3步/s)。产物 `outputs/act-so101-real-pickorange-simwarm`。
- **开环对比**(双方同参数重扫 val=0-29/249窗/k=1;⚠️旧表 97 窗量纲不可混用):**warm 每 step 全面压制从零**,MSE 低 ~40%(warm@5000=53.9 已优于从零真机 best 6000 的 69.0),夹爪 step1000 就 commit(var 1.10/miss 0.51)vs 从零要 5000 步过 var=1.0。≈白赚 3000-6000 步。
- **坑**:`offline_action_mse.py` 默认 `--rename-map` 是 X-VLA 的(front→image),ACT 必须传 `--rename-map ""` 否则全窗 forward failed。
- **真机定论(2026-07-12,已量化)**:每轮 3 只橙子记放入数。从零 6000 = 5 轮 [2,0,1,1,0] = 0.8 只/轮(≥1只 3/5,即"放入1–2只≈50%"口径);**simwarm 5000 = 10 轮 [2,1,1,3,2,1,0,1,2,2] = 1.5 只/轮(≥1只 9/10,一轮3只满放)≈ 2× 从零 → ACT 线 best = simwarm**,开环增益变现。
- **FlowDP simwarm(2026-07-13)**:init=sim best `flowdp-eval/009800`(仿真榜45.0%)续训真机30集,`outputs/flowdp-real-pickorange-simwarm`;**真机 best = step 7000**(4.5ep,开环MSE 34.3)= 10 轮 [2,2,3,1,1,2,0,3,2,1] = **1.7 只/轮 = 56.7%,超自身仿真榜45%,反超 ACT simwarm 1.5 = 真机全场 best**(两轮3只满放,≥1只 9/10)。真机最终排序:FlowDP simwarm 1.7 > ACT simwarm 1.5 > ACT从零 0.8 > FlowDP从零(抓不稳放)≈ GR00T 2400(抓未放)。**sim-warm-start 是两线共同决定性增益;从零判别头最强,warm 后 flow 头反超**。
- **已发布**:数据集 `wsagi/leisaac-real-pick-orange`(30集 v3.0 标准parquet,卡含截图/三头结果表)+ 模型 `wsagi/ACT-Real-PickOrange`(ACT simwarm 5000)+ `wsagi/FlowHeads-DiffusionPolicy-Real-PickOrange`(FlowDP simwarm 7000,base_model=仿真版repo);各带演示mp4+Top-Cam图。LeIsaac/README.md §2½ 真机小节 + SO-101.ipynb §7c/§7½c/§8½a/§8½c/§9c 已同步。

## sim-warm-start 战役收官(2026-07-13)
- **ACT simwarm 真机变现实锤**:simwarm-5000 = **1.5 只/轮**(10 轮,9/10 有放入,1 轮 3 只满放)vs 从零-6000 = 0.8 只/轮(5 轮)→ **近 2 倍,ACT 线 best 换 simwarm-5000**,已发布 wsagi/ACT-Real-PickOrange。开环增益(MSE 低 ~40%)真机兑现。
- **FlowDP simwarm 真机变现更猛(2026-07-13)**:开环全表 warm 在 1000-10000 每步压制从零(1000 时 2×);**真机 best=7000 = 1.7 只/轮=56.7%**(10 轮 2,2,3,1,1,2,0,3,2,1;9/10 有放入,两轮满放)→ **超 ACT simwarm(1.5)成真机全场 best,且超自身仿真榜 45.0%**;已发布 wsagi/FlowHeads-DiffusionPolicy-Real-PickOrange。⚠️开环候选原推 8000(29.6)/最低点 10000(16.8),真机 best 落 7000(34.3)——开环只能圈甜点带,带内排序必须真机裁决。
- **GR00T v3-simwarm**(sim-81% 头拼官方骨干):候选 2000(用户改试 2400),§9½ 真机待测。
- **结论模式**:sim-warm 对无预训练的头(ACT/flowdp)= 白赚 2000-6000 步 + 真机更强;对有机器人预训练的 GR00T 是伪变量(v2 已是官方头)。
- **真机终榜**:FlowDP-simwarm 7000(1.7)> ACT-simwarm 5000(1.5)> ACT 从零(0.8)> flowdp/GR00T 从零(只抓不放)。**sim-warm 后 flow 头反超判别头**——与从零排序(ACT 最强)相反,也与仿真榜(GR00T 家族领先)相反。README §2½ 已录。
