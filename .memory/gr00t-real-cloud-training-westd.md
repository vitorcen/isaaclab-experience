---
name: gr00t-real-cloud-training-westd
description: GR00T-N1.7 真机30集pick-orange云端训练(westd PRO6000)踩平的6个bring-up坑+起训配方;train_real.sh脚手架
metadata:
  type: project
---

> ⚠️ 本文按时间线累积:「待办」「重训计划」等段已被后文「✅ 重训 v2 完成」「🧪 v3-simwarm」取代,真机排序以 [[so101-real-demo-recording-and-act]] 终榜为准。

GR00T-N1.7 真机 30 集 pick-orange 云端训练(2026-07-10,AutoDL westd 克隆机)。见 [[so101-real-demo-recording-and-act]] flowdp 段、[[gr00t-n17-v4-v5-recipe-align-live]] 配方、[[wallx-autodl-cloud-training]] 云端纪律。

## 决策:上云非本机
本机 GR00T = 27h RAM-bound + AV1 崩溃风暴(见 [[so101-real-demo-recording-and-act]]);云端 PRO6000 big96 = **~4-6h**。用户选云端。

## box = westd 克隆机(**RTX PRO 6000 Blackwell 96G VRAM + 110G cgroup RAM**)
`connect.westd.seetacloud.com`,端口每次开机变(问用户),密码在用户手上非 pass(pass 里是旧的)。克隆自带 `dependencies/Isaac-GR00T/.venv`(py3.10.16 uv-editable)+ warm base `groot_n17_base`(6.5G 完整 Gr00tN1d7,含 Cosmos 骨干)+ `hf_cache`(含 Cosmos 完整权重)。**克隆前先无卡模式清 `pickorange_frames`(63G 旧sim帧缓存,可再生)**——克隆更小。GPU 96G→big96 profile(真 batch48/adamw/无梯度累积,快)。

## ⚠️ 6 个 bring-up 坑(全是真 bug,按此顺序踩平)
1. **`uv: not found`**:非交互 SSH 的 PATH 缺 uv → `export PATH="/root/.local/bin:$PATH"`(见 [[wallx-autodl-cloud-training]] 同类)。
2. **`checkpoint_files[0]` is None**:AutoDL 克隆重置了 `/`(overlay)→ 默认缓存 `~/.cache/huggingface` 的 Cosmos-Reason2-2B **只剩 config 没权重**(4.87G safetensors 丢);autodl-tmp/hf_cache 那份完整 → 拷完整缓存进默认位 + `HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub`。
3. **`OfflineModeIsEnabled`**:GR00T 骨干 `Qwen3VL.from_pretrained` 的 `transformers_loading_kwargs={'trust_remote_code':True,'local_files_only':False}`(强制在线校验)撞 `HF_HUB_OFFLINE=1` → **别用离线**(用 DBG print 抓出 tlk 才看清,隔离测试骗人:单独 load 无这些 kwargs 会过)。
4. **改代码没生效**:stale `.pyc`(traceback 行号仍指旧代码 + print 不打)→ 改 gr00t 源码后必 `find gr00t -name __pycache__ -exec rm -rf`。
5. **`401 gated repo`**:Cosmos-Reason2-2B 是**门控模型**,骨干加载要拉缓存缺的 `processor_config.json` → 需 `source /etc/network_turbo`(代理通 HF,curl 200)+ **`export HF_TOKEN=$(cat ~/.cache/huggingface/token)`**(token 在 box `~/.cache/huggingface/token`;别硬编码、别抄进文档)。
6. **`CUDAGraphs overwritten`**(DiT 动作头 backward):big96 profile 启用了动作头 torch.compile → **`export COMPILE_ACTION_HEAD_DISABLE=1`**(本机 guarded 脚本有,云端 train_real.sh 初版漏了)。

## 起训脚本 `/root/autodl-tmp/train_real.sh`(无卡准备好,GPU 模式跑)
关键 env:`PATH`+`source /etc/network_turbo`+`HF_HOME=/root/autodl-tmp/hf_cache`+`HF_HUB_CACHE=.../hub`+`HF_TOKEN=$(cat ~/.cache/huggingface/token)`+`COMPILE_ACTION_HEAD_DISABLE=1`+`GLOBAL_BATCH=48`。
DATASET=`.../datasets/v2-gr00t/so101-real-pickorange`(rsync 上传 502M,含 modality.json:SO-101 6维 single_arm[0-5]+gripper[5-6]、front/wrist AV1)、BASE=warm `groot_n17_base`、MAX_STEPS=4000(~7.6ep,523步/ep)、SAVE_STEPS=400(10 ckpt)、GPU_PROFILE=auto→big96。tmux 没装 → `setsid nohup ... </dev/null & disown`。
**AV1 patch 也要打**:box 的 `Isaac-GR00T/gr00t/utils/video_utils.py` 3 处 `num_ffmpeg_threads=0→1`(dav1d 多线程堆腐蚀,见 [[starvla-av1-dav1d-thread-leak-enomem]];实测单线程解 8/8 绿)。
**实测速度**:warmup 后 ~3-5 s/it,GPU 81%/44G,RAM 71G/110G 安全。step6 时 4.74s/it。

## 完成(2026-07-10 20:10)+ ckpt 拉取/serving
**训完:4000 步 2.71h(train_runtime 9756s),final train_loss 0.0404**。`loss_prune_top_k=2` 只留 loss 最低 2 个:**checkpoint-3600(loss 0.0088,best)+ checkpoint-4000(0.0092)**——早期 ckpt 全被剪(想要 epoch 曲线重训加 `LOSS_PRUNE_DISABLE=1`)。
**盘管理**:ckpt 完整态 24G(13G optimizer + 12G model);盘守卫 `diskguard.sh`(box `/root/autodl-tmp/`)新 ckpt 出现就剥旧的 optimizer(留 model 供 eval)+ loss-prune 双保险 → 全程 free 100-123G 稳。
**lean 拉取(用户提点=只拉变化层)**:实测**只 `action_head` 变(6.48G),骨干 6.1G 完全冻结(逐字节==base)**;但本地 sim V2 骨干是 bf16 vs 真 ckpt fp32→不能直接复用骨干。**最优解=box 侧整个 ckpt cast bf16(12.58G→5.9G 砍半)再拉**(`/root/autodl-tmp/repack_bf16.py`,含删 optimizer/训练态),比 head-delta 合并稳(无骨干配准)。box→local pull ~5MB/s→5.9G ~50min。
**真机 serving**:GR00T 跑 `lerobot-060` env(lerobot≥0.6),`run.py` 加 `GR00T_CKPT` env 覆盖(同 ACT_CKPT/FLOWDP_CKPT)→ `_ensure_groot_lerobot_ckpt` 用 `convert_groot_ckpt.py`(`GrootPolicy.from_pretrained`+save_pretrained)转 lerobot 格式缓存(`<ckpt>-lerobot`,含 policy_preprocessor.json);本地有 Cosmos 缓存+sim V2 转过=convert 可行(`lm_head MISSING` 警告无害)。**SO-101.ipynb §9**=候选 checkpoint-1200(2.3ep 较少过拟合)+ checkpoint-3600(best-by-loss)点启动 + 成功率表,横向比 ACT(6k 放入1–2只≈50%)/flowdp(9k成功)。
**待办**:3600 拉完→本地 convert→用户真机测 §9;拉完可关 box 省钱;LeIsaac 未提交(run.py GR00T_CKPT + SO-101.ipynb §9 + 前面 flowdp 那堆)。

## ⚠️ 教训:`LossDrivenPruneCallback(top_k=2)` 对 VLA 甜点选择是错配(2026-07-10 用户揪出)
loss-prune 按**训练 loss 最低**保留 → 删的恰是 loss 更高的**早期甜点**(记忆少=没过拟合=泛化可能最好),留的恰是最记忆化/最可能过拟合的晚期 ckpt。**训练 loss 最低≠开环最好≠真机最好**。这次删了 2800/3200(5.4-6.1ep,甜点嫌疑区)永久丢失。**后续 GR00T 云端训练默认 `LOSS_PRUNE_DISABLE=1`(launcher 已有开关)+ disk-guard 剥 optimizer(24G→12G,10个~120G 装得下 180G)保全 epoch 曲线 → 开环 MSE 扫全部取拐点∩甜点**(见 [[feedback-three-tier-eval-funnel]] 铁律)。
**GR00T 开环 eval**:本地 `dependencies/Isaac-GR00T/.venv/bin/python -m gr00t.eval.open_loop_eval --model_path=<native ckpt bf16> --dataset_path=<v2-gr00t 真机集> --embodiment_tag=new_embodiment --action_horizon=16 --traj_ids 0..N --denoising_steps=4 --steps=200`;metric=Unnormalized Action MSE(across trajs);train-set 口径(30集无held-out)。**实测:1200(2.3ep)MSE 20.57 / 3600(6.9ep)3.73 / 4000(7.6ep)3.07** —— 越训越低。

## 🔴🔴 决定性真机结果(2026-07-11):3600 真机**不如** 1200 = 过拟合实锤 + train-set 开环 MSE 会骗人
用户真机测:**checkpoint-3600(train-set 开环 MSE 最低 3.73、train_loss 最低 0.0088、最"收敛")真机效果**反而**不如** checkpoint-1200(欠训 2.3ep、开环 20.57)**。铁律教训:
1. **train-set 开环 MSE / train_loss 越低 = 在 30 集上越记忆化 = 真机泛化越差**。对小数据 warm-start VLA,这俩是**过拟合指标不是能力指标**,方向都反了。别用它们选 best。
2. **真机是唯一判据**(30 集无 held-out,一切离线指标只是记忆化代理)。
3. **甜点在早期**(warm-start 从**官方 NVIDIA 原生预训练头**起——⚠️2026-07-11 sha256 实锤:box `groot_n17_base` 两 shard 与 `nvidia/GR00T-N1.7-3B` 官方逐字节相同,**init 从来不是 sim 头**,别再提"换 base-init 重训"的 A/B,变量不存在;GR00T 真机弱的锅在 30 集小数据过拟合非 init——几百步就有能力,~1-3ep 就够;再训只是往 30 集过拟合)。1200(2.3ep)已比 3600(6.9ep)好 → 真甜点可能在 **400-2000 步(0.8-3.8ep)** 区间,正是 loss-prune 删掉的那批。
4. 印证了 [[feedback-three-tier-eval-funnel]] 的"开环单调降陷阱:别取全局最低(=过训),取拐点∩甜点带"——GR00T 上同样成立,且这次是**真机反证**。

## 🔁 重训计划(compact 后执行)
**目标**:关 loss-prune 留全 10 ckpt(400-4000 每 400),真机重点测**早期 400-2000**(甜点区)。
- **box**:westd `connect.westd.seetacloud.com`,端口每次开机变(问用户),密码用户给(pass 里是旧的);现无卡 2G,需切 **GPU 模式**;克隆机自带 env/base/dataset/patches(若没被删/关机重置——先探,缺则重跑上面 setup)。
- **改一处**:`/root/autodl-tmp/train_real.sh` 加 `export LOSS_PRUNE_DISABLE=1`(launcher 认这开关),其余配方不变(warm base groot_n17_base + batch48 big96 + COMPILE_ACTION_HEAD_DISABLE=1 + HF_HUB_CACHE + network_turbo + HF_TOKEN + AV1 num_ffmpeg_threads=1)。可考虑 MAX_STEPS 缩到 2000-2400(甜点在早期,不用训到 4000)。
- **disk-guard.sh** 已在 box(剥旧 ckpt optimizer);关了 loss-prune 后 10 ckpt × 12G(剥后)+ 1×24G ≈ 140G < 180G,装得下。
- **拉取**:早期 ckpt repack bf16(`repack_bf16.py`,GPU 模式 110G RAM 才够;无卡 2G 会 OOM→改拉 fp32 model 本地 cast)拉回,`convert_groot_ckpt.py` 转 lerobot,进 SO-101.ipynb §9(现有 1200/3600 格改成早期候选)。
- **横向**:同 30 集真机数据 ACT 从零 6k=放入1–2只≈50%(simwarm 续训更强=ACT新best)/ flowdp 9k 成功 / GR00T 待真甜点。
**产物现状**:本地已有 checkpoint-{1200,3600,4000}-bf16(+ -lerobot 转好);1200 是目前真机最好的 GR00T,3600/4000 过训档留证。

## ✅ 重训 v2 完成(2026-07-11,新克隆机 westd:19810,250G 盘)+ 三个可复用坑
**训练**:3600 步 2.31h(train_runtime 8317s,~1.9s/it big96),final train_loss 0.043。8 个甜点 ckpt(800-3600 每 400)全保住(400 在修 bug 前被删,0.76ep 太生无所谓)。OUTPUT_DIR=`gr00t-n17-real-pickorange-v2`(旧 run 48G 不动)。**per-ckpt 自动抽 head+delta 拉回本地**:box `head_autopull.sh`(纯 bash 完整性门→硬链接 ckpt_safe→venv-python 抽 action_head bf16→`.ready`)+ 本地 `groot_head_pull_watcher.sh`(韧性 rsync→merge)。

**坑1 — 第二个删除 callback(`CheckpointPruneCallback`,不受 `LOSS_PRUNE_DISABLE` 管)**:launcher `launch_finetune_ckpt_n17.py` 的 `_patch_save_pruning` 有独立第二个 prune,默认 `KEEP_MULTIPLE=500`/`KEEP_TEMPORARY=3`=只留「500 倍数+最近3」。存点是 400 的倍数几乎不撞 500→它把 400/800/1200/1600/2400 甜点区全删(只留 2000+2800/3200/3600)。**已根治(committed)**:①加 `CKPT_PRUNE_DISABLE=1` 开关 ②`KEEP_MULTIPLE` 默认改成 `=SAVE_STEPS`(每 ckpt 都是倍数→全永久→天然不删,省盘才显式设粗)③加 `CKPT_ARCHIVE_HEAD=1` 删前先 detached 子进程抽 head+delta 到 `CKPT_ARCHIVE_DIR`,只有 `.done` 落盘才 rmtree(=「留档才删除」,训练步零阻塞,helper=`scripts/finetune/gr00t/archive_head.py`)。**`train_n17.sh` 必须 export 这些 env**(import-time patch 读 os.environ,`uv run python` 不继承非 export 的 shell 变量,否则 KEEP_MULTIPLE 悄悄回退 500)。

**坑2 — 硬链接保命(改不了运行中进程时)**:callback 已 import 到运行进程改不了 env→用 `cp -al OUTPUT_DIR/checkpoint-N ckpt_safe/`(硬链接,0 额外字节,pruner 的 rmtree 删不掉被别处引用的 inode)。独立 guard 每 20s 扫,解耦于抽取节奏。实战验证:pruner 从 output 删了 4 个,ckpt_safe 8 个全在。

**坑3 — box SSH 传输不稳→韧性拉取**:box→本地 rsync 3.2G/head 反复「connection closed by remote host / timed out」传到一半断(1.9G 处)。原始 watcher 无重试一断即跳→只有 chunk 全到的能 merge。**修=`rsync -a --partial --append-verify --timeout=120 -o ServerAliveInterval=15` + 重试 6 次 + 本地 manifest 完整性校验(所有 chunk 在)+ 外层 while 重扫直到全 merge**。本地重建=`scripts/so101/merge_head_bf16.py`(逐 shard 换 537 个 action_head,复用 v1 checkpoint-1200-bf16 冻结骨干模板,已验证)。**nohup/setsid 起后台 watcher 反复起不来→改用 harness run_in_background 托管更稳**。

**v2 开环 MSE 表**(本地 8 ckpt,同 v1 口径 horizon16/denoise4/steps200/traj0-4):800=38.5 / 1200=17.7 / 1600=12.2 / 2000=8.9 / 2400=5.3 / 2800=5.6 / 3200=3.5 / 3600=3.7(v2 3600≈v1 3600=3.73 复现良好)。**降势 800→2400 陡降后走平,拐点≈2000-2400**;按漏斗「拐点∩早期」+ v1 真机(1200>3600)→ **真甜点疑似 1200-2000**。开环只诊断(越低=越记忆≠越好),best 认真机。**merge 后 convert smoke 验过**(800→lerobot 成功=8 个 ckpt 都能 GrootPolicy.from_pretrained load)。

**🔴 真机实测定论(2026-07-11)**:1200/1600/2000/2400/2800 真机都测过,**整体差;best=checkpoint-2400(能夹起1只橙子但没放进盘=部分成功),其余更差**。开环降势拐点≈2000-2400,**真机 best 2400 正落在拐点上**(印证 eval 漏斗「取拐点」;非最早、非最低 MSE——这次拐点比 v1"甜点在早期"更靠中,任务/run 差异)。**三头横向对比(同30集真机数据同pick-orange):ACT(判别头)> flowdp 9k(抓起1只)≈ GR00T 2400(抓起1只未放入);ACT 从零 6k=放入1–2只概率≈50%(非整任务成功率口径),基于仿真best续训的 simwarm ACT 更强。判别头(ACT)最强;flow头与GR00T头都只能抓不能稳放。GR00T-N1.7 在此真机小数据 pick-orange 是三家最弱。** SO-101.ipynb §9 已砍成单候选 2400+结果表;§7 加 ACT 开环表(25 ckpt)、§8 加 flowdp 开环表(12 ckpt)。CSV 源:`outputs/{act-so101-real-pickorange,flowdp-real-pickorange}/openloop_mse.csv`。

box 250G 盘 8 个 full(ckpt_safe)+8 个 head 全留不删,可整机关机(本地已全有 bf16)。**待 push**:launch_finetune_ckpt_n17.py/archive_head.py/train_n17.sh/merge_head_bf16.py/SO-101.ipynb。

## 🧪 v3-simwarm(2026-07-12,init=sim-81% 头,真机待测)
用户要求补"sim 头 init"真变量 A/B(v2 已实锤是官方头)。**做法=本地抽 sim best v10-4500 的 action_head(537 tensors/3.1G bf16)→传 box→`merge_head_bf16.py` 拼进官方骨干**(张量级校验:head 哈希=sim≠官方,骨干哈希=官方)→ 同 v2 配方训 3600 步(final loss 0.0174,远低于 v2 0.043=sim 头起点低)。9 个 bf16 全拉回本地 `outputs/gr00t-n17-real-pickorange-v3-simwarm/`。
**开环对比(同口径)**:v3 曲线更抖(800=61.9/1600=29.7 尖峰=sim 先验与真机数据拉扯),拐点 2000-2400 同 v2;v3@2400=3.50 vs v2@2400=5.3。**候选=2000(用户改试 2400),SO-101.ipynb §9½,真机待测**——赢 v2-2400 则 sim 头有效,平/输则坐实"GR00T 弱在 30 集小数据过拟合,换 init 救不了"。
**⚠️ 上文"三头横向对比"已被 sim-warm 战役终榜取代**:FlowDP-simwarm 7000(1.7 只/轮)> ACT-simwarm 5000(1.5)> ACT 从零(0.8)> flowdp/GR00T 从零(只抓不放),见 [[so101-real-demo-recording-and-act]] 终榜段。box 已无卡模式,产物三重备份(本地 bf16 / box ckpt_safe / head_out),可关机。
