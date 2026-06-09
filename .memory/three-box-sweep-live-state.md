---
name: three-box-sweep-live-state
description: StarVLA PI_v3 多骨干 sweep 终态(2026-06-08)+ compact 后 4 任务接续:新env训DP/续训2B/开训9B/监控4B
metadata:
  type: project
---

# StarVLA PI_v3 sweep — 终态 + compact 后任务(2026-06-08)

## ✅ 已定案结果(写进 README leaderboard + 发布)

- **PI_v3 head > GR00T head 实锤(8B 档)**:`wsagi/StarVLA-Qwen3-VL-8B-PI_v3-PickOrange` **已发布**
  (18.91G ckpt + card + 视频全在 HF)。**step-78000 = 4.3 ep,strict 20-round = 63.3% (38/60, P3=40%, P≥2=55%)**,
  **rank 3**(hi-space N1.7 66.7% 之下,反超 N1.5 58.3% 与同骨干 GR00T head 53.3%)。
  ⚠️ 20-round 必要性铁证:3-round 里 66k 虚高 77.8% 是噪声(20-round 真值 50.0%),真峰 78k。
  raw `[1,2,3,2,1,1,1,1,1,2,3,3,3,3,1,1,3,3,3,0]`。
- **Cosmos-Reason2-8B backbone = 负面**(已杀 westd 训练):6k-48k 全 0,唯 12k 闪 22% 噪声。
- **旧 Qwen3-VL-2B GR00T = 18.3%**(已发)。
- ✅ **Qwen3.5-2B PI_v3 已发布**:`wsagi/StarVLA-Qwen3.5-2B-PI_v3-PickOrange`(9 文件齐,ckpt 5009MB)。**best=step-27000(3.0ep)strict 20-round=43.3%(26/60,P3=15%,P≥2=50%,avg160s)**,raw `[0,2,1,0,0,2,2,2,2,3,0,0,2,3,1,0,2,0,1,3]`;35000(3.85ep)已过峰=35.0%。README 两表已插 rank8 + 重排 + 分析注解 + 计数。demo webm→mp4 已转。⚠️**HF 大文件传必关 `HF_HUB_ENABLE_HF_TRANSFER=0`**(hf_transfer finalize 卡死,关掉用标准上传器+resume partial 秒成)。
- ✅ **DP(diffusion)4ep 20-round = 8.3%(5/60,P1=25%,P3=0,P≥2=0,avg182s)**,raw `[0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1]`。**非零(>榜单旧 0.0%)→ 待更新榜单 DP 行 0.0%→8.3%**(DP=lerobot-diffusion h16;8.3%>X-VLA 6.7% 故 DP 上移一位)。DP epoch-sweep 3-round:3ep0/9·4ep1/9·5ep未出·6ep0/9 → 4ep 最佳故选它做 20-round。**serving 根因已修**:`run_one.sh→start_server.sh` 默认 `envs/lerobot`(未打 patch)→ n_obs_steps=2 撞空队列 `'observation.images'` 报错 → 臂不动假0;修=`LEROBOT_PYTHON=/home/david/miniconda3/envs/lerobot-v040/bin/python`(patched editable,有 predict_action_chunk populate_queues)。见 [[starvla-checkpoint-resume-migration]]。
- README Epoch 列 + 口径已加(`epoch=step×global_batch/36293`,见 [[feedback-vla-epoch-budget-6ep]])。

## ✅ 4B/9B 选型完成 — PI_v3 家族终榜(2026-06-09)

**Qwen3.5 PI_v3 family 20-round strict 最终(已写进两份 README leaderboard 表+分析):**
- **4B/21000(4.6ep)= 46.7%(28/60,P3=20,P≥2=45,env25%,147s)👑4B胜出** raw `[2,1,0,3,1,0,2,0,3,0,1,1,1,2,2,3,2,3,0,1]`。三方 19k/20k/21k=46.7%/31.7%/46.7%(19k与21k打平,21k靠env胜)。
- **9B/10000(4.4ep)= 45.0%(27/60,P3=20,P≥2=50,env25%,125s)🥇9B胜出** raw `[2,2,0,0,1,3,3,2,0,1,2,0,3,3,2,2,1,0,0,0]`。5rd筛9k-13k→top-2(10k/9k)各20rd,10k与9k又27/60完全打平;13000(5.7ep)=43.3%。
- **结论**:**参数饱和~46% — 2B(43.3%)→4B(46.7%)视觉红利+3.4点真,4B→9B翻倍几乎零增益(冻VLM时head表达力是天花板)**;4-4.5ep sweet spot;quick跨度差几乎全是噪声(4B quick 21k=66.7%vs19k=55.6%,20rd双双收敛46.7%→必须5rd筛+20rd定)。
- 榜单:README rank7(4B)+rank8(9B);LeIsaac同。⚠️**这俩PI_v3 ckpt未上HF**,行标 `local, HF pending`(待用户定是否发布,发了再加链接+card)。**不替换**现有 `StarVLA-PickOrange`(Qwen3-VL-4B GR00T 35%)行=不同模型。

**本轮修的真bug**:`starvla_strict_eval.sh` 对 ROUNDS≠20 自动加 `_r5demo` 后缀写 `strict_eval_q8_r5demo.*`(防覆盖canonical);旧orch归档却抓stale `strict_eval_q8.json`(=13000旧20rd)→ 9B 5rd归档全字节相同43.3%垃圾。修=归档抓正确源`_r5demo.*`,每步跑完立即快照到`.r5.s<step>.*`(`/tmp/run_9b_screen_fix.sh`)。**教训:5rd/非20rd的strict结果在带`_r5demo`后缀的文件,别抓canonical**。

## ✅ 已完成发布(2026-06-09)
- **2B**:`wsagi/StarVLA-Qwen3.5-2B-PI_v3-PickOrange` best=27000@**43.3%**(P3=15/P≥2=50),9文件齐+README 两表+分析注解。
- **DP**:`wsagi/DiffusionPolicy-PickOrange` 更新到 4ep(step-18000)=**8.3%**(model+README+新图全更新);榜单 0.0%→8.3%。**DP serving 必 `LEROBOT_PYTHON=lerobot-v040`**(patched populate_queues);HF 大文件坑见 [[hf-upload-tricks]] §0.07(`HF_HUB_DISABLE_XET=1`)。
- 工具:serve-hang 检测+重测 `scripts/benchmark/{flag_serve_hang,merge_valid_episodes}.py` + `starvla_strict_eval.sh` RETEST=1。
- **eval_queue**(4090)已重启正常后台跑(单实例按PID查勿pkill自匹配)。**puller已停**。

## 🟡 各机终态(全部待用户控制台**关机**,别释放)
- **4B-bjb1:22263**:训停 23000=5ep;heads 1000-23000 全本地;resume 料齐(head+`vlm_base_qwen35_4b.pt` 9.1G 本地)。
- **9B-clone:52021**:训停 13000=5.73ep(13610 不存,见 [[starvla-checkpoint-resume-migration]]);heads 1000-13000 全本地;resume 料齐(head+`vlm_base_qwen35_9b.pt` 18.8G 本地)。
- **westc(2B)/westb(8B)/westd(Cosmos)**:早完成。**关机保盘可续训,释放清盘**。
- 本机 `_head_sweep_tools/` vlm_base 全在:8b/2b/4b/9b/cosmos。

## 🟡 各机终态

- **westb(PI_v3-8B)**:训完 90k,15 head 全拉回,**待关机**。
- **westd(Cosmos)**:训练已杀,**待关机**。
- **westc(Qwen3.5-2B)**:**正常跑完 30k=3.3ep**(非崩),GPU idle。完整 `steps_30000_pytorch_model.pt`(6.27G)在盘。
  **建议切无卡模式**保 ckpt+env 待续训(别关机)。2B sweep 弱(5k-30k 多 0-22%)。
- 本机 eval 料齐:`LeIsaac/outputs/_head_sweep_tools/` 有 vlm_base_8b.pt / vlm_base_qwen35_2b.pt / **vlm_base_qwen35_4b.pt(9.1G)** / vlm_base_cosmos.pt。

## 🔜 compact 后 4 任务

1. ✅ **新 env 训 DP(2026-06-08 跑通)**:**py3.11 根治确认**——`conda create -n lerobot-dp311 python=3.11` + torch2.7.1+cu126(aliyun pytorch-wheels **扁平HTML目录非PEP503,必须直连 wheel URL** `https://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.7.1+cu126-cp311-...whl` `--no-deps`,再 `pip install -e /home/david/work/lerobot-v040` 补 nvidia CUDA 库)+ 锁 diffusers0.35.2/datasets4.1.1/av15.1.0/torchcodec0.5。**import 8/8 全过**(py3.10 是 40-87% 段错误);**训练越过 step200**(py3.10 堆腐蚀崩点)→ 0.121s/step ~8/s,**DP~6G 与 eval_queue 共驻 4090 无 OOM**。脚手架 `/tmp/build_dp311_b.sh`(建env)+`/tmp/dp_6ep_311.sh`(训:NW=4 batch8 steps27220 save4537=每ep)。eval 3/4/5/6ep 验"早epoch DP 是否脱 do-nothing"(DP 训22ep 仍0/60)。数据集 **v3.0** `LeIsaac/datasets/raw/leisaac-pick-orange`。**✅ 2026-06-08 改造为 eval_queue 协作填空(解决单4090争用)**:`eval_queue.sh` 加 `dp_chunk()`(无 starvla 待评时 resume DP +1 epoch,`--config_path=.../last/pretrained_model/train_config.json --resume=true --steps=<下个epoch边界>`,**--steps 覆盖+resume 实测生效**:从 9074/2.03ep 续 loss 0.026 不重训)+ `dp_eval()`(epoch ckpt≥3ep 经 lerobot-v040 patched async server :8080 + `scripts/benchmark/run_one.sh ... lerobot-diffusion 8`,best-effort 不崩队列)。主循环 `dp_eval || dp_chunk || sleep`,**starvla 永远优先**(下一轮 preempt),DP 只填 ~9min/epoch 空隙→零 OOM/不黑屏/DP 自然到 6ep。DP 训用 lerobot-dp311(py3.11)、服务用 lerobot-v040(patched)。ckpt `LeIsaac/outputs/dp-6ep-earlytest/checkpoints/`(4537/9074 已有,训13611=3ep)。备份 `eval_queue.sh.bak_pre_dp`;`DP_ON=0` 或 `touch dp-6ep-earlytest/.dp_off` 禁用。**⚠️ DP resume 段错误坑(2026-06-08 踩+修)**:`video_backend=torchcodec` + NW=4 在 kernel-6.17 上 resume 后跑 ~100-300 步堆腐蚀**段错误**(py3.11 修 import 不修运行时 torchcodec 解码),队列疯狂重试自旋 16 崩/28 试。尝试 `--num_workers=0 --dataset.video_backend=pyav --save_freq=500` + "有进展重置失败计数"——**仍败**:resume 在加载/早期步就腐蚀,既出段错误(rc=139)也出 `UnboundLocalError: _parameters`(torch 模型加载变量错乱,同一堆腐蚀的另一张脸),到不了 500 步存盘。**结论:DP resume 在本机 kernel-6.17 上根治不了**(fresh 0→9074 能跑是运气/曝露短;resume 重载必腐蚀)。**dp_chunk 连续 4 次无进展自动 `touch .dp_off` 禁用生效**,队列回归纯 starvla 不受影响。**DP 停在 2ep(9074),要补 3-6ep 只能搬云端**(westc/bjb1 kernel 5.15 稳)**或直接放弃**(22ep=0/60 负面已够)。统一队列架构+抢占+安全机制全部验证 OK,只是 DP filler(epoch 大 chunk resume)这台机器跑不动。⚠️ 多实例坑:`pkill -f eval_queue.sh` 自匹配杀不净→残留旧队列+新队列并存双 dp_chunk 抢卡,要按 PID 杀。
- **✅ DP 攻克法(2026-06-08,FastWAM watchdog 模式 [[feedback-training-save-policy]]):** kernel-6.17 让 resume 在 ~100-300 步崩,但**只要 save_freq(100) < 崩溃间隔,崩前就存住了进度** → watchdog 从最近临时盘续 → **每次崩只丢 ≤100 步(~30s)**,一点点爬到 4ep。脚手架 `/tmp/dp_watchdog.sh`(TARGET=18148=4ep;fresh 分支必须先 `rm -rf $OUT` 否则 lerobot 撞 FileExistsError;NW0+pyav;每 100 步存+keep-last-3+epoch 边界归档 `dp-grind/keep/` 当正式 ckpt;失败 15s 退避;连续 30 次无进展放弃)。本机 fresh-grind(save_freq=4537 大盘)失败的根因就是**存盘频率 > 崩溃间隔**,崩前没存住。**关键洞察:本机不可 resume 的工作流,把 save_freq 压到 < 崩溃间隔就能用 watchdog 续命。**
2. ✅ **续训 2B(已启动 2026-06-08)**:见上"westc 2B 续训中"。MAX_STEPS=41000=4.52ep,ETA ~6.1h。新 head puller 自动拉、queue 自动评。盯曲线看 3.5-4.5ep 有没有反超 3.3ep 旧峰。
3. ✅ **开训 9B(2026-06-08 smoke 中)**:克隆机 `connect.bjb1.seetacloud.com:52021`(**SSH 密钥免密**,克隆带走 authorized_keys;known_hosts 冲突用 `-o StrictHostKeyChecking=accept-new`;密码独立**不写 memory**)= **A800×2-80G**。已清 4B 残留(38→7.7G)+ 下完 Qwen3.5-9B(19G/4shard,text_hidden 4096/32层,`/root/autodl-tmp/models/Qwen3.5-9B`)+ 写 `so101_qwen3_5_9b_pi_v3.yaml`(base_vlm→9B,vl_hidden 4096)+ `run_train.sh` 参数化(DEVICES/NUMPROC/GC,克隆机专属)+ ds_config batch=auto。**bench 实测 batch8/GC-on=2.95s/it,峰值 76.4G/80G(93%触顶)→ batch16 & GC=off 必 OOM,∴ 最高效超参=batch8/GC-on**(global batch 16=5.4samples/s)。脚手架 `/root/{bench_9b,smoke_9b,prep_9b,train_9b_full,extract_heads_qwen35_9b}.sh`。9B 6ep=13610步,**save 每1000**(对齐sweep)。**eval 管线已通(2026-06-08)**:本机已有 Qwen3.5-9B→建 `vlm_base_qwen35_9b.pt`(18.8G,760+600=1360keys;`extract_vlm_base.py` 需 run_dir 有占位 ckpt 文件+config.yaml+dataset_statistics.json+modality.json;**merge/extract 要用 env python 直跑别用 conda run**——conda run 会吞参数报 `/vlm_base...`)。eval_queue FAM 加 `qwen35_9b|...|q8bit=8|min_step=4000|...`、puller SRC 加 `qwen35_9b|52021|...`(key auth+StrictHostKeyChecking=no 自动接受克隆新host key)。**smoke_500(0.22ep)eval=0/3橙但跑满180s无崩/NaN→端到端管线validated**(8bit serve 13.6G+Isaac+merge 全通);能力等全量 sweep。⚠️ **双进程 RAM 坑(2026-06-08 踩)**:smoke step338 崩 `av.error.MemoryError [Errno 12]`(PyAV dataloader)——非 GPU OOM,是**容器 cgroup mem=240G**,而 `load_all_data_for_training: true` 在 num_processes=2 下**各进程预载 1 份数据集→2×超 240G**(4B 单进程 1 份压线过)。**修=config `load_all_data_for_training: false`(流式,data_times 本就0)+ `dataloader/__init__.py:52` num_workers 4→2/进程**(双进程共4解码器)。是 [[starvla-so101-cloud-training]] "16workers爆RAM改4"坑的双进程升级版。⚠️ 不上 PRO 6000:env torch2.x+cu124/126 arch 只到 sm_90,**PRO 6000=Blackwell sm_120 跑不起**(no kernel image),A800 克隆即用。多卡=`accelerate launch + deepspeed ZeRO-2 --num_processes 2`。
4. **监控 4B**:bjb1 训到 30k(6.6ep)不中途停,eval 经 queue 自动 sweep。盯曲线挑峰,够好照 PI_v3-8B 流程发布。

## 🔑 复用要点
- **HF 大文件单传必 `HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 hf upload`**——否则 xet commit_chunk 被代理掐卡 99%(见 [[hf-upload-tricks]] §0.2)。PI_v3 18.9G 反复卡就是漏了这个。
- 大文件 merge/serve 在本机也间歇段错误(同 kernel6.17 堆腐蚀)→ retry 循环(78k 试 3 次才过)。
- 关联 [[starvla-so101-cloud-training]] [[frozen-vlm-head-extraction-sweep]] [[feedback-vla-epoch-budget-6ep]] [[feedback-smoke-500step-quick-gate]] [[hf-upload-tricks]] [[wallx-env-py310-torch-segfault]]。
