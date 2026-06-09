## Benchmarking & eval standards
- [📊 20-Round STRICT 统计基准](feedback-20round-strict-benchmark.md) — leaderboard/model card 必须用 20-round + P(placed=k) 分布 + 5-round sub-sample σ；脚手架 `scripts/benchmark/run_one_strict.sh` + `aggregate_distribution.py`
- [📏 5-Round benchmark 唯一标准](feedback-5round-benchmark-standard.md) — `scripts/benchmark/run_one.sh` 是权威；EPISODE_LENGTH_S=120, MAX_ROUND_WALL_S=180, EVAL_ROUNDS=5, STEP_HZ=60 (N1.7), action_horizon per baselines.tsv col3
- [每模型自己的 action_horizon](per-model-action-horizon.md) — N1.7=40, N1.6=50, ACT=100, SmolVLA=50, X-VLA=32, π0.5=50；TSV lookup + `get_action_horizon.py`
- [eval ≥ 5 round 强制](eval-5round-mandatory.md) — 3 round variance 太大 (1/3-3/3, 67-89%, 51-133s)；强制 ≥ 5 round (15 ep) 降到 ±10%
- [benchmark 表格排序规则](feedback-benchmark-table-sort.md) — README leaderboard 排序：strict Rounds DESC → oranges DESC → time ASC

## Training discipline
- [⏱️ smoke 用 ~500 步控制 20min 快判能动](feedback-smoke-500step-quick-gate.md)
- [📐 best 通常 3-4 ep / 新训练默认 max ~6 ep](feedback-vla-epoch-budget-6ep.md) — 60-demo 小数据 VLA;max_steps=6×frames/batch,eval 从 3 ep 起;epoch 跨 run 比 — 新 backbone 起训前 smoke `MAX_STEPS=500 SAVE_INTERVAL=500` 只验"不崩+VRAM+臂会动",能力看全量 5k 起 sweep
- [🚦 PI_v3 sweep 终态 + compact后4任务(2026-06-08)](three-box-sweep-live-state.md) — PI_v3-8B已发63.3%rank3;新env训DP/续训2B跨4.5ep/开训9B/监控4B;daemon重启+各机终态+XET上传坑
- [🧬 冻结-VLM sweep 保全=抽head拉回](frozen-vlm-head-extraction-sweep.md) — 8B ckpt 92.7%是冻结VLM(每个相同)；box端mmap抽~1.4G可训head(action_model+project_layers)拉回，本地用一次性vlm_base合并还原；绕开全量拉不动+keep_last删峰值+ENOSPC
- [🔁 StarVLA ckpt/resume/迁移规则](starvla-checkpoint-resume-migration.md) — ckpt只存模型权重(无optimizer);resume重建optim+快进scheduler;迁到别的机续训只需head+vlm_base合并(box可关机不丢能力);max_train_steps必须取save_interval整数倍否则最后epoch不存(9B设13610→6ep那个根本不存,白训610步)
- [🧹 云端复用box起训前清死重+磁盘预检](feedback-cloud-env-reuse-disk-cleanup.md) — 旧base/smoke/旧run死重累积→训到中途torch.save ENOSPC崩(伪装flash-attn,签名=PytorchStreamWriter+unexpected pos+.tmp残片)；起训前du三处清死重+算(KEEP+1)×ckpt<盘；崩后RESUME=1从最新完整ckpt续
- [🏆 MimicKit LAFAN G1 × 4 motion SHIPPED](mimickit-lafan-fight-training-plan.md) — 2026-06-02 fight/run/dance/jumps 15s 切段，4h pipeline，3/4 触顶 ≥98%，run plateau 63%；scripts/mimickit_train_{one,queue,pipeline}.sh + eval_chain.sh + g1_usd_recolor.py 全套；doc/mimickit_lafan_training.html
- [🎨 MimicKit g1.usd 全白根因 + 修复](mimickit-g1-usd-material-fix.md) — MJCF per-geom rgba 转 USD 被压成单一 DefaultMaterial 白；scripts/g1_usd_recolor.py 反推 + per-mesh rebind；`MIMICKIT_G1_USD` env var 切入
- [🧹 训练完 benchmark 后 outputs 清理](feedback-training-output-cleanup.md) — leaderboard 落地后每家族留 1 dir + 3-6 ckpts；负面家族也留 1 个存档；规则写入 `LeIsaac/CLAUDE.md`
- [🎯 GPU util 是训练效率优化的判断锚](feedback-gpu-util-as-efficiency-anchor.md) — 不看 step/s 看 mid-window GPU util %；1Hz GPU+CPU 双采样 + per-phase profile + micro-bench 隔离；瓶颈分层判断表
- [训练 = 训练 + 自动每阶段 eval](auto-eval-watcher-standard.md) — `lerobot_finetune.sh AUTO_EVAL=1` spawn `eval_watcher.sh` poll ckpts/ → 3 连 0 → `.eval_abort` SIGTERM；CSV 持久 dedup 保 resume
- [长训练每 1/10 步 quick eval 强制](feedback-incremental-eval-during-training.md) — 总 step 10 等分，每格 1-round 3-ep quick eval；连续 3 slice 0 → abort + diff config。已写入 `LeIsaac/CLAUDE.md`
- [长训练 ckpt 双层 prune](feedback-training-save-policy.md) — watchdog 留最近 3 + 完成 phase collapse 到 final-only；不裁剪 → disk full SIGSEGV 看似 flash-attn 实为 ENOSPC
- [长训练拆 5 段 resume](feedback-training-resume-chunks.md) — 长跑 (>2h) 默认拆 N≈5 段 via save_every + resume，不一把梭哈

## MimicKit motion training (G1)
- [🔴 AMP dance 负面 vs ✅ PPO-30s 成功（同 clip dance1_s2）](mimickit-amp-g1-dance-negative.md) — AMP Disc_Agent_Acc 平台 0.98 跟不上节奏；DeepMimic PPO-30s Test_Return 244>15s基线227 收敛；dance 保真用 phase-tracking 不用 AMP；含 AMP/PPO 启动 (BASE_ENV/AGENT, init ckpt 要绝对路径) + 指标解读 (Return=0正常/Agent_Acc应降0.5/γ折扣返回值15s/30s同档/eval_chain硬编码1500对本地30s会SKIP)

## VLA — 架构侧 (SONIC WBC) 路线
- [🕺 架构侧 GR00T×SONIC-WBC 路线 + 路 A 动作源](sonic-wbc-vla-route.md) — VLA 只吐 64 维 FSQ token，WBC 当平衡底座；**下一步走路 A**：本地 deploy demo(macarena/kick/dance 13条) 经 `convert_soma_csv_to_motion_lib.py --fps 50`(唯一坑) 转 robot_filtered，smpl=dummy，跑 eval 看 WBC 跟不跟得住踢/舞；评审验过 smpl_joints非坑、HF无多动作robot_filtered别下30G；doc/groot_sonic_wbc_route + sonic_dance_motion_source.html
- [🔬 SONIC VLA 三模型联合评审 + P0-P4 roadmap](sonic-vla-critique-roadmap.md) — GR00T-N1.7-G1-SONIC HF页 9条共识缺陷(无held-out=记忆/FSQ离散码当连续回归/无记忆单帧闭环=本质缺陷/不摔疑RELAX-confound)；正序=先loss→history→数据；LeSONIC/doc/sonic_vla_critique_roadmap.html

## VLA distillation (planned)
- [🧬 MimicKit→VLA 蒸馏路径计划](mimickit-to-vla-distill-plan.md) — DeepMimic PPO → prompt-conditioned 人形 VLA；基建优先于 expert，先补 recorder/modality.json/第三人称camera 跑 10ep×2motion sanity；doc/mimickit_to_vla_dataset.html
- [🐛 lerobot-v040 转换 segfault + recorder teardown 修法](lerobot-v040-convert-segfault-fix.md) — 多 ep 写 LeRobotDataset 随机崩(SIGSEGV/pandas/sre/datasets)= dual-ffmpeg(PyAV+torchcodec)堆损坏；修=encode 改 ffmpeg-CLI 子进程 + get_task_index 用 .at + shape 用 tuple；另 Isaac headless recorder teardown 卡死用 os._exit(0)；脚手架 rollout_record.py + rollout_common.py + mimickit_episodes_to_lerobot.py
- [📊 VLA 蒸馏数据多样性 ROI 排序](vla-distill-data-diversity-roi.md) — prompt侧 > DR/RSI/action-noise > multi-clip > 训更长；DR ≠ semantic diversity；clip 选择不同编舞>不同phrase>不同subject；2-3验证/4-6覆盖

## Sim eval bugs / patches (reusable)
- [ACT/DP 默认关 stuck-detector](feedback-stuck-detector-off-act-dp.md) — chunked policy 短暂 pause/replan 不是 dead；`run_one.sh` 按 POLICY_TYPE 切 STUCK_WINDOW_S=99999
- [🐛 lerobot DP async server stack 空 bug + patch](lerobot-dp-async-server-bug.md) — `predict_action_chunk` 不 populate_queues → `n_obs_steps>1` 必崩；v0.4+v0.5 都中招；一行 patch 已 apply 到 `lerobot-v040` editable
- [placement 计数 reset 假阴 bug](gr00t-placement-bug-fix.md) — `policy_inference.py` 在 `env.step` 后读 placed_flags，触发 termination 时 obs 已 auto-reset → 严格 0/9 假阴；修法挪到 `env.step` 之前
- [OpenVLA Trainer 解包错根治补丁](openvla-floatingpointops-fix.md) — train.py 顶部 1 行 monkey-patch `floating_point_ops=lambda: 0`，消除 bnb+PEFT `_named_members` 解包错
- [LeIsaac eval timeout + DP DDIM swap](leisaac-eval-timeout.md) — DDPM 100-step → DDIM 32-step 不重训直接 swap，inference 393→147ms；4090 sweet spot 公式：`(target_ms - 36) / 3.3 ≈ 29 step`

## Current baseline architecture
- [🦿 GEAR-SONIC G1 预览跑通配方](gear-sonic-preview-setup.md) — 走 `gear_sonic_preview.sh` 单进程 Isaac-eval（非 DDS/C++ sim2sim）；isaaclab env 补 easydict/loguru/open3d/vector_quantize_pytorch + **trl==0.28.0**（新版删了旧路径）；WBC 已 submodule + patches/gear-sonic symlink 448M ckpt 进 HF cache
- [🤖 LAFAN G1 motion-tracking ecosystem 地图](lafan-g1-ecosystem.md) — 数据 + 现成 ckpt + retargeter 全集；结论 LAFAN_fight G1 ckpt 生态里 0 个，要么 ProtoMotions 不兼容，要么自训 (MimicKit)
- [🏗️ GR00T N1.5/N1.6/N1.7 多 release 环境分离](gr00t-multi-release-env-split.md) — 3 submodule + 3 venv；transformers 4.51.3 (N1.5/N1.6) vs 4.57.3 (N1.7) 隔离；policy_type 统一 `gr00t`
- [🔧 GR00T-N1.7 × LeIsaac 4 层 wire 协议 debug](gr00t-n17-leisaac-wire-debug.md) — 剥洋葱 4 bug：missing observation envelope / 缺 T 轴 + float32 / msgpack-numpy bytes-key 漏解码 / mnp.decode 强制 bytes keys。fix 在 `service_policy_clients.py:Gr00tServicePolicyClient`
- [🏆 ACT framework drift root cause 锁定 + 修复验证](act-framework-drift-root-cause.md) — lerobot v0.4→v0.5 间 PR #3406 (dataloader uint8/persistent_workers/prefetch) + PR #3442 (ACT padding loss fix) 是 root cause；锁版本 lerobot v0.4.0 + torch 2.7.1+cu126 → 3/15 → 8/15
- [hi-space GR00T-N1.7 验证](gr00t-n17-hi-space.md) — 真 N1.7 (Gr00tN1d7+Cosmos-Reason2-2B+action_horizon=40)；submodule 落 `dependencies/Isaac-GR00T`；Cosmos-Reason2-2B gated 要 Request access
- [🎯 PickOrange VLA 选型铁律=vision 分辨率](vla-pickorange-vision-resolution-selection.md) — 橙子10-40px需≥448;Wall-X(40-65%官方SO-101示例,默认256要调高)>StarVLA+Qwen3-VL(30-60%SO101=TODO)>OpenVLA-OFT(15-40%仅验action head)>multi_task_dit(5-30%CLIP-CLS@224弱)>LingBot(等单臂ckpt);两份 plan + roadmap§3.5
- [🔧 本机 py3.10.20→import torch 间歇segfault + Wall-X env 配方](wallx-env-py310-torch-segfault.md) — **头号坑=python 3.10.20 conda构建让 import torch ~40%间歇段错误(C栈溢出,纯Python稳);根治=env换 python 3.11**(与torch/CUDA版本无关;100%相关实测);那些posixpath/sre/pyc诡异错全是同一堆腐蚀表象,非独立bug。wallx配方:py3.11+torch2.6cu124+transformers4.51.3(非4.49)+flash-attn cp311+TORCH_CUDA_ARCH_LIST=8.9编csrc;freeze_vlm训0.47B expert;底座wall-oss-0.5;脚手架 workspace/leisaac_pick_orange/
- [☁️ Wall-X PickOrange 搬上 AutoDL 云端训练](wallx-autodl-cloud-training.md) — 本地kernel6.17腐蚀放弃→AutoDL(kernel5.15稳/4080-32G/CUDA12.4);env=torch2.5.1(非2.6)+nvidia库从base拷(triton别拷=cp312坑)+flash-attn torch2.5wheel;**启动铁律=用env二进制全路径别conda activate(非交互SSH会挂)**;`/root/train_cloud.sh`+`pass autodl/westd`;config路径已改云端;compact后续训看本文§当前状态
- [🔌 Wall-X 闭环 eval serving 适配器 + 3坑](wallx-eval-serving-adapter.md) — serve_wallx.py(WallXPolicy+WebsocketPolicyServer,action_tokenizer传None非"")+WallXServicePolicyClient;**坑1**=bf16显存14.8G(fp32→bf16残留)→`torch.cuda.empty_cache()`掉到8.4G本地和Isaac共存;**坑2**=msgpack-numpy openpi(`__ndarray__`)vs标准(`nd`)不兼容→client override infer用标准codec;**坑3**=trainer step-save `data_config`+x2robot池索引在lerobot必崩→`hasattr(primary_pool_start_index)`守卫;真eval前要patch process_images硬编码256
- [🧩 StarVLA 换 VLM 骨干 2B/4B/8B 零源码配方](starvla-vlm-variant-2b-4b-8b.md) — QwenGR00T runtime 对齐 hidden_size→换骨干只改 config;2B/4B hidden=2048 8B=4096;bs(8B=4其余8)+save密度(4B1500/8B6000/2B1000,峰是120k样本驱动);gradient_ckpt/flash-attn Qwen3死字段,真VRAM大头repeated_diffusion_steps;westd:15528-4090 / westc:31709-4080S
- [🔢 大VLM本机24G必须8bit eval + 间歇载入堆腐蚀重试 + 单卡争用坑](starvla-8bit-eval-load-corruption.md) — 8B bf16+Isaac必OOM→8bit(VLM int8/head bf16,11G);8bit≈bf16实锤35.0%=35.0%;大ckpt load~40%间歇崩(段错误或'int'no stale_possible_simple_keys)→serve重试;wallx_sweep_supervisor while-true复活偷GPU→假0先杀supervisor
- [📈 8B sweep 30k峰7/9后悬崖塌 + 2B live state + 下载finalize-hang](starvla-8b-2b-sweep-result.md) — 8B曲线0/0/1/4/7(30k峰)/0/0/0;watcher拉完即删云端防ENOSPC+self-heal+MIN_STEP;hf_transfer传完字节卡finalize→标准下载器秒rename;2B训练中westc~step1200/30000
- [🌟 StarVLA(Qwen3-VL-4B)westc 云端训 SO-101 + eval-sweep](starvla-so101-cloud-training.md) — 第二台云机 westc:31709;env=py3.10+torch2.6cu124(aliyun无→pytorch.org代理)+transformers4.57.0+flash-attn cp310torch2.6;repo自带SO101Config+v2.1走v2.0路径+av1用torchvision_av;**四真坑=_pack_sample硬编码224死穴(改448)/build_dataloader硬编码16workers爆62G-cap-RAM(改4)/冒烟进程不退占24G显存(逐PID kill)/ckpt无裁剪填满60G盘ENOSPC崩(patch原子save+keep-last-N)**;Run1=QwenGR00T冻VLM只训head,GPU25/32G-97%util-1s/step;**eval方法论=先快速eval确认机械臂会动再auto-sweep**:本地clone wallx→starvla_eval(补rich/pytorch3d等),每ckpt rsync10GB回本地4090,serve_starvla(stateless+448+度转弧度)+StarVLAServicePolicyClient+starvla_sweep_watcher.sh GUI轮流用卡;脚手架examples/SO101_PickOrange/ + LeIsaac/scripts/evaluation/serve_starvla.py

## Negative archives (published to HF, 防止重踩)
- [🔴 π0.5 PyTorch expert-only FT 完整失败](pi05-pytorch-expertonly-phase15-negative.md) — 13 ckpt sweep + ckpt-2000 strict 20-round = 1/60 (1.7%)；根因 = **SigLIP@224 vision bottleneck**（橙子缩后 ≤1 SigLIP patch）。HTML: pi05_pytorch_expert_ft_negative.html

## Cloud / mirror / HF (reusable infra)
- [🇨🇳 PyTorch / pypi mirror 优先 aliyun](cn-pypi-mirror-aliyun.md) — `download.pytorch.org` 从本机 TLS handshake EOF；切 `mirrors.aliyun.com/pytorch-wheels/cu128` 14 MB/s 稳；`prefetch_uv_cache.sh` sed 替换不污染 git diff
- [🔧 AutoDL × CN uv sync Isaac-GR00T 6-patch 配方](autodl-uv-sync-cn-strategy.md) — aliyun pypi default + 直接 URL torch + x86_64 only + 砍 tensorrt + no_proxy + flash-attn cxx11abiFALSE
- [hf upload-large-folder 实战坑](hf-upload-tricks.md) — `pip install hf_transfer` + `HF_HUB_ENABLE_HF_TRANSFER=1` 5-10× 提速；`--exclude` 单 flag 多 value；resume cache `.cache/huggingface/upload/<file>.metadata`
- [☁️ AutoDL box 直下 HF ~64MB/s](autodl-hf-download-speed.md) — network_turbo + hf_transfer 直下快(64MB/s,模型在box直下别本地下再传)；box→HF快 vs box→本机PULL才是~5MB/s瓶颈,别混淆
- [💰 AutoDL 云端训练成本纪律](feedback-autodl-cost-discipline.md) — 每步先问"真的需要 GPU 模式吗"，无卡 ¥0.1/h vs GPU ¥6-8/h 差 60-80×
- [HF README 项目链接按相关性](feedback-hf-readme-project-links.md) — model card 顶部插入与本 ckpt **任务相关**的 GitHub repo 链接（不是固定两个）；manipulation→两条都放，motion-tracking/locomotion→只放 isaaclab-experience；无关项目反向链接 = SEO 噪音 + 看着像 spam
- [HF frontmatter 必填 datasets + base_model](feedback-hf-frontmatter-datasets-basemodel.md) — model card YAML 必须有 `datasets:` 和 `base_model:` (无则 `[]`)，否则 HF UI 不渲染数据集卡片 + 血缘关系；body 用 markdown 链接不算数
- [📁 发布后自动归入对应 Collection](hf-collections-auto-place.md) — `scripts/hf_make_collections.py` 权威幂等；按任务族选 collection（PickOrange→LeIsaac PickOrange）；发完新 repo 加进 GROUPS 再跑

## Collab style
- [HTML 文档样式规则](feedback-html-doc-rules.md) — 必须 `:root{color-scheme:light}` + body 显式 `background:#fff`；SVG/pre/table 都要成对显式 background+color；内嵌 SVG 不外链
- [启动 GPU 任务前先检查显存](feedback-pre-run-gpu-check.md) — nvidia-smi + pgrep 残留进程，发现 >2GB 占用 / 老 server 状态错位先清理再启动
- [协作风格偏好](feedback-style.md) — 设计文档先行（HTML+中文+SVG）、目录按语义不按工具、开源化是默认目标、本地优先、负面结果如实写、不问废话
- [🔍 mimo 独立审查习惯](feedback-mimo-independent-review.md) — 关键设计/取舍前后习惯性开 `opencode run -m xiaomi/mimo-v2.5-pro` 拿第二意见；tmux 跑不阻塞、用完收 session
