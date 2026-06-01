## Benchmarking & eval standards
- [📊 20-Round STRICT 统计基准](feedback-20round-strict-benchmark.md) — leaderboard/model card 必须用 20-round + P(placed=k) 分布 + 5-round sub-sample σ；脚手架 `scripts/benchmark/run_one_strict.sh` + `aggregate_distribution.py`
- [📏 5-Round benchmark 唯一标准](feedback-5round-benchmark-standard.md) — `scripts/benchmark/run_one.sh` 是权威；EPISODE_LENGTH_S=120, MAX_ROUND_WALL_S=180, EVAL_ROUNDS=5, STEP_HZ=60 (N1.7), action_horizon per baselines.tsv col3
- [每模型自己的 action_horizon](per-model-action-horizon.md) — N1.7=40, N1.6=50, ACT=100, SmolVLA=50, X-VLA=32, π0.5=50；TSV lookup + `get_action_horizon.py`
- [eval ≥ 5 round 强制](eval-5round-mandatory.md) — 3 round variance 太大 (1/3-3/3, 67-89%, 51-133s)；强制 ≥ 5 round (15 ep) 降到 ±10%
- [benchmark 表格排序规则](feedback-benchmark-table-sort.md) — README leaderboard 排序：strict Rounds DESC → oranges DESC → time ASC

## Training discipline
- [🧹 训练完 benchmark 后 outputs 清理](feedback-training-output-cleanup.md) — leaderboard 落地后每家族留 1 dir + 3-6 ckpts；负面家族也留 1 个存档；规则写入 `LeIsaac/CLAUDE.md`
- [🎯 GPU util 是训练效率优化的判断锚](feedback-gpu-util-as-efficiency-anchor.md) — 不看 step/s 看 mid-window GPU util %；1Hz GPU+CPU 双采样 + per-phase profile + micro-bench 隔离；瓶颈分层判断表
- [训练 = 训练 + 自动每阶段 eval](auto-eval-watcher-standard.md) — `lerobot_finetune.sh AUTO_EVAL=1` spawn `eval_watcher.sh` poll ckpts/ → 3 连 0 → `.eval_abort` SIGTERM；CSV 持久 dedup 保 resume
- [长训练每 1/10 步 quick eval 强制](feedback-incremental-eval-during-training.md) — 总 step 10 等分，每格 1-round 3-ep quick eval；连续 3 slice 0 → abort + diff config。已写入 `LeIsaac/CLAUDE.md`
- [长训练 ckpt 双层 prune](feedback-training-save-policy.md) — watchdog 留最近 3 + 完成 phase collapse 到 final-only；不裁剪 → disk full SIGSEGV 看似 flash-attn 实为 ENOSPC
- [长训练拆 5 段 resume](feedback-training-resume-chunks.md) — 长跑 (>2h) 默认拆 N≈5 段 via save_every + resume，不一把梭哈

## Sim eval bugs / patches (reusable)
- [ACT/DP 默认关 stuck-detector](feedback-stuck-detector-off-act-dp.md) — chunked policy 短暂 pause/replan 不是 dead；`run_one.sh` 按 POLICY_TYPE 切 STUCK_WINDOW_S=99999
- [🐛 lerobot DP async server stack 空 bug + patch](lerobot-dp-async-server-bug.md) — `predict_action_chunk` 不 populate_queues → `n_obs_steps>1` 必崩；v0.4+v0.5 都中招；一行 patch 已 apply 到 `lerobot-v040` editable
- [placement 计数 reset 假阴 bug](gr00t-placement-bug-fix.md) — `policy_inference.py` 在 `env.step` 后读 placed_flags，触发 termination 时 obs 已 auto-reset → 严格 0/9 假阴；修法挪到 `env.step` 之前
- [OpenVLA Trainer 解包错根治补丁](openvla-floatingpointops-fix.md) — train.py 顶部 1 行 monkey-patch `floating_point_ops=lambda: 0`，消除 bnb+PEFT `_named_members` 解包错
- [LeIsaac eval timeout + DP DDIM swap](leisaac-eval-timeout.md) — DDPM 100-step → DDIM 32-step 不重训直接 swap，inference 393→147ms；4090 sweet spot 公式：`(target_ms - 36) / 3.3 ≈ 29 step`

## Current baseline architecture
- [🏗️ GR00T N1.5/N1.6/N1.7 多 release 环境分离](gr00t-multi-release-env-split.md) — 3 submodule + 3 venv；transformers 4.51.3 (N1.5/N1.6) vs 4.57.3 (N1.7) 隔离；policy_type 统一 `gr00t`
- [🔧 GR00T-N1.7 × LeIsaac 4 层 wire 协议 debug](gr00t-n17-leisaac-wire-debug.md) — 剥洋葱 4 bug：missing observation envelope / 缺 T 轴 + float32 / msgpack-numpy bytes-key 漏解码 / mnp.decode 强制 bytes keys。fix 在 `service_policy_clients.py:Gr00tServicePolicyClient`
- [🏆 ACT framework drift root cause 锁定 + 修复验证](act-framework-drift-root-cause.md) — lerobot v0.4→v0.5 间 PR #3406 (dataloader uint8/persistent_workers/prefetch) + PR #3442 (ACT padding loss fix) 是 root cause；锁版本 lerobot v0.4.0 + torch 2.7.1+cu126 → 3/15 → 8/15
- [hi-space GR00T-N1.7 验证](gr00t-n17-hi-space.md) — 真 N1.7 (Gr00tN1d7+Cosmos-Reason2-2B+action_horizon=40)；submodule 落 `dependencies/Isaac-GR00T`；Cosmos-Reason2-2B gated 要 Request access

## Negative archives (published to HF, 防止重踩)
- [🔴 π0.5 PyTorch expert-only FT 完整失败](pi05-pytorch-expertonly-phase15-negative.md) — 13 ckpt sweep + ckpt-2000 strict 20-round = 1/60 (1.7%)；根因 = **SigLIP@224 vision bottleneck**（橙子缩后 ≤1 SigLIP patch）。HTML: pi05_pytorch_expert_ft_negative.html

## Cloud / mirror / HF (reusable infra)
- [🇨🇳 PyTorch / pypi mirror 优先 aliyun](cn-pypi-mirror-aliyun.md) — `download.pytorch.org` 从本机 TLS handshake EOF；切 `mirrors.aliyun.com/pytorch-wheels/cu128` 14 MB/s 稳；`prefetch_uv_cache.sh` sed 替换不污染 git diff
- [🔧 AutoDL × CN uv sync Isaac-GR00T 6-patch 配方](autodl-uv-sync-cn-strategy.md) — aliyun pypi default + 直接 URL torch + x86_64 only + 砍 tensorrt + no_proxy + flash-attn cxx11abiFALSE
- [hf upload-large-folder 实战坑](hf-upload-tricks.md) — `pip install hf_transfer` + `HF_HUB_ENABLE_HF_TRANSFER=1` 5-10× 提速；`--exclude` 单 flag 多 value；resume cache `.cache/huggingface/upload/<file>.metadata`
- [💰 AutoDL 云端训练成本纪律](feedback-autodl-cost-discipline.md) — 每步先问"真的需要 GPU 模式吗"，无卡 ¥0.1/h vs GPU ¥6-8/h 差 60-80×
- [HF README 必带项目链接](feedback-hf-readme-project-links.md) — 发到 HF Hub 的 model card 顶部图片之后、TL;DR 之前固定插入 vitorcen/isaaclab-experience + vitorcen/LeIsaac 两个 repo 链接

## Collab style
- [HTML 文档样式规则](feedback-html-doc-rules.md) — 必须 `:root{color-scheme:light}` + body 显式 `background:#fff`；SVG/pre/table 都要成对显式 background+color；内嵌 SVG 不外链
- [启动 GPU 任务前先检查显存](feedback-pre-run-gpu-check.md) — nvidia-smi + pgrep 残留进程，发现 >2GB 占用 / 老 server 状态错位先清理再启动
- [协作风格偏好](feedback-style.md) — 设计文档先行（HTML+中文+SVG）、目录按语义不按工具、开源化是默认目标、本地优先、负面结果如实写、不问废话
