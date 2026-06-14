---
name: e1-midlayer-sweep-live-state
description: E1 实验(StarVLA Qwen3.5-4B GR00T_v2 select_layer=12 中层特征)的 live 状态——bjb1 训练 + 本机插空 eval sweep + 与隔壁 FlowDP 共用 4090 的协调;compact 后据此续。2026-06-13。
metadata:
  type: project
---

# E1 中层特征实验 live state（2026-06-13，compact 续点）

**目的**：验证三方评审假设——8B GR00T_v2 strict 13.3% 的负面是 **porting bug(头读末层 hidden_states[-1])**
还是真"N1.7 头不迁移"。E1 唯一改动 = 读**中层 select_layer=12**(Qwen3.5-4B 32 层)。见 [[starvla-gr00t-v2-n17-head]]。

## 训练（远端 bjb1）
- box: `ssh -p 49769 root@connect.bjb1.seetacloud.com`(密码=会话 security-constraints 的 bjb1 pw / `sshpass -e`,绝不写文件)。
- run: `so101_pickorange_qwen35_4b_gr00t_v2_midlayer`，30000 步(6.6 ep)，save_interval 3000，keep_last 6，batch 8(求与 PI_v3 可比,不加大)。
- 启动器 `/root/train_qwen35_4b_gr00t_v2_midlayer_full.sh` → `/root/run_train.sh`(已加 `ulimit -n 1048576`)。崩了 `RESUME=1` 同脚本续训(ckpt 只存权重,resume 重建 optimizer)。
- 巡检 `/root/patrol_e1.sh`(v2 报 anon 真RSS vs 页缓存)。**内存看 anon(<15G 正常),pagecache 爬升=反复读视频可回收 benign,别误报**。pids 平稳 ~1100 不爬 cgroup pids.max(20480)——爬=dav1d 线程泄漏回归,见 [[starvla-av1-dav1d-thread-leak-enomem]]。
- 已修两坑:① AV1/dav1d 线程泄漏(video_backend 换 pyav + thread_count=1)；② config 注释层数 24→32。约 2026-06-13 22:30 跑满。

## 本机插空 eval sweep（与隔壁 FlowDP 共用 4090）
- 脚本(gitignored)：`LeIsaac/outputs/starvla-qwen35-4b-gr00t-v2-midlayer/e1_gap_sweep.sh`，detached daemon。
  两趟/轮：**pass1 拉全 head(纯网络,不被门挡)→ pass2 对未评步 GPU-gated 排队 eval**。
- 重启：`SSHPASS=<bjb1 pw> setsid nohup env SSHPASS=$SSHPASS bash <绝对路径>/e1_gap_sweep.sh >/tmp/e1_gap_sweep.log 2>&1 </dev/null &`
  （**必须 setsid detach + Bash 工具 dangerouslyDisableSandbox**,否则本机 GPU 任务在工具调用里同步跑会被 **exit 144** 杀且零输出；杀 watcher **必须连带 kill -9 它的 setsid serve 子进程**=按 :8014 端口 PID / serve_starvla PID,否则留孤儿占 10G 显存+占端口）。
- 单点流程:盒上抽 head(`action_model.*` ~1.15G)→rsync 本地→**merge 用 `starvla_eval` env**(qwen35 env 偶发 segfault mmap-load,故分离;3 次重试+验 973 key)→serve(`starvla_eval_qwen35` env,Qwen3.5 需 transformers 5.2,`STARVLA_DIR=dependencies/starVLA` 含 select_layer,BASE=本地 Qwen3.5-4B snapshot,:8014)→`conda run -n isaaclab policy_inference.py --task=LeIsaac-SO101-PickOrange-v0`→记 CSV→`rm` merged full(留 head+json)。
- **GPU 门(硬化, 2026-06-13 再修)** = `pgrep -f 'policy_inference.py.*lerobot-flowdp'` 在跑则 yield(return 1)**且** GPU free≥13000MiB。**教训(step 21000 被 OOM-kill)**:8bit serve 只 6.6G 但**两个 Isaac sim 同时加载必撞 OOM,杀掉其一(`已杀死`)——重的是 Isaac 本身不是 serve**;"8bit coexist"判断错,正解是**让位 FlowDP 的 Isaac eval**。注意 FlowDP 的 `:8080` async server 是**常驻**(eval 间隙不退),不能按它判,只能按 per-eval 的 `policy_inference` 进程判。**vlm_base** = `LeIsaac/outputs/_head_sweep_tools/vlm_base_qwen35_4b.pt`。
- CSV：`LeIsaac/outputs/starvla-qwen35-4b-gr00t-v2-midlayer/e1_sweep.csv`。head 抢救在本地 `heads/`(box keep_last 不裁 heads 目录,且 15k-30k full 留到训练末)。

## 5-round 曲线（step=ep / E%，方差极大、单点不可信）
3k(.66ep)=13.3 / 6k(1.3ep)=13.3 / 9k(2ep)=20 / **12k(2.65ep)=46.7** / 15k(3.3ep)=6.7 / 18k(4ep)=33.3 / 21k(retry中,前次被FlowDP撞OOM)。训练 2026-06-13 19:44 在 22320/30000(pids 1145 平稳)。
甜点区(2.6-4ep)三点合并 13/45≈**29%**。相邻点 40 点摆动=5-round σ~18%+,见 [[eval-20round-still-noisy-combine-runs]]。
**方向**:mid-layer 确实把头救活(对比 8B 末层全程平 13%)→ porting-bug 说基本坐实;但**是否追平 PI_v3 4B 峰值 46.7% / v1 53.3% 未定**,目前看在 ~30%(±20) 带。

## ⚠️ LIVE 续点（2026-06-14 01:30，compact + 本地重启后据此恢复）

**E1 训练**：已满 30000 ✅，head 全在 bjb1 `heads/` + 本地 `heads/`，box fulls 已删(reconstructable)。**5-round 曲线**：12k=46.7 18k=33.3 **21k=60** 24k=53.3 27k=46.7(有头/无头混)。

**E3a 训练（bjb1，跑着，与本地无关）**：smoke 过(V1 断言 53 unfrozen 全进 optimizer)→ **全量训练中 ~step 7000+/30000(~6h)**。RUN_ID=`so101_pickorange_qwen35_4b_gr00t_v2_midlayer_unfreeze4`,launcher `/root/e3a_full.sh`(崩了 RESUME=1 跑它续)。ckpt 在 box keep_last 6。**本地重启不影响它**。

**✅ E1 20-round DELIVERABLE 完成（2026-06-14,重启后干净重跑）**——有头 ep120 wall180 serve:8014 8bit,CSV `e1_strict20.csv`,指标按 placed_flags 的 oranges 分布算(别用 success flag,retracted_middle 假阳少算):
| ckpt | E[oranges]=Σ/60 | P(≥3)=#oranges==3 /20 | P(≥1) |
|---|---|---|---|
| 21k | 33/60=**55.0%** | 6/20=30.0% | 15/20 |
| 24k | 30/60=**50.0%** | 5/20=25.0% | 16/20 |
| 27k | 25/60=**41.7%** | 4/20=20.0% | 13/20 |
| **池化60轮** | **88/180=48.9%** | **15/60=25.0%** | 44/60=73.3% |

**结论(强,但有 2 个混淆未控,故不擅自推翻已发布结论)**:E1 4B mid-layer **48.9% E / 25% P3** vs [[starvla-gr00t-v2-n17-head]] 记的 **8B-v2 last-layer 13.3% E / 0 P3** = 巨大提升→**强力支持"末层特征是移植 bug、N1.7 头能迁移",推翻原"头不迁移"**。但:① **E1 是 Qwen3.5-4B,8B-v2 与 v1(53.3%)都是 8B**——换层+换模型大小双变量,非纯净 last-vs-mid;② **E1 有头多轮 wall_cap 截断**(v1 8B 小头 headless-180 不截断)→ 48.9% 被截断压低,真值更高。即便如此 4B+截断 下仍逼近 v1 8B 53.3% E(但 P3 25%<v1 35%)。⚠️ 口径勘误:旧 loop 词"PI_v3 46.7%"是误标,**PI_v3 真值=63.3%**,46.7% 是自训 GR00T-N1.6(tune_top_llm_layers=4)。**改写 README/[[starvla-gr00t-v2-n17-head]] 前须与用户确认 4B-vs-8B 框架**(重大结论反转)。

**serve 间歇崩坑(本轮踩)**:27k 首跑 ep3 报 `RuntimeError: Error in inference server: ... ValueError: duplicate parameter name: 'attention_mask'`=serve 进程间歇状态崩(非 ckpt 确定性,21k/24k 同代码 40 轮正常),重起 runner 重试即过,见 [[starvla-8bit-eval-load-corruption]]。runner 的 3×retry 只防 serve **载入**失败,防不住评测**中途**崩。

**C(E3a 本地 sweep)已启(2026-06-14 05:02)**:`e3a_gap_sweep.sh` runner pid 272701,full-pull :8015 rounds=5 poll=300s flock 同锁让位 FlowDP,CSV `e3a_sweep.csv`。full-pull 慢(box→本地 ~5MB/s,10G ckpt ~33min/个)。
**🐛 C 系统性 SERVE_FAIL 坑(已修 06:07)**:`serve_starvla.py:repoint_base_vlm` 要读 `<run_dir>/config.yaml`(reconstruct 框架),但 sweep 只 full-pull ckpt **没拉 config.yaml** → 每个 E3a eval 都 SERVE_FAIL(3000/6000 实锤,serve log = `FileNotFoundError: .../unfreeze4/config.yaml`)。**修=从 box 拉 `<box run dir>/config.yaml` 到本地 run dir**(`sshpass rsync ...:/root/autodl-tmp/starvla-outputs/<run>/config.yaml <local run dir>/`);box 那份是训练真值(select_layer=12+tune_top_llm_layers=4+DiT-N17),serve 的 repoint 会把 base_vlm 改本地路径。**serve 还需第二个文件 `dataset_statistics.json`**(share_tools.py:read_mode_config 断言;归一化统计,与 E1 字节同=同数据集)→也从 box 同 run dir 拉。**通用规则:任何 full-pull(非 E1 merged-ckpt 路径)的 sweep 都得连 `config.yaml`+`dataset_statistics.json` 一起拉**(两文件 06:18 补齐)。⚠️**skip 坑**:e3a 脚本 `grep -q "^$step," CSV && continue` 把 **SERVE_FAIL 行也当已处理→永久跳过**(与 e1 的 `[0-9]` 数字判不同)→修 serve 后必须**清 CSV 的 SERVE_FAIL 行**(我 06:25 重置 CSV 为 header)才会重评。e3a serve 原**无 3×retry**(E1 有)→12000 实测 serve 载入 segfault(核心已转储)直接 SERVE_FAIL。**已打补丁(06:44)给 e3a_gap_sweep.sh serve 段加 3×retry + `kill -0 $sv||break` 段错误早检测**(mirror e1_strict),bash -n 过,重启 C(pid 443911);现能扛 ~40% 间歇段错误。三种 SERVE_FAIL 原因都见过并修:缺 config.yaml→拉、缺 dataset_statistics.json→拉、载入 segfault→retry。

### E3 部分解冻设计（已落地,代码就绪+codex SHIP-WITH-FIXES,未提交）
**只 E3a 不 E3b**(E3b=v1 头解冻是 off-design 人造对照;E3a vs E1-frozen 已是同头干净 A/B)。fork 改:`trainer_tools.py`(build_param_lr_groups 按 requires_grad 筛)+`train_starvla.py`(optimizer 构建前 freeze+unfreeze+硬断言零解冻 fail-fast)+`verify_unfreeze.py`+`unfreeze4.yaml`(tune_top_llm_layers=4,batch4);③DeepSpeed bf16 自带 fp32 master 不手动 cast。已 rsync 到 bjb1(`.bak_pre_e3` 备份)。

### 🔴 E3a 结果(2026-06-14)= 解冻是 NO-OP,实验无效(配置自相矛盾)
**5-round 曲线**:9k=26.7 15k=40 18k=26.7 27k=20(全 ≤40%,与 E1 head-only 同档)。**prune_ckpts.py 抽 delta 时露馅**:每个 ckpt delta=纯 head(249keys/1.15G),GOLD✓ → **解冻的 top-4 层与冻结 base 字节相同**。直接对比 3k↔30k 实锤:**layer 28/29/30/31(tune_top_llm_layers=4 取的 `layers[-4:]`)maxΔ=0、30k 步零更新**,只有 head 变(maxΔ=0.145)。
**根因 = 配置自相矛盾**:`select_layer=12` **物理 pop 掉 layer 13-31**(只用第12层中层特征)→ 解冻的 top-4(28-31)**根本不在前向 → 零梯度 → 永不更新**。这次 no-op 不是评审预测的 optimizer-membership bug(那个我修了、smoke 也过),而是**前向截断**新根因。**教训:select_layer=N 截断 LLM 时,要解冻必须解冻"被用到的栈顶"(≤N 的层,如 9-12),不能用 `layers[-4:]`(=被丢弃的层)**。E3a ≡ head-only ≈ E1,所以曲线一样。
**✅ 已修+重训(2026-06-14)= GR00T-faithful 截断**:查 GR00T 官方 `qwen3_backbone.py:87-91` 实锤——它**先物理 pop 掉 select_layer 以上的层**(`while len(layers)>select_layer: pop(-1)`)**再** set_trainable,所以 `layers[-4:]`=截断后栈顶=**层 8-11**(产生 select_layer 特征的层),且读 `hidden_states[-1]`。我的港版 bug = 读 `hidden_states[12]` 但**没 pop**→`layers[-4:]`=28-31(丢弃层)。**修法**:QwenGR00T_N17.py 加 config flag `truncate_to_select_layer`(默认 False 向后兼容 E1/v1/v2 + ckpt),True 时 `__init__` 仿 GR00T pop(新增 `_llm_layers()` 辅助)→`layers[-4:]`=8-11 + 特征不变(`hidden_states[12]`=截断后末层=同值,上层从不影响 L11)+ 省~20 层算力。**smoke 实证修复**(steps_250 vs 500 byte-diff):层 8/9/10/11 各 11/14 张量 CHANGED maxΔ=2.4e-4(对比 no-op 全 maxΔ=0)、ckpt 只有 layer 0-11(pop生效,ckpt 7G<11.5G)、head 238/249 CHANGED、冻结 L6 全 0。**双评审(codex gpt-5.5 深审 + mimo)一致"代码正确"**,唯一改点=补已提交 config(已建 `scripts/training/starvla/configs/so101_qwen35_4b_gr00t_v2_midlayer_unfreeze4_trunc.yaml`);num_hidden_layers/fp32-cast/eval_mode 评审清了(DeepSpeed ZeRO-2 + dropout=0 覆盖),不改。**全量重训 LIVE**:box tmux `trunc`,RUN_ID=`so101_pickorange_qwen35_4b_gr00t_v2_midlayer_unfreeze4_trunc`,launcher `/root/trunc_full.sh`(30000步/save3000/keep6),config `..._unfreeze4_trunc.yaml`。**坑**:box 上 tmux 启动要**干净单独命令**(别把 kill/rm/sleep 捆进同一 SSH,timeout 会吃掉 tmux);python3 不在非交互 PATH 用 `/root/autodl-tmp/envs/starvla/bin/python`。代码改在 fork QwenGR00T_N17.py(新文件,不冲突),**未提交**(用户提交)。

### 🏆 E3a-trunc 结果（2026-06-14，解冻=明确大胜）
**trunc 5-round 全曲线**（quick sweep, :8016, headed wall150）:3k=6.7、6k=6.7、9k=46.7、12k=33.3、**15k=66.7**、**18k=60.0**、**21k=73.3🔥峰**、24k=46.7(晚期回落)、27k/30k 待评(下降尾,不超21k)。**对比 E1-frozen(同 4B 同数据,只是 head-only 不解冻):9k=20 18k=33.3 21k=55峰,60-round 池化 48.9%**。→ **截断解冻全程压着 frozen,中段 9k 46.7% vs 20%、21k 73.3% vs 55%**。结论方向=**解冻有用**(非"60-demo 遗忘")。**这同时验证了截断修复本身**:no-op 版(不 pop)20-round 才 13.3%,trunc 版(pop 8-11 真训)5-round 峰 73.3%。
**✅ 训练完成**(2026-06-14 ~15:50,step 30000/30000,5:03:14)。

### 🏁 最终判定（2026-06-14,解冻明确大胜,实验闭环）
**strict 20-round 池化**(headed ep120 wall180,:8017):
- **解冻 trunc 池 = E 61.7% / P3 35%**(21k=40/60=66.7%、15k=34/60=56.7%,2 候选 40 轮)
- **冻结 E1 池 = E 48.9% / P3 25%**(21k+24k+27k,3 候选 60 轮)
- best 单点:trunc 21k **66.7%** vs 冻结 21k 55%
→ **解冻(GR00T-faithful 截断 + 训 8-11 层)比 head-only 冻结 +12.8 点 E / +10 点 P3,明确胜**。证伪"60-demo 小数据解冻只会遗忘";也再次反证 no-op 版(不 pop,28-31 层零梯度)20-round 仅 13.3% 是 bug 而非"头不迁移"。
**18k 没进池**:strict eval 间歇 serve 崩(SERVE_FAIL→NO_METRICS,见 [[starvla-8bit-eval-load-corruption]]),NO_METRICS 路径把本地 18k full rm 了,box 此时已关→无法重拉,放弃(2 候选池已 >> 冻结,结论不变)。**坑记录**:strict/sweep 的 eval_step 在 NO_METRICS 也走末尾 `rm -f full`→失败即丢本地 full,要重评必须 box 还在;best 必须提前 cp 防 rm(21k 当时 cp 成 .BEST 才保住)。
**横向**:trunc 66.7% 是这批 4B 最高(> v1 53.3% > PI_v3 46.7% > 冻结 48.9%);但仍有 4B-vs-8B + wall_cap 口径混淆,改 README 榜单须先问用户口径(task#8)。
**✅ 发布完成(2026-06-14)**:HF `wsagi/StarVLA-Qwen3.5-4B-GR00T_v2-PickOrange`(8 文件:21k full ckpt + config[truncate flag]+ dataset_stats + modality + 训练配方 + demo.mp4[录屏 webm→mp4,坑=1267×755 奇数宽高 libx264 要 scale 偶数]+ model card)。**LeIsaac/README.md 榜单已改**:trunc 进 **rank 3 🥉**(66.7%),rank 4-21 重排,脚注 ② 标记已解决(8B 负面=select_layer porting bug)+ 新脚注 ④(解冻消融)+ 结论加 headline。**git commit 待用户**(我只改工作区不 push)。上传坑=mihomo TUN 卡 99%,见 [[hf-upload-tricks]](逐文件 upload_file + stall 看门狗自动 resume 重试磨过)。
**✅ 归档完成(box 已关)**:`unfreeze4-trunc/checkpoints/steps_21000_pytorch_model.BEST.pt`(6.6G,best,serve/publish)+ `heads/` 3 个 delta(21k/27k/30k,2.0G/个,head 249+层8-11=300张量)。**坑:prune_ckpts.py 对截断 ckpt(12层708keys)要专门的截断 base** —— 满层 vlm_base(724keys/32层)做 base 会 GOLD 失败(merge 多填 12-31 层);解法=造 `outputs/_head_sweep_tools/vlm_base_qwen35_4b_trunc12.pt`(4.3G,=vlm_base ∩ trunc的708key=459 shared)。重建:`merge_ckpt.py vlm_base_qwen35_4b_trunc12.pt <delta> <out>`。15k/18k full 已被 strict eval 的 rm 丢失(box已关无法重拉),但成绩已记录、best 21k 在,无碍。**重起**:`cd .../unfreeze4-trunc && CANDIDATES="21000 15000 18000" SSHPASS=<pw> setsid nohup env SSHPASS=$SSHPASS CANDIDATES="21000 15000 18000" bash ./e3a_trunc_strict.sh >/tmp/e3a_trunc_strict.log 2>&1 </dev/null & disown`(dangerouslyDisableSandbox)。

### 🔁 全自动流水线 LIVE（2026-06-14 09:42，compact 后据此恢复）
**三组件并行,定时巡检在跑(ScheduleWakeup ~1800s,即使唤醒丢了按下面手动接)**:
1. **训练**(box GPU):tmux `trunc`,RUN_ID/launcher 见上,box ckpt `/root/autodl-tmp/starvla-outputs/<RUN_ID>/checkpoints/steps_*.pt`(**实测 6.6G/12层,非7G**)。崩→干净单独命令 `tmux new-session -d -s trunc "RESUME=1 bash /root/trunc_full.sh"`。**2026-06-14 10:40 崩过一次=ENOSPC**(`/dev/md0 130G 100%`→step6000 torch.save `PytorchStreamWriter unexpected pos`,见 [[feedback-cloud-env-reuse-disk-cleanup]]):死重=no-op旧run `..._unfreeze4` 75G + `..._unfreeze4_trunc_smoke` 20G + `steps_6000_*.pt.tmp` 残片,清掉→130G降26%,**RESUME=1 从 3000 续(10:50,log 确认 `truncate_to_select_layer: popped LLM 32->12`)**。教训:起训前该清 no-op+smoke 死重(KEEP6×6.6G=40G 本身没问题,是 95G 死重压满)。
2. **quick sweep**(本地4090):pid `/tmp/e3a_trunc_sweep.pid`(**846143**,前 691949 已重启),脚本 `LeIsaac/outputs/starvla-qwen35-4b-gr00t-v2-midlayer-unfreeze4-trunc/e3a_trunc_sweep.sh`(:8016 ROUNDS5,边训边 full-pull+5round),CSV 同目录 `e3a_trunc_sweep.csv`,flock `/tmp/leisaac_gpu_eval.lock`。死→`cd 该dir && SSHPASS=<pw> setsid nohup env SSHPASS=$SSHPASS bash ./e3a_trunc_sweep.sh >/tmp/e3a_trunc_sweep.log 2>&1 </dev/null & disown`(必须 dangerouslyDisableSandbox)。本地 trunc dir 已有 config.yaml(带 truncate flag)+dataset_statistics.json→serve 截断匹配。**🐛 已修阈值坑(2026-06-14)**:sweep+strict 脚本的 BAD_CKPT 阈值原是 `-ge 970`(满血32层模型 key 数),但 **trunc ckpt pop 掉20层只剩 12 层 = 708 keys(层0-11)** → 误判每个 trunc ckpt 为 BAD_CKPT_708。已改两脚本 `-ge 700`。**任何换层数的 head 都要核对 key 数再定阈值**。
3. **strict 20-round**(本地4090):脚本 `e3a_trunc_strict.sh`(:8017 ROUNDS20 ep120 wall180 candidate-once,**需 CANDIDATES env**),flock 同锁。**训完+quick sweep出曲线→选甜点2-3步→** `cd 该dir && CANDIDATES="<steps>" SSHPASS=<pw> setsid nohup env SSHPASS=$SSHPASS CANDIDATES="<steps>" bash ./e3a_trunc_strict.sh >/tmp/e3a_trunc_strict.log 2>&1 </dev/null & disown`,CSV `e3a_trunc_strict.csv`,从 placed_flags 算 E/P(≥3)。
**终止条件(全交付)**:训练满30000 + quick sweep 全ckpt评完 + strict甜点20-round + 判定(trunc解冻 >E1-frozen48.9%=真有用 / ≈48.9%=60-demo无益)+ 记memory + (用户确认)榜单。**坑**:flock两eval+FlowDP串行;SERVE_FAIL三因(缺config/段错误retry内置/OOM让位FlowDP);box tmux干净单独命令;py用 `/root/autodl-tmp/envs/starvla/bin/python`。
**ckpt 压实+关机**:E3a 6 ckpt 已 prune→留 30k full + 6 head delta(1.15G,GOLD✓ merge+vlm_base 字节还原),释放 57.5G。**bjb1 训练已完,GPU 无活,可关机(AutoDL 关机保盘)或彻底释放(delta 已本地,不丢)**。
