---
name: flowdp-sonic-bonesseed-live-state
description: FlowDP(conv-UNet flow-matching head)移植到 SONIC BonesSeed 7动作 token 生成 — 设计/脚手架/坑/live-state
metadata:
  type: project
---

**2026-06-14 起**:把 [[act-flowhead-vita-variant]]/[[flowdp-e1-e2-live-state]] 的 **flowdp 头**
(LeIsaac/FlowHeads/flowdp,= lerobot DiffusionPolicy 骨干 verbatim,只把 DDPM→rectified-flow)
移植到 SONIC BonesSeed 7动作 motion-token 生成(对比 GR00T finetune / StarVLA CE,见
[[starvla-sonic-ab-baseline]] [[sonic-wbc-vla-route]])。目标=训完能 `@flow2` 在 Isaac GUI 预览实时动作串联。

## 核心设计(用户拍板)
- **flowdp/DP 无语言通路** → 7动作只能靠 prompt 区分。方案=**7维 motion-onehot 拼进 observation.state**
  (不新增条件通路,flowdp 头保持 DiffusionConfig verbatim 零改动;UNet 第一层 cond Linear 自动学成 embedding)。
- state(53)=joint(43, **raw 列序**)+projected_gravity(3)+motion_onehot(7);action(78)=motion_token(64)+左手(7)+右手(7)。
  serve 只回 token[:64](SONIC decoder 只吃 token,手部是训练辅助目标被丢)。
- onehot 每 ep 恒定(ep_index==task_index==motion:0 dance/1 lunge/2 macarena/3 kick/4 squat/5 jump/6 walk)。

## 脚手架(都在 LeSONIC/scripts/)
- `sonic_flowdp_build_dataset.py`:GR00T 多键 v2.1 → DP 友好 `datasets/sonic_vla_flowdp`(单 action+单 state)。
  **零视频重编码**(cp 视频→改 parquet→再跑官方 v21→v30 converter,concat 是 stream-copy)。
  **图像 stats=注入 ImageNet mean/std**(ResNet18 是 ImageNet 预训练,这才是对的 MEAN_STD,且免解码视频)。
  STATE/ACTION=MIN_MAX 自己 numpy 重算。**lerobot 0.4.4 要 v3.0 格式**(v2.1 直接加载报 BackwardCompatibilityError)。
- `sonic_flowdp_train.sh`:lerobot-v044 env,`flowdp.train`,n_obs_steps=1(配 injector 单帧)/horizon=40
  (镜像 GR00T,可被8整除)/n_action_steps=32/resize=[240,320](ego 看地面视觉弱)/flow_steps=10。RESUME=1 走 `--config_path+--resume`。
- `sonic_flowdp_watchdog.sh`:**自动 resume 重试**(本机 py3.10/kernel6.17 间歇堆腐蚀崩,见下),SAVE_FREQ=1000。
- `serve_flowdp_sonic.py`:ZMQ 镜像 gr00t PolicyServer wire(msgpack-numpy)。prompt→task_index(读
  `meta/tasks.parquet`,**v3 把 tasks.jsonl 改成了 parquet**:prompt 作 index、task_index 作列)→onehot;
  state 分组**散射回 raw 列序**;归一化用策略自存的 PolicyProcessorPipeline(pre);action **手动 MIN_MAX
  反归一化**(`(x+1)/2*(max-min)+min`,因 from_pretrained 不还原 postprocessor 的 to_transition/to_output 转换器)。
  返回 `{"action.motion_token":(1,32,64)}`。未知 prompt **fail-closed**(raise,不发零 onehot=未训练输入)。
- `gear_sonic_flowdp_demo.sh`:一键=起 flowdp server(:5557)+`GR00T_PORT→gear_sonic_sequence.sh @flow2`
  (injector wire-agnostic,Isaac 侧零改动);ACTION_HORIZON=32 配 n_action_steps。

## 坑 / 教训
- **🔴 kernel-6.17 堆腐蚀崩训练**:跑 934 干净步后崩 `ValueError: too many values to unpack (expected 0)`
  在 torch `named_parameters()` 核心迭代里(和 import 时的 sre_compile 错同源,见 [[wallx-env-py310-torch-segfault]]
  [[starvla-8bit-eval-load-corruption]])——**非训练逻辑 bug,间歇/可阵发**。对策=watchdog 自动 resume。首存在 1500
  时崩=丢 934 步白跑 → SAVE_FREQ 降 1000。
- lerobot **0.4.4 只认 v3.0 数据集**;v2.1 用 `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30
  --repo-id=<name> --root=<父目录> --push-to-hub=false`(repo_id 追加到 root;原地转换 root→root_old 备份)。
- serve 归一化:`PolicyProcessorPipeline.from_pretrained(ck, config_filename="policy_preprocessor.json")` 可加载
  pre(Rename+Batch+Device+Normalizer);但 post 的 unnorm 靠 code 转换器 to_transition/to_output,from_pretrained
  不还原 → 裸 tensor 报 "EnvTransition must be a dictionary"。故 action 反归一化手动做。
- DP resize 在**模型内部**(config.resize_shape),pre pipeline 不含 resize → serve 发原始 480×640 CHW float[0,1] 即可。

## 评审(codex gpt-5.5 + mimo,2026-06-14,见 [[feedback-mimo-independent-review]])
两者一致确认 state 散射/onehot 归一化/图像通路/horizon 切片都对。修复的真问题:① **P0 tasks.parquet**(serve 原读 jsonl
会崩,v3 已改 parquet)② 未知 prompt fail-closed ③ action 输出范围断言 ④ denom eps ⑤ train 默认步数对齐。
mimo 说 "checkpoints/last/ 不存在" **是错的**(lerobot-v044 实测会建 last)。codex 标的 chunk 间随机性(每 replan 新 prior)
=DP 固有,加了 `--seed` 可选确定性钩子,默认仍随机。

## ⚠️ epoch 修正(2026-06-14,用户纠正)
初版闷设 15000步≈**250 epoch** = 错。用户指出该带相似度评分扫 ckpt 找甜点,且 flowdp 甜点是**几个 epoch**。
实证:**LeIsaac flowdp PickOrange sweep best = step 9800 ≈ 4.3 epoch**(14000步≈6.2ep run;36293帧/b16→2268步/ep)。
flow-matching velocity loss 会一直降(误导),但采样动作质量(eval 才测得到)早见顶。**改为密集低-epoch 存档
(STEPS=2400≈40ep,SAVE_FREQ=120≈2ep/ckpt=20个)+ 自动评分扫**。评分标准见 [[feedback-lesonic-similarity-leaderboard-standard]]。
- 评分器实测 ckpt @16.7ep(旧粗run):macro-MSE 路径 step1000 frame-MSE=0.0116 beat 模板(per-motion-mean 0.0396)3.4×
  (GR00T 0.0011)→ 证明 motion-onehot conditioning 学到了,7动作 token 分布可区分。

## ✅ 训练完成 + 终态（2026-06-14 晚）
- **100ep 跑完**(STEPS=6000/batch64/save240=25 ckpt),3 层自愈栈(watchdog resume + 纯-bash patrol 重拉 + eval-watcher 自动评分)全程扛住 kernel-6.17 间歇崩溃(多次 burst,patrol 解了 MAX_RETRIES 耗尽风险)。
- **best = step 6000(100.7ep):frame-MSE 0.00059 / macro 0.00059,skill 64.7×,7/7 全破模板**。**比 GR00T 0.0011 还低 ~2×**(同 raw token 口径)。
- **曲线实证甜点**:4ep=0.33垃圾→12ep首破模板→单调降→88-100ep plateau~0.0006。**SONIC flowdp 甜点在 ~90-100ep(记忆任务无held-out,过拟合即目标),不是 PickOrange flowdp 的 4.3ep(那有held-out)**。这条曲线就是"必须带评分扫epoch"的最佳证据。
- 产物:`outputs/flowdp_sonic/{checkpoints/*(25个),openloop_eval.csv,leaderboard.md}`。**25 ckpt×3G=75G 待裁**(留 best 6000 + 几个邻居,删其余,确认后)。
- FlowHeads doc 新增:`dependencies/FlowHeads/doc/flowdp_diffusion_to_flow_matching.html`(DP→flow 原理,3 SVG,中英)。
- 待办:① best ckpt 跑 `bash scripts/gear_sonic_flowdp_demo.sh @flow2` GUI 目检串联(需 GUI);② 裁 ckpt;③ 提交(见下迁移)。

## 🎯🎯 实时闭环 = 一个数据 bug,非模型脆(2026-06-15 重大翻案)
live 部署 flowdp(`gear_sonic_flowdp_demo.sh squat`)**机器人秒倒**;查了一整轮(IID 增强/per-step clamp/codex+mimo 评审"需 DAgger+相位条件")**全是误判,追错症状**。真凶 = **数据集 `observation.projected_gravity` 全是 [0,0,0]**(录制时没采 gravity)→ MIN_MAX stats 退化(min==max==0)→ **lerobot 退化归一化把 min→-1,但任何非零输入算成 `2x/eps-1`≈±1e8**;部署时 injector 发真实 gravity(z≈-1)→ 归一化成 **~-2e8** → 喂进模型输出爆炸 |chunk|~5 → 倒。
- **诊断铁证**:同一帧 live-obs,gravity 原值 |chunk|=4.82 vs gravity 置零 |chunk|=1.06;50 帧真实 live-obs(GT回放捕获)gravity 原值 **47/50 爆**,置零 **0/50 爆**(全 |chunk|~1.07)。关节其实完美匹配(漂移仅 0.006),模型对真实漂移**完全鲁棒**(归一化空间 σ=0.2 都没事;之前"σ=0.15爆"是 raw-space 扰动对紧关节≈归一化σ~1.0 的口径错配)。
- **修复 = serve 一行:`pg3 = np.zeros(3)`**(匹配训练的全零 gravity → 归一化成 -1,模型本来就没用 gravity 信息)。**无需重训,best ckpt 6000 直接可用**。已在 `serve_flowdp_sonic.py` get_action 修。
- **教训**:① degenerate(min==max)归一化维在部署遇非训练值会**爆 1e8**,不是 clamp 到常数——务必检查 dataset stats 有无退化维;② "脆"先量真实 obs 漂移(GT回放捕获 live-obs 离线喂模型),别凭 raw-space 扰动猜 σ;③ 多专家评审也会被错误的问题框定带偏(我把"爆炸"当成 flow 脆性,其实是归一化 bug)。
- **次级限制(真·no-memory,gravity修后才暴露)**:不倒但**幅度小/初动后趋静** = 单帧 n_obs_steps=1 无相位/速度 → 退化成姿态条件均值 token。GR00T(3B VLM)单帧能自持,flowdp(小+ego看地面)需 **history 条件(n_obs_steps≥2)+ serve 缓冲多帧**(评审也指此)或 bootstrap 带入。**待办(用户暂搁)**:n_obs_steps≥2 重训治本。
- **serve 其余修(评审 P0/P1,已做)**:FSQ-snap token→k/16 格点 + off-grid 率;`import os` 移出热循环;per-step clamp 安全网(env FLOWDP_SAMPLE_CLAMP)+ proprio 噪声增强(env FLOWDP_STATE_NOISE,默认关——其实不需要了,gravity 才是真因)。
- **诊断脚手架(都在/tmp,可重建)**:GT-token dump `datasets/sonic_vla_gt/<KEY>.npz`(录制真token);serve `SONIC_GT_DIR`(GT回放,robot正确不倒,捕获live-obs)+`SONIC_CAPTURE_LOG`(记raw state)。BonesSeed.ipynb §5.2b/5.2c 加了 flowdp flow2 + 7动作单独 live cell。
- **本机 Isaac GUI 坑**:反复起 Isaac 后 Vulkan 崩(`VkResult NOT_READY`/weakly-ref);harness Bash 工具沙箱杀 detached GUI 进程(空日志秒死)→ 用户自己 notebook kernel 跑才稳(等~70s server+Isaac加载)。

## 📦 submodule 迁移（2026-06-14,用户要求,工作区已备,用户提交）
- `LeIsaac/FlowHeads` → **`LeIsaac/dependencies/FlowHeads`**(`git mv`,gitlink+worktree+doc 都迁;LeIsaac 内已 staged)。
- 新增 **`LeSONIC/dependencies/FlowHeads`** submodule(`git submodule add` 远端 a3001a4 克隆,**不含未提交的 doc**——需先在 LeIsaac 那份 commit+push FlowHeads,LeSONIC 再 `submodule update --remote` 同步)。
- `LeSONIC/MaskBeT` → **`LeSONIC/dependencies/MaskBeT`**(手动迁移:MaskBeT 之前是 orphan 缺 gitlink → git mv 拒绝;移 worktree + 修 `.git`指针(`../../.git/modules/MaskBeT`)+ `core.worktree`(`../../../dependencies/MaskBeT`)+ 改 .gitmodules path + `git add` 补 gitlink b21cf15。**顺带修好了它的 orphan**)。metadata 留 `.git/modules/MaskBeT` 不动(按 name 不按 path)。
- LeSONIC 脚本引用全改:flowdp 脚本 `FLOWHEADS=$REPO_ROOT/dependencies/FlowHeads`(自包含不再向上够 LeIsaac);MaskBeT 脚本 `$REPO_ROOT/dependencies/MaskBeT`。import 冒烟过。
- 提交顺序(用户做):先 FlowHeads(commit doc+README→push)→ LeIsaac(submodule mv+指针)→ LeSONIC(MaskBeT mv+FlowHeads add+脚本)→ 伞仓指针。见 [[git-submodule-gitlink-gotcha]] [[feedback-commit-message-oneline]]。
- starVLA 子模块已 `pull --rebase` 到最新 59faca5。
