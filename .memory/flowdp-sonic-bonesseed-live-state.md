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

## live-state（2026-06-14 晚,重启后）
- 训练:`sonic_flowdp_watchdog.sh` 跑着,`outputs/flowdp_sonic`,**2400步/batch64/save120**,~2.2 step/s,**ETA~18min**,GPU 9G。
- 自动评分:`sonic_flowdp_eval_watcher.sh` 跑着→`outputs/flowdp_sonic/{openloop_eval.csv,leaderboard.md}` 实时更新。
- 日志:/tmp/flowdp_sonic_{train,watchdog,evalwatch}.log。
- serve 端到端已验(真 ckpt-1000):wire/7动作可区分/fail-closed 都过。
- 待办:① 看 leaderboard 挑 best ckpt;② best 跑 `@flow2` GUI 目检串联(最终交付,需 GUI,等训完腾卡)。
- starVLA 子模块已 `pull --rebase` 到最新 59faca5。
