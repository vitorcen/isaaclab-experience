---
name: wallx-eval-serving-adapter
description: Wall-X (wall-oss) VLA 在 LeIsaac 闭环 eval 的 serving 适配器 + 三个非显然 bug 修复（显存/wire/trainer）
metadata:
  type: project
---

2026-06-06 给 wall-oss-0.5 微调 ckpt 接通 LeIsaac SO-101 PickOrange 闭环 GUI eval，跑通冒烟（机械臂在 GUI 里动）。脚手架 + 三个坑。

## 架构（server-client，本地单卡 4090）
- **server**（`wallx` conda env）：`LeIsaac/scripts/evaluation/serve_wallx.py` 包 `WallXPolicy` + wall-x 自带 `WebsocketPolicyServer`。
  - ckpt 是训练 step-ckpt（`0_2` 等），有 model.safetensors + normalizer_{action,propri}.pth + config.yml，**无 config.json/processor** → 脚本自动 symlink `base/config.json` 进 ckpt 目录，processor 从 `train_config["processor_path"]=base`（HF cache 里的 wall-oss-0.5 snapshot）。
  - **`action_tokenizer_path` 必须传 `None` 不是 `""`**：modeling `from_pretrained` 是 `if action_tokenizer_path is not None` 才加载 action_processor，`""` 会触发 HFValidationError(空 repo id)。Flow-matching（use_fast_tokenizer=False）不需要它。
  - `predict_mode="diffusion"`（Flow），`camera_key=["face_view","left_wrist_view"]`，action_dim=6/pred_horizon=32。
- **client**（`isaaclab` env）：`WallXServicePolicyClient(WebsocketServicePolicy)` 在 `service_policy_clients.py`；`policy_inference.py` dispatch `--policy_type=wallx` + `preprocess_obs_dict` 加 `"wallx"`。
- 启动：`run_wallx_smoke_eval.sh`（nohup 包 `conda run -n isaaclab python policy_inference.py ... --enable_cameras` 无 --headless = GUI）。**注意 conda-run + setsid 内联常 exit 1 静默失败；用 nohup 包 .sh 脚本最稳**。

## 🔴 坑1：wall-oss bf16 显存 14.8G（GR00T 才 7-9G）→ 本地和 Isaac 同卡 OOM
- 根因：`vla_mixin.py:to_bfloat16_for_selected_params` 非 FSDP 分支**先 `self.to(float32)` 再把多数参数转回 bf16**，fp32 阶段的 ~7G 被 caching allocator **保留不还**，nvidia-smi 显示 14.8G（live 其实 8.3G）。
- 修复：load 后 `torch.cuda.empty_cache()` → 掉到 **8.4G**，给 Isaac 留 15G，单卡 24G 共存。serve_wallx.py 已内置 + 启动加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 对比：GR00T 全 bf16，磁盘 5-6.5G→显存 ~8G（1.3×）；wall-oss 磁盘 7.8G→显存 2.2× 膨胀就是这个 fp32 残留。**云端跑 server 会和训练抢 4080，别上云，用 empty_cache 本地解决**。

## 🔴 坑2：msgpack-numpy 两套不兼容编码 → server 收到 image 是 dict（`'dict' has no attribute 'size'`）
- LeIsaac `WebsocketServicePolicy` 用 **openpi vendored** packer：ndarray 编码 `{b"__ndarray__":True,...}`。
- wall-x `WebsocketPolicyServer` 用**标准 msgpack_numpy**（`m.patch()`）：`{b"nd":True,...}`。
- 两套互不解码 → 双向都崩（image 进 / predict_action 出）。
- 修复：`WallXServicePolicyClient.infer` **override** 成标准 codec：`msgpack.packb(obs, default=msgpack_numpy.encode)` + `msgpack.unpackb(resp, object_hook=msgpack_numpy.decode)`，两端都用标准。isaaclab env 需 `pip install msgpack-numpy`。

## 🔴🔴 坑0（最致命）：serving 丢了 proprioception 归一化 → 0/9 真凶（2026-06-07，4 方确认）
**症状**：训练 loss 收敛(2.4→0.05)、4 epoch 训满,但闭环 eval 全 0/9 橙子,机械臂**够到橙子但夹不住**(隔壁 StarVLA 同任务同数据 2.6 epoch 就 5/9)。
**根因**：`utils.py:143`(prepare_batch state 块)`normalizer_propri.normalize_data(state, ...)` **返回值被丢弃**,下一行 `inputs["proprioception"] = state` 喂的是**原始电机度数(±100)**。而 `normalize_data`(action_head.py:112)是**纯函数**(返回新 tensor 不原地改),训练侧 `load_lerobot_dataset.py:363` 有赋值 → proprio 归一化到 **[-1,1]**。
**为何致命(非次要)**：proprio 经学到的线性投影 `propri_proj`(action_head.py:652,训练在[-1,1])→ 喂 ±100 输出放大 ~100× → 污染 `<|propri|>` token(vla_mixin.py:387 注入 hidden state),而**该 token 紧邻 action token 且条件化整个 flow-matching ODE 循环**(modeling:1681)→ 图像/语言驱动粗略够到、proprio 反馈烂导致精细关节控制+夹爪时机失效。"手麻了接球"。
**修复**(一行,已应用)：`state = normalizer_propri.normalize_data(state, [obs["dataset_names"]] * state.shape[0])`。
**4 方独立确认**：2 个 trace agent + codex-gpt5.5 + mimo-v2.5-pro,一致 ~90% 此为真凶;freeze_vlm 只压上限非 0/9 元凶(StarVLA 冻 vision 也 5/9);其余(dataset_name key/flow steps/action horizon/mask/action 解码)全排除。
**和分辨率坑同一模式**:serving 静默丢 transform(= lerobot DP populate_queues bug 第三例)。**两个修复都是 serving 专属,训练路径(load_lerobot_dataset.py)本来就对——它正是揭示 serving bug 的参照;所以修复不影响训练、不需因修复重训**。
**⚠️ 但实测:双修复 + 最终 best ckpt `3`(gs4060,4 epoch 训满)+ retract-detect 关闭跑满 240s×3 = 仍 0/9（2026-06-07 定论）**。3_3500 也 0/9。即**两个 serving bug 是必要非充分**——修完行为从"乱挥打翻"改善到"有目的伺服靠近橙子"(用户肉眼确认朝好方向),但**跨不过 grasp 那道坎**。训练 loss 已收敛(2.4→0.05)→ 偏向容量/迁移问题(freeze_vlm 只训 0.47B expert)而非纯欠训,但 StarVLA 同任务要 6.6-13 epoch、Wall-X 只 4 epoch,欠训论亦有据,**两可**。下一步在 [续训扩 schedule(num_epoch 4→6 + num_training_steps→6000,lr 已归0 必须扩展否则学不动,从`3` resume)] vs [停+负面归档] 间决策。

## 🔴 坑3：wall-x trainer step-save 在 lerobot 数据上必崩
- `save_checkpoint` 末尾 `if step != 0:` 块是 **x2robot 多池 dataset 专属**：引用了**不存在的 `self.data_config`**（应是 `self.dataload_config`）+ `self.dataset.primary_pool_start_index`（lerobot dataset 没有）。每次 step-ckpt 存完 model+normalizer 后崩 → 训练每存档即死（但 ckpt 本身已写全，可用）。
- 修复：`if step != 0 and hasattr(self.dataset, "primary_pool_start_index"):` 整块对 lerobot 跳过（true-resume 用 accelerate save_state + global/epoch/within-step，不需要池索引）。

## 单位转换（和 GR00T/LeRobot client 一致）
- obs：LeIsaac joint_pos(弧度) → `convert_leisaac_action_to_lerobot` → 电机度数(state 6维=arm5+grip1)，对齐训练数据(norm_stats 是度数 ±100)。
- action：模型出度数 → `convert_lerobot_action_to_leisaac` → 弧度喂 env。
- LeIsaac cam `front`→wallx `face_view`，`wrist`→`left_wrist_view`。

## sweep（边训边评，自愈，2026-06-06 起跑）
**目标**：每个训练 ckpt 拉回本地 → GUI 3-round quick eval → CSV 排行 → 找最优 → 赢家 strict 20-round。
- **supervisor**（`LeIsaac/scripts/evaluation/wallx_sweep_supervisor.sh`，nohup 启）：watcher 死了 30s 自动重启,直到 CSV 出现 epoch-3(`^3,`)行才停。**起因=watcher 曾凭空消失(session/nohup teardown)停摆 6h**。
- **watcher**（`wallx_sweep_watcher.sh`）：轮询云端 `wallx-outputs/` → 新 ckpt rsync 回 `LeIsaac/outputs/wallx-sweep/<name>/`(跳 optimizer.bin) → `serve_wallx.py` **port 8001** → policy_inference **GUI**(用户要看视觉异常,`--enable_cameras` 无 `--headless`,`DISPLAY=:0`) 3-round → 解析 `Final success rate: <sr> [<succ>/<rnd>], oranges: <pl>/<tot>` 写 `sweep.csv`。
- **launch 坑**：`pkill -f serve_wallx/policy_inference` 名字会匹配自身命令行触发 exit/255；`setsid` 本地常 exit1 → **用 nohup**。每个 ckpt eval ~11min(flow 扩散推理 ~1s/次 × ~112次/ep × 3 round,avg_round_s ~216)。GPU server 8.4G + Isaac 7G = ~15.6G 共存(empty_cache 后)。
- **stage2**：赢家跑 `wallx_strict_eval.sh <ckpt_dir> ROUNDS=20`(port 8002,EPISODE_LENGTH_S=120,出 strict_eval.json)。
- **已测**(`LeIsaac/outputs/wallx-sweep/sweep.csv`)：gs1014=0/3(1/9)、gs1200=0/3(0/9)，都 epoch0-1 早期 0%(正常,peak 在 epoch2-3=gs2k-4k 还没到)。待测 gs1.5k/2.0k/epoch1/2.5k/3.0k/epoch2/3.5k/epoch3。
- **compact 后查 sweep**：`pgrep -f wallx_sweep_supervisor`(没有就 `DISPLAY=:0 nohup bash .../wallx_sweep_supervisor.sh >/tmp/wallx_sweep_sup.out 2>&1 &`)；`column -t -s, LeIsaac/outputs/wallx-sweep/sweep.csv` 看排行。

## ✅ 分辨率 256 crush bug 已修（2026-06-06，曾让所有 sweep eval 假阴）
**症状**：gs2029 GUI eval 机械臂乱挥打翻盘子、0/9 橙子；隔壁 StarVLA(serve@448) 同 ckpt 区间能 6/9。**根因**：`process_images`（utils.py:179）**硬编码 target_size=256**,无视训练 `data.resolution.face_view:-1`(原生 640)/`left_wrist_view:480` → 橙子(~40px)缩成 ≤1 vision patch,train(640)/serve(256) 双重失配 → pi0.5 级别乱挥。env 相机本身是 640×480(`single_arm_env_cfg.py` front/wrist width=640)采集端没问题,**纯 serving bug**。
**修复**（per-camera resolution 串 4 处,逐字对齐训练 `_vision_preprocess` load_lerobot_dataset.py:99）：
- `process_images(... resolutions=None)`：`target_size = resolutions[i] if resolutions else -1`（默认 -1 原生,**不再是 256**）
- `prepare_batch(... resolutions=None)` 透传；`WallXPolicy.__init__(resolutions=None)` 存 `self.resolutions`，调用 prepare_batch 传入
- `serve_wallx.py`：`res_cfg=train_config["data"]["resolution"]; resolutions=[res_cfg.get(cam,-1) for cam in camera_key]` → 传 WallXPolicy
- 验证：serve 日志打印 `per-camera resolutions: {'face_view': -1, 'left_wrist_view': 480}` = 生效。editable 安装,下个 serve 自动带。
**⚠️ pre-fix 的 sweep 行(gs1014/1200/2000/2029 全 0%)是 256 废配置,早期欠训也叠加,别当真值**。相关 [[vla-pickorange-vision-resolution-selection]]

## ⚠️ 双 sweep 共卡 OOM（Wall-X + StarVLA 同时 eval）
24G 4090 上 Wall-X eval(~8.4G serve + 7G Isaac=15.6G) 与 StarVLA eval(~9.5G+6.6G=16G) **不能并存** → 后启的 serve OOM(只剩 76MB)。早期靠 eval 时序自然错峰共存,时序重叠后必撞。**已加一侧让步 GPU 闸门**(`wallx_sweep_watcher.sh` eval_one 开头):serve 前轮询 `nvidia-smi memory.free`,<16000MiB 就 sleep 60 等(最多 30 次),Wall-X 主动避让 StarVLA,避免白加载 Isaac 2min 被 OOM-killer 杀("已杀死"无 Final line)。实测 2_2500 撞死一次→加闸门后完整跑完 13min。真双向互斥仍需 StarVLA branch 也加 flock(未做)。**注意:重启 watcher 别在它 mid-serve 时 kill,否则 serve_wallx 子进程成孤儿占 8001+9G,得手动 `kill -9` 清。**
相关：[[wallx-autodl-cloud-training]] [[gr00t-n17-leisaac-wire-debug]]
