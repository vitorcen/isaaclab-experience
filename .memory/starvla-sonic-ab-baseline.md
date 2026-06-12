---
name: starvla-sonic-ab-baseline
description: StarVLA(Qwen3.5-4B PI_v3) vs GR00T N1.7 在 SONIC LAFAN flow3 上的本机 A/B 基线——接线配方 + 5 个真坑 + 结果
metadata:
  type: project
---

**2026-06-10 实施**（用户拍板 framing ①=干净 A/B 基线，先经 brainstorm 文档二轮批判补盲点）。
测床 = `LeSONIC/datasets/sonic_vla_lerobot_flow3`（8 窗口/5777 帧/8 prompt），GR00T 锚点 =
checkpoint-6000（6000 步×global batch 4=24k 样本≈4.15 epoch，loss 0.052，开环 MSE avg
0.00257、全窗口≤0.004）。设计文档 `LeSONIC/doc/sonic_starvla_swap_brainstorm.html` §11
（含预注册预测：会收敛但不优于 GR00T）。关联 [[sonic-wbc-vla-route]] [[starvla-vlm-variant-2b-4b-8b]]。

## 接线（全离线，不碰 serving 栈）
- 脚手架：`LeSONIC/scripts/starvla/{data_registry/data_config.py,configs/sonic_qwen3_5_4b_pi_v3.yaml}`
  + `starvla_sonic_finetune.sh`（symlink 部署 kit 进 LeSONIC 嵌套 `dependencies/starVLA/examples/UNITREE_G1_SONIC/train_files/`，
  registry 只扫那里）+ `starvla_dump_pred_tokens.py`（teacher-forcing 开环 dump + MSE 表，npz 兼容
  `gear_sonic_inject.sh`，prompt→key 映射对齐 `sonic_vla_pred_flow3` 文件名）。
- env：`starvla_eval_qwen35`（py3.11+torch2.6cu124+**transformers 5.2.0**(Qwen3.5 必须)+flash-attn+
  accelerate 1.10.1，**补装 deepspeed 0.16.9** 后可 accelerate+zero2 单卡训）。
- 关键决策：action=motion_token(64)+左右手(7+7)=**78 维**（镜像 GR00T embodiment，MSE 口径可比）;
  horizon **40**; action 归一化=**identity**（key 不进 `normalization_modes` 即跳过，token 1/16 网格
  原样保留，dataloader 冒烟实测 off-grid=0）; state=43+gravity3=**46 维 min_max**; bs4×6000 步
  =GR00T 同样本量; Qwen3.5-4B（hidden 2560/32 层/有 vision_config）冻结只训 PI_v3 head。

## 5 个真坑（部分是 brainstorm 文档批判时抓的盲点）
1. **`repeated_diffusion_steps` 是 trainer.\* 键不是 action_model 键**：`QwenPI_v3.forward()` 读
   `config.trainer.get("repeated_diffusion_steps", 16)` —— 不显式设就是 **16 倍 DiT batch**。
   设 2 在 24G 上 step1 即 OOM（22.2G+954M）→ **设 1**（bs 保 4，epoch 对齐不变）+
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → 2.1 it/s @23.2G 稳。
2. **PI_v3 state 离散化 bins 固定 `[-1,1]`**（`state2str_transform` 256 级）：关节弧度（膝>2 rad）
   不先 min_max 会全 clip 进边缘 bin，proprio 信号被毁 → state 必须 min_max。
3. **`datasets.vla_data.include_state` 默认 off**（SO101 就是 stateless 跑的）：忘开就退化成
   (image,prompt)→token，与吃 proprio 的 GR00T 不可比。
4. **全局索引≠轨迹顺序**：`ds[0]` 的 lang 是 ep7 的 prompt —— dump 必须用 `inner.all_steps` 的
   (traj_id, base_index) 显式映射 + prompt→key 查表，别赌顺序。
5. **min_max 对 min==max 维归零**（手指 14 零维安全）；`_pack_sample` action cast fp16（1/16 网格
   间距 0.0625 >> fp16 精度，无损）。

## 结果（2026-06-10 当晚，🔴 负面坐实——比预注册预测更狠）
- smoke 500 步门通过：loss 1.17→0.21；全量 6000 步训练健康收敛（loss→~0.06，2.1 it/s，48min）。
- **A/B（teacher-forcing 开环，同 24k 样本）**：GR00T ≈**0.0026**（破模板基线 14×）vs StarVLA
  steps_6000 MSE64=**0.0374** ≈ **per-window-mean 模板基线 0.0367 打平**（global-mean 0.0408；
  bin_acc 16.5%）；steps_5000=0.0385，1000 步仅降 3% = **平台非欠训**。逐窗 0.013-0.051 无一进
  GR00T 区间。**判读：PI_v3 从零 head 等预算下学不动逐帧 token 结构，塌缩成 prompt 条件均值模板；
  "换更好 VLM 就行"被实测堵死，瓶颈不在骨干**（§7 预判=丢机器人预训练 base+horizon 2.5× 拉伸）。
- **GUI 闭环目检（用户裁决，combat 注入）**："有动作，但比 GR00T 生硬很多，勉强维持不倒地"——
  与开环互证：模板 token 给轮廓、缺逐帧结构、WBC 平衡底座硬扛。
- **用户真实目标 = 彻底开源可自由改结构的底座来迭代** → A/B 杀死的是 stock PI_v3 头从零训，
  不是 StarVLA 代码库；下一步最高 ROI = framing ②（per-dim CE / pre-quant latent 的 FSQ-aware
  损失，改 `LayerwiseFM_ActionHeader` 一处，接线/评测/注入全复用；两根标尺已立：模板 0.0367 / GR00T 0.0026）。
- caveats：rds=1 是 24G 逼的（但平台说明非主因）；双方都是记忆口径无 held-out；head-only 单 run。
- 另坑：dump 时 `import torch`/ckpt load 间歇 GPF（kernel 日志 general protection fault，
  本机 kernel 6.17 堆腐蚀家族）→ retry 循环 ≤8 次即愈。
- 产物：`outputs/starvla/sonic_qwen3_5_4b_pi_v3/`（只留 steps_6000）+
  `datasets/sonic_vla_pred_starvla/`（8 npz+metrics.json，兼容 inject）。结果表在 brainstorm 文档 §11.1。

## 🎉 QwenPI_CE（per-dim FSQ 交叉熵 head）= 突破（2026-06-10 深夜）
用户定向：「彻底开源可自由改结构的底座来迭代」→ 实施 framing ②。
- **实现**：`LeSONIC/scripts/starvla/framework/QwenPI_CE.py`（源已收编 LeSONIC，运行时 symlink 部署进 starVLA 框架注册表；曾是 untracked 孤儿，
  要转 patch**；registry 自动扫 VLM4A/*.py，`@FRAMEWORK_REGISTRY.register("QwenPI_CE")` 即挂上）。
  结构=冻结 VLM last hidden → LayerNorm+Linear(1024) → 40 个 learned horizon query 的 6 层
  TransformerDecoder cross-attn → logits (40,78,33bins)，bin k∈[-16,16] 值=k/16，argmax 推理=构造性在格。
  CE head 挂名 `self.action_model` 以吃 trainer 的 per-module LR 组（1e-4）。
  ⚠️ QwenPI_v3 的类名是 **`Qwen_PI_v3`**（带下划线）——import 踩过。trainer 只读
  `output_dict["action_loss"]`，多返回 bin_acc 等 key 安全。
- **Run3 单窗口验证（749 帧, bs4×2000 步≈10.7 epoch）**：bin_acc **16.5%→50.5%**，MSE64
  **0.0117 = run3 模板基线 0.0236 的一半**（stock flow head 同窗口 0.0354）——**逃出均值模板，
  loss 形式正是瓶颈实锤**。CE loss 2.1→~0.5-0.8 nats（ln33≈3.5 起点参照），未完全收敛。
- 脚手架：run3 单动作数据集 `datasets/sonic_vla_lerobot_run3`（ep5 重编号，**沿用全量 stats.json**
  保 state 归一化一致）+ configs/sonic_run3_qwen3_5_4b_ce.yaml + sonic_qwen3_5_4b_ce.yaml（8 窗口）。
- **8 窗口 epoch 对齐（6000 步）CE 结果（2026-06-11 凌晨）= 放量突破确认**：MSE64 avg
  **0.0174 / bin_acc 41.6%**（stock 0.0374/16.5%；模板 0.0367；GR00T 0.0026）——**8/8 窗口逐一破
  各自模板基线**（stock 0/8），最佳 fierce_swings 0.0043 摸到 GR00T 区间边缘。改一刀 loss 同预算
  2.1×，flaw #2 实测坐实为主瓶颈之一。与 GR00T 仍差 6.7×（候选=预训练 base/layerwise 特征/head
  容量 90M/软标签未做）。CE head 3.5 it/s @14.3G（比 flow 快 60%）。预测 token
  `datasets/sonic_vla_pred_starvla_ce/`。结果表 brainstorm 文档 §11.2。
- **🛰️ CE live 闭环链路（2026-06-11）**：`scripts/serve_starvla_sonic.py`（ZMQ REP，**完全仿
  gr00t PolicyServer wire**=msgpack_numpy `{"endpoint":"get_action","data":{"observation":...}}`，
  obs 收原始值 **server 内做 min_max**(直读 stats.json，与训练 transform 数学一致)，回
  `{"action.motion_token":(1,40,64)}`）→ Isaac 侧 vla_live_injector/gear_sonic_sequence.sh
  **零改动**，只要 `GR00T_PORT=5556`。一键=`scripts/starvla_sonic_live_demo.sh @flow3`
  （自动起 server:5556+GPF 重试+复用、PRED_DIR 默认 CE 预测 npz 当 bootstrap）。server bf16 9.4G
  与 Isaac(~7G) 24G 共存。wire 冒烟=token 构造性在格 off-grid=0。LAFAN.ipynb 已加 cell。
- **📥 LAFAN.ipynb 下载/回退（2026-06-11）**：一键 `snapshot_download('wsagi/GR00T-N1.7-G1-SONIC-LAFAN')`
  进默认 HF cache；live cell 自动解析 ckpt=`outputs/gr00t_sonic_flow3/checkpoint-6000` 存在用自训、
  否则 `local_files_only=True` 取 cache（没下载则提示先跑下载 cell）；另配可选 dump cell 重建
  pred npz（gitignored 产物）。
- **🎮 CE live flow3 用户目检（2026-06-11）**："动作幅度很小，还需改进" —— 与开环差距（6.7×）一致；
  机理=argmax 解码偏保守，不确定维度塌到众数 bin（≈静止值）→ 幅度被压。
- **✂️ 第①刀已做（2026-06-11）：解码侧 expected-value，不重训 MSE −28%**。`QwenPI_CE._decode()`
  支持 argmax/expected/sample，env `SONIC_CE_DECODE`/`SONIC_CE_TEMP`/`SONIC_CE_SNAP` 切换（默认 argmax
  零行为变化）。同 steps_6000 ckpt 扫 6 档（doc §11.3 全表）：**expected T=0.5 = Pareto 甜点**
  （MSE64 0.0174→0.0125、std 保 88% GT、jerk 反低于 argmax），已设为 `starvla_sonic_live_demo.sh`
  默认（snap=1 回格点，wire 零变化）。**反直觉**：离线 argmax token std 已=GT(0.179)——"幅度小"
  不是 token 空间整体塌缩而是闭环精度不足；sample 解码保 std 但 jerk 3-5×GT + MSE 升，排除。
- **✂️✂️ 第②刀 layerwise（2026-06-11）：run3 假突破，8 窗口放量负结果，不采纳**。实现=CEActionHead
  加 ELMo 式可学 softmax 混 37 层 hidden（`ce_layerwise: true`，+37 参数，训速不变）。run3 快验
  亮眼（argmax 0.0117→0.00425、expT0.5 0.00363 进 GR00T 带、std 95% GT）但 **8 窗口同预算全面退步**
  （argmax 0.0208 vs v1 0.0174、expT0.5 0.0142 vs 0.0125、bin_acc 39.1% vs 41.6%、std 无优势）。
  **大教训：run3 单窗口快验测床系统性高估改动**——单窗 10.7 epoch 新容量先去记忆；佐证=lw 逐窗
  分布（短窗 run_circle/fierce_swings 仍优、dance 长窗全退=记忆型增益形状）。**以后快验要么
  训 run3 测 held-out 窗，要么直接 8 窗口短跑**。ckpt 负面留档 `sonic_qwen3_5_4b_ce_lw/steps_6000` +
  `sonic_run3_qwen3_5_4b_ce_lw/steps_2000`。live 默认维持 **CE v1 + expected T=0.5（0.0125）**。
  ③ 软标签 CE 未做。**坑**：finetune.sh 相对 CONFIG 路径在 cwd=starVLA 下断→已修 `readlink -f`
  绝对化+不存在 exit 2；确定性错误 vs GPF 鉴别=连崩等间隔 ~66s 且 run 目录都没建。
- **🧹 磁盘大清理（2026-06-11，152G→376G 空闲）**：每族只留最终 ckpt —— gr00t_sonic_8k 只剩
  checkpoint-8000、flow3 只剩 checkpoint-6000(fp32)、skills/8plus_fight 各剩末 ckpt、flow3_bf16 整删
  （HF 有+可重做）、starvla 三 run 删 `final_model/`（与 checkpoints 末 .pt 重复；`config.yaml`+
  `dataset_statistics.json` 必须保留=from_pretrained 依赖）。serve_starvla_sonic 已停（要看 live 点
  notebook cell 自动拉起）。kernel 6.17 的 GPF 不止 ~40% 间歇，会**阵发**（连续
  6-7 次启动即 SIGSEGV，过几分钟自己好）；崩点有三种=import 段/训练中段/存 ckpt 段（原子保存兜住，
  只留 .tmp 残片）。**根治姿势=`scripts/starvla_sonic_watchdog.sh`**（用户提议参照
  [[feedback-training-save-policy]]）：SAVE_INTERVAL=250 密滚动 + KEEP=3 + pgrep 互斥 +
  60×60s 重试 + 自动 RESUME + 清 .tmp；MAX_STEPS 必须整除 SAVE_INTERVAL。
