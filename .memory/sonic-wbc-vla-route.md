---
name: sonic-wbc-vla-route
description: 架构侧 GR00T×SONIC-WBC 路线 + 路 A 动作源（本地 deploy demo → robot_filtered）的可执行计划与评审验过的坑
metadata:
  type: project
---

**目标（架构侧，区别于数据侧 DART 蒸馏 [[mimickit-to-vla-distill-plan]]）**：GR00T N1.7 VLA 只输出
SONIC WBC 的 64 维 FSQ 潜 token，WBC（`sonic_release/last.pt`）当现成平衡底座，按 prompt 出舞蹈/武术。
设计文档：`doc/groot_sonic_wbc_route.html`（架构 Stage A/B/C）+ `doc/sonic_dance_motion_source.html`（动作源三路对比）。
2026-06-03 经 codex gpt-5.5 + mimo 两轮独立评审（见 [[feedback-mimo-independent-review]]），结论：**先走路 A**。

## 下一步 = 路 A（也是架构侧 Stage A / Milestone 0 去风险实验）
把本地 SONIC deploy demo 动作转成 Isaac-eval 的 `robot_filtered`，跑 eval 看 **WBC 跟不跟得住踢/舞而不摔**。
```bash
cd dependencies/GR00T-WholeBodyControl
# ⚠️ --fps 必须 50（eval target_fps=50），转换器默认 30；先确认 deploy CSV 源帧率，源≠50 用 --fps_source
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input gear_sonic_deploy/reference/example/neutral_kick_R_001__A543 \
    --output data/kick_robot.pkl --fps 50
python gear_sonic/eval_agent_trl.py +checkpoint=sonic_release/last.pt +headless=False ++num_envs=1 \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/kick_robot.pkl" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy"
```
启动方式同 [[gear-sonic-preview-setup]]（isaaclab env、preview.sh 同款 hydra override）。

## 评审验过的关键事实（别再踩）
- **动作源**：`gear_sonic_deploy/reference/example/` 本地已有 **13 条**（含 `_M`）：macarena/dance_in_da_party（舞）、
  neutral_kick（踢）、lunge、one_leg_jumping、squat、walking_quip_360。CSV(joint_pos/body_pos/body_quat)。
- **fps 是唯一真坑**：converter 默认 `--fps 30`，必须传 `50`；源帧率不符会动作变速变形。smpl_joints **不是**坑。
- **smpl 可 dummy**：`smpl_motion_file=dummy`（`motion_lib_base.py:252`）；converter `:301` 已输出 `smpl_joints=zeros(T,24,3)`；
  g1 encoder 只吃 `command_multi_future_*`+`motion_anchor_ori`（robot FK 派生），**根本不读 smpl_joints**（mimo 追源码证）。
- **robot_filtered schema**：`{motion_key:{root_trans_offset(T,3), pose_aa(T,30,3), dof(T,29), root_rot(T,4 xyzw), smpl_joints(T,24,3), fps}}`，joblib(zlib)。
- **转换器** `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` 输入支持 CSV 目录 / deploy PKL，输出正是上面 schema；内部 `MJ_TO_IL` 处理关节重排（建议跑一次 A/B sanity）。
- **HF `nvidia/GEAR-SONIC` 没有多动作 robot_filtered**（只 walk）；30G `bones_seed_smpl` 是 SMPL human data，不是 robot_filtered 必经前置 → **别为拿动作下 30G**。
- **降调**：deploy demo 可能精选"能跑"的 → 路 A 全过是 **go/no-go 信号**，不等于"覆盖任意武术"。

## 扩容优先级（评审一致）
路 A（13 条 demo，验架构）→ **Bones-SEED 的 G1 retargeted CSV 子集**（关键词筛 dance/kick/jump，同转换器，非 30G SMPL）→ LAFAN→robot（有物理 gap，最后）→ 全量 30G SMPL。

## T1 选定（2026-06-03）：合成 token 数据集 → finetune GR00T → Isaac 内解码（绕开 VR+C++）
路 A 已跑通（7 条 demo SONIC 跟踪 OK，`scripts/gear_sonic_demo.sh`）。下一步 = **T1 第一步：建 `UNITREE_G1_SONIC` token 数据集**（用户 2026-06-03 选定）。
官方端到端 workflow 实锤（repo `docs/source/tutorials/vla_workflow.md`+`vla_inference.md`+`gear_sonic/scripts/run_vla_inference.py`）：
- **设计正是架构侧**：prompt→GR00T N1.7(2.5Hz)→`action.motion_token`(64)+左右手各7=78维→SONIC decoder(50Hz)→关节；prompt 运行时 `t <text>` 改。
- 🔴 **无现成 G1-SONIC VLA ckpt**，base N1.7-3B 吐不出合法 token，**必须自己 finetune**；官方 deploy 走 ZMQ+C++（连 `--sim` 都要 C++ build）→ 所以走 T1 替代。

### 建数据集的验证过的接口（都读源码确认）
- **写库直接复用官方 `gear_sonic/data/exporter.py:Gr00tDataExporter`**（继承 LeRobotDataset）：`.create(save_root,fps=50,task,features,...)` + `.add_frame(frame_dict)` + `.save_episode()`。schema/modality 用 `gear_sonic/data/features_sonic_vla.py:get_features_sonic_vla / get_modality_config_sonic_vla(robot_model)`。
- **action 关键字段**：`action.motion_token` shape(64) float64（其余 smpl_*/planner_*/vr_* 是 teleop 遥测，盲跳舞可填 0/省）；video=`observation.images.ego_view`(480×640)；prompt 进 `annotation.human.task_description`(=add_frame 的 `frame["task"]`)。
- **token 抽取点（零侵入）**：eval 每步调 `cb.eval_step(env,results)` 且循环给 `callback.model=model` → 写个 **recorder eval-callback** 即可拿 `model.policy.actor_module`(UniversalTokenModule)。Hook 住 `actor_module.decode(name,decode_input_dict)`，入参 `decode_input_dict["token_flattened"]` 就是 64 维 motion_token（= 官方 `proprio["token_state"]`）。GR00T 训练目标若要 pre-quant 连续 latent 用 `actor_module.encode(name,obs,no_quantization=True)`。
- **相机**：env cfg 自带，`++enable_cameras=True`（或 `render_ego=True`）建 `self.ego_camera`(TiledCamera, prim `{ENV_REGEX_NS}/Robot/ego_camera`，默认[108,192]可调)，recorder 读 `env.scene["ego_camera"].data.output["rgb"]`。**不用 dummy 帧**。
- **lerobot 写库 segfault 坑**（[[lerobot-v040-convert-segfault-fix]]）：多 ep 写 LeRobotDataset 随机崩=dual-ffmpeg 堆损坏；若复现，encode 改 ffmpeg-CLI 子进程。先小样本（1 motion×few step headless）smoke 验证再放量。
- 注入用 eval-callback（注册进 `config.callbacks` + `eval_callbacks`），别改 eval 主循环；启动复用 `gear_sonic_demo.sh` 同款 hydra override。

### T1-step1 recorder 已跑通（2026-06-03，验证过）
- **产物**：`gear_sonic/data/vla_token_recorder.py`（submodule，eval-callback）+ `scripts/gear_sonic_record.sh`（headless 逐条录）。裸 npz 出在 `datasets/sonic_vla_raw/<motion_key>/episode_NNN.npz`。
- **npz schema**：`motion_token(T,64)f32` + `joint_pos(T,29)f32` + `root_quat(T,4)wxyz` + `ego_rgb(T,480,640,3)u8` + `fps=50` + `prompt` + `motion_key`。
- **token 验证实锤**：值落 **1/16 量化网格** `[-0.5625,0.5]`（FSQ levels=16），逐帧变化 → 确是 SONIC 真 token。抽取 = hook `actor_module.decode("g1_dyn",d)` 取 `d["token_flattened"]`，reshape(-1)→64。
- **启动 override**（都在启动器里）：`+headless=True ++num_envs=1 ++manager_env.config.enable_cameras=True "++manager_env.config.cameras.camera_resolution=[480,640]" ++manager_env.commands.motion.{use_paired_motions=True,motion_lib_cfg.{motion_file=...pkl,filter_motion_keys=<key>,smpl_motion_file=dummy}} "++eval_callbacks=[vla_recorder]" ++callbacks.vla_recorder.{_target_=gear_sonic.data.vla_token_recorder.VlaTokenRecorder,output_dir,motion_key,prompt,max_steps,episode_tag}`。
- **踩过的坑**：① callback 必须有 `on_step_end(*a,**k)`（eval 循环对所有 cb 调；顺便从 kwargs 收 model）；② `enable_cameras`/`cameras` 在 **`manager_env.config.*`** 命名空间不是顶层（`env_config=config.manager_env`）；③ token 多一维要 reshape；④ eval 循环结尾自带 `os._exit(0)`，teardown 不卡。
- **ego 相机看地面看不到自己**（attach 到 pelvis 朝前）→ **符合部署现实**（真机相机看世界），盲跳舞视觉本就弱，保留 ego 不要第三人称（否则 train/deploy 失配）。
- **diversity**：env 自带 startup DR（physics_material/joint_default_pos/base_com/body_mass randomize），每次 fresh boot 不同 → 跑 N episodes（N 进程）天然有 DR 多样性；RSI 可加 `sample_from_n_initial_frames`。

### T1-step1 converter 已跑通（2026-06-03，全链路打通）
**SONIC eval → 抽 token → npz → convert → GR00t LeRobot v2.1 数据集**，端到端验证过。
- **走法 A 落地**：`bash install_scripts/install_data_collection.sh` 装好 `.venv_data_collection`（uv,py3.10,lerobot **0.1.0**=老 `lerobot.common` 布局 + gear_sonic + PyAV/OpenCV）。converter = `scripts/sonic_vla_npz_to_lerobot.py`，**在 `.venv_data_collection` + cwd=WBC 目录跑**（gear_sonic 资产路径相对 WBC）。
- 用官方 `Gr00tDataExporter.create(...).add_frame(frame).save_episode()`，schema=`features_sonic_vla`。**遍历 features 把没采的 teleop 字段零填**保 `validate_frame` 过；实填 motion_token/observation.state/action.wbc/root_orientation/ego_view。
- **num_joints=43**（含手指）但录的 joint_pos 是 29 → `_fit` 零填 29→43（observation.state 后 14 维是零，盲跳舞无碍；训练目标是 motion_token 不是关节）。
- **产物** `datasets/sonic_vla_lerobot/`：codebase_version **v2.1**、fps50、7 ep/3815 帧、`meta/modality.json`(action.motion_token[0:64])、tasks.jsonl 7 prompt、ego_view mp4。load 回测 motion_token(64,)[-0.5,0.44] + ego_view(3,480,640) 全过。**3.2M**（mp4 压地面视角极小，裸 npz 584M）。无 segfault。
- **下一步**：① 放量 `EPISODES=N bash scripts/gear_sonic_record.sh`(DR 多样性)凑 50-100 demo/任务 → 重跑 converter；② 进 Stage C：Isaac-GR00T `launch_finetune.py --embodiment-tag UNITREE_G1_SONIC` finetune → in-process decode 部署。

### Stage C finetune 已跑通配方（2026-06-03，单 4090）
**全部前置已就位**：`dependencies/Isaac-GR00T` = N1.7（有 `gr00t_n1d7` + `.venv` torch2.7.1+cu128/cudnn90701 干净栈）；`unitree_g1_sonic` embodiment **已注册**（`gr00t/configs/data/embodiment_configs.py:67`，action=[motion_token,左右手joints] horizon40 ABSOLUTE，video=ego_view，lang=task_description）；base **N1.7-3B(6.5G)+Cosmos-Reason2-2B 都在 HF cache**（无 gated 阻塞）。
- 启动器 `scripts/gr00t_sonic_finetune.sh`（`SMOKE=1` 2步验证 / `MAX_STEPS=N`）。冻结 VLM、训 DiT头+projector（**可训练 1.62B/51.5%**）。
- **两个 skill[gr00t-4090-finetune] 没覆盖的本 launcher 特定坑（OOM 根因）**：① `launch_finetune.py:96` **硬编 `optim="adamw_torch"`**（Adam 动量 state→24G OOM @22.7G），FinetuneConfig 不暴露 optim/grad_ckpt → 改成读 `GR00T_OPTIM`(默认仍 adamw 不破坏) + `GR00T_GRAD_CKPT`；② `experiment.py:227` 的 TrainingArguments **缺 `gradient_checkpointing_kwargs`** → 加 `{"use_reentrant":False}`（冻结 VLM+训 DiT 时 reentrant=True 报无梯度）。**这两处是 Isaac-GR00T 改动，要转 patch**。
- 修后 `GR00T_OPTIM=adafactor GR00T_GRAD_CKPT=1` + global_batch=4/grad_accum=4/1GPU(micro=1) → 峰值 **22688 MiB 不 OOM**，2 步 smoke loss 1.18 exit0。skill 4 个 env flag(COMPILE_ACTION_HEAD_DISABLE/DATALOADER_NUM_WORKERS=0/PIPELINE_OVERLAP_DISABLE=1/expandable_segments)照搬。
- **embodiment 已设 rep=ABSOLUTE → launcher 硬编 `use_relative_action=True` 不触发双重相对化**（只 rep==RELATIVE 才触发），motion_token 安全。
- derisk 目标：2000 步看 train_loss 降不降（能降=GR00T 学得会 (ego,prompt)→token，go；plateau=查视觉瓶颈/token回归）。

### Stage C derisk 结论（2026-06-03）：✅ 学得会，欠训，方向 GO
2000 步跑完 exit0，train_loss 1.18→0.50 平滑单调降（DiT diffusion loss，绝对值不可直接判，0.5 平台是噪声底正常）。
**真判据 = `gr00t/eval/open_loop_eval.py`（预测 vs GT 动作 MSE）**，全 7 trajectory（`--action_horizon 40 --embodiment_tag unitree_g1_sonic --traj_ids 0..6`）：
- checkpoint-1000 平均 token MSE **0.0875** → checkpoint-2000 **0.0650**（**单调降 ~25%/1000步**）。
- 基线：predict-global-mean **0.048**、predict-per-motion-mean **0.039**（=各动作 token 序列方差，脚本现算）。7 动作 mean-token L1 距离 **0.114** → token 空间有真信号可分。
- **判读**：2000 步 MSE 仍高于基线（欠训，loss/MSE 都还在降），但**单调改善 = 在学习轨道上，不是根本不行**。结论 **GO**：加步数（skill 推荐 8k–20k）+ 后续多样性（RSI/更多 clip）应能压到基线下。
- 开环 eval 廉价好用，比 loss 值可信；MSE 含 14 维零手部会拉低一点，token-only 实际略高。denoising_steps=4 默认。
- ~~下一步 8000 步重训~~ **作废**：见下方闭环诊断，先别加步数。

### in-process decode 部署 + 闭环诊断（2026-06-03，codex 评审驱动）
建了 token 注入器 `gear_sonic/data/vla_token_injector.py`（hook `actor_module.decode` **替换** `token_flattened` 为外部 token）+ `scripts/gear_sonic_inject.sh` + dumper `scripts/gr00t_dump_pred_tokens.py`（GR00T venv 跑 open-loop 出每动作预测 token → `datasets/sonic_vla_pred/<key>.npz`）。**prompt→GR00T→token→SONIC decode→G1 在 Isaac viewer 全内存跑通，无 C++/ZMQ。**
- **codex gpt-5.5 评审推翻"GO/加步数"**：硬伤=每动作1ep无held-out（只能背训练集/退化成prompt均值模板）、**相位观测缺失**（ego看地面+prompt静态+state可能错位+gravity零→40-step chunk相位无源→开环漂）、**连续回归量化FSQ token是hack**（应预测pre-quant latent过SONIC quantizer，或当16-level分类）、LR@2000已近0 loss在0.5抖→0.065可能就是平台。codex判：derisk**还不能证明架构侧GO**，连per-motion-mean基线都没打过。
- **codex 的决定性实验 = 3路闭环token回放**，已跑：① 预测token注入 → G1 每**1-2s重置**；② **GT token（我录的真token）注入也每1-2s重置** → 不是模型问题。③ 根因=`base_adaptive_strict_ori_foot_xyz` 终止严（anchor_pos 0.15m/ee 0.15/foot 0.2/ori 0.2），**开环注入无actor闭环纠偏→漂移>阈值→终止**。④ **放宽终止**（`++manager_env.terminations.{anchor_pos,ee_body_pos,anchor_ori_full,foot_pos_xyz}.params.threshold=99`）→ **GT token 开环能完整复现踢腿** → **注入接口/架构层通了**，1-2s重置是终止严格度+开环的confound不是接口坏。
- **最终对照结果（放宽终止）**：GT token → 完整踢腿；**predicted token(旧2000) → 出"微踢腿"轮廓（方向对、幅度被压扁）**。

### 决定性翻案（2026-06-03 晚）：MSE 暴跌，架构侧 GO 坐实，之前是 LR-schedule 欠训
旧 2000-run loss 卡 0.50 是 **max_steps=2000 的 LR schedule 衰减到 0** 不是天花板（codex 对 LR 判断正确）。**全新 max_steps=8000 跑（让 LR 铺满）**：loss 1.18→**0.10**（step~2400），同样 step 2000 的 checkpoint open-loop MSE：
- **平均 0.0167**（旧 2000-run 是 0.065）；逐动作 dance0.019/lunge0.010/macarena0.033/kick0.023/squat0.009/jump0.011/walk0.010。
- **比 per-motion-mean 基线(0.039)低 2.3×、比 global-mean(0.048)低 2.9×** → 模型**确实学会 (ego,prompt)→token**，不是背均值。之前"高于基线"纯 LR 欠训假象。
- **架构侧 GO 确认**（接口通 + 模型真学会 + 远破基线）。caveat：仍是 7-ep 训练集，泛化(held-out)未测；macarena/kick 最难但仍破基线。
- **8k run 在 step 2429 崩**（`RuntimeError: d.is_cuda() INTERNAL ASSERT FAILED` backward 时偶发 CUDA 故障；skill gotcha#3，干净 env 也偶发）→ resume from checkpoint-2000（`trainer.train(resume_from_checkpoint=True)` 自动找最新 ckpt；SAVE_STEPS=500 最多丢 500 步）。
- 新预测 token dump 在 `datasets/sonic_vla_pred_8k/`；放宽终止注入看清晰版踢腿。

### 8k 训练完成（2026-06-03 晚，含死机恢复）：MSE 0.0011，架构 GO 强确认
- **train_loss 1.18→~0.04**（8000步）。**open-loop MSE 随步数单调暴跌：旧2000-run 0.065 → 8k@2000 0.0167 → 8k@8000 0.0011**（比 per-motion-mean 基线 0.039 低 **35×**）。逐动作 dance0.0010/lunge0.0026/macarena0.0003/kick0.0018/squat0.0009/jump0.0005/walk0.0006。kick 0.023→0.0018(13×) → 踢腿从"跨步"变利落。
- **结论坐实**：GR00T 能把 (ego,prompt)→SONIC token 学到极紧（7-ep 训练集过拟合）。之前所有"糊/跨步/高MSE"全是 **LR-schedule 欠训**，非建模天花板。codex 的"continuous回归抹平"担忧在足够训练下被克服（至少训练集内）。**caveat 仍在：泛化(held-out新prompt/新clip)未测**——这才是下一个真问题。
- **死机恢复经验（reboot mid-save 损坏）**：① checkpoint 写一半被 reboot 截断 → safetensors `incomplete metadata` 损坏（dir 变小，如 8.4G vs 完整 12G），resume 死循环卡它；② 修=删损坏 ckpt 回退到上一个完整的；③ **自愈 resume loop** `/tmp/resume_8k_heal.sh`：每次 attempt 前用 `/tmp/ckpt_intact.py` 校验所有 ckpt，损坏即删，再 resume latest intact + crash 自动续。
- **checkpoint 双层保留**（用户偏好 [[feedback-training-save-policy]]）：`SAVE_LIMIT=3` 滚动临时（崩恢复）+ **阶段点 hardlink 到 `outputs/.../keep/`**（`/tmp/keep_stage_ckpts.sh`，2000倍数，`cp -al` 同 inode 零额外磁盘，HF 滚删原件 inode 仍由 keep/ 保住）。keep/ 存 2000/4000/6000/8000。每 ckpt 12G。
- **下一步（真问题=泛化）**：① 拿 held-out（新 episode/新 phase/改写 prompt）测 MSE 看是否仍低（过拟合 vs 真泛化）；② 多 clip + RSI 扩数据；③ 闭环（非放宽终止）下 predicted token 的 tracking 误差/相位漂。脚手架：`scripts/gr00t_sonic_finetune.sh`(SAVE_LIMIT/GR00T_OPTIM/GR00T_GRAD_CKPT env)+ `gr00t_dump_pred_tokens.py` + `gear_sonic_inject.sh` + open_loop_eval。
- **下一步（codex 序）**：先别 8k。① 修 observation.state joint mapping + 补真 projected_gravity；② predicted-token 放宽终止闭环 vs GT vs per-motion-mean 对照（看动作质量/相位漂）；③ 若预测明显劣于 mean → 改 token 目标形式（分类/pre-quant latent）+ 加 held-out 多clip，再谈步数。

### 部署/可视化基建 + 一键化（2026-06-03 晚，compact 续接点）
- **注入器** `gear_sonic/data/vla_token_injector.py`(submodule)：hook `actor_module.decode` 替换 `token_flattened` 为外部 token。支持 **per-env 多 token**（`token_npz` 逗号分隔，env i 用 list[i]）→ 老师/学生并排。`tint_env0_green` 选项**已废**（见下）。
- **一键脚本**：`scripts/gear_sonic_inject.sh <motion>`（默认 `datasets/sonic_vla_pred_8k_final/` 的 8000-token + `RELAX=1` 放宽终止 + `debug_vis=True`）。预测 token 在 `datasets/sonic_vla_pred_8k_final/`（7 条全有，checkpoint-8000 dump）。
- **多-env 并排对比配方（codex 验证，重要）**：单视图只见 1 个机器人 = 相机自动跟随 **env_0**（`manager_env_wrapper.py:169 update_view_to_asset_root`）+ 默认 `eye=[4.5,0,4]` 从 +X 看，env_1 在后方。修法：`++num_envs=2 ++manager_env.config.terrain_type=plane ++manager_env.config.env_spacing=3.0 '++manager_env.config.viewer.eye=[0.0,-7.0,3.0]' '++manager_env.config.viewer.lookat=[0.0,0.0,1.0]'`。⚠️ `env_spacing` 在 **`manager_env.config.env_spacing`** 不是顶层；hydra 逗号 list 值（eye/token_npz 多路径）要 **单引号**。
- **🔴 机器人本体逐 env 上色 = 死路**：robot 是 instanceable 资产，mesh 是只读 instance proxy；`SetInstanceable(False)` 去实例化能绑色(bound 50 meshes)但**破坏物理 articulation**（"Failed to set DOF actuation forces"）。结论：用左右**位置**区分老师/学生，别给本体上色；真要参考可视化用 `goal_pos_visualizer` 绿球 marker（独立 prim 不碰物理，`debug_vis=True` 单按 V 因 `not True=False` 反而关，需正确 `set_debug_vis`，未深究）。
- **SONIC.ipynb ④ 区**：自训全流程一键（4.1 record / 4.2 convert / 4.3 finetune / 4.4 open_loop_eval / 4.5 dump / **4.6 七条 GR00T 驱动动作各一键 `gear_sonic_inject.sh <motion>`** / 4.7 老师vs学生并排）。③ 区是 SONIC 原生动作一键。
- **prompt 性质澄清**：训练已 prompt-conditioned，但当前是**开环回放**（离线 dump token 再注入），**非实时下 prompt**。实时"敲 prompt 当场动" = GR00T 在环 live get_action（in-process 或官方 ZMQ）+ 需泛化。
- **✅ 已整理待提交（2026-06-03 收尾，用户自己 commit）**：
  - **submodule 编辑全转 patch + apply 脚本**：① WBC 新文件 `vla_token_{recorder,injector}.py` 被 WBC `.gitignore:209 data/` 吞（`git diff` 抓不到，**必须 force-add**）→ `patches/gear-sonic/0003-vla-token-recorder-injector.patch`（396 行 creation diff）；② Isaac-GR00T `launch_finetune.py`+`experiment.py` 6 行 → `patches/gr00t-n17/0001-finetune-oom-fit-adafactor-gradckpt.patch` + 新 `scripts/apply_gr00t_n17_patches.sh`（照搬 `apply_gear_sonic_patches.sh` 幂等模式）。两 apply 脚本 dry-run 全 SKIP。
  - **归属厘清**：WBC `features_sonic_vla.py`/`exporter.py` = WBC 已 tracked 提交；`unitree_g1_sonic` embodiment = Isaac-GR00T **已 commit**（3df8b38），非我改动不用 patch；`download_from_hf`/`g1.py` 由 gear-sonic 0001/0002 覆盖；Isaac-GR00T `lerobot_episode_loader`(mmap)+`server_client`(N1.5/6/7 wire)+`pyproject/uv.lock` = **早先 LeIsaac 工作流**不归本路线。
  - **自愈脚本固化**：`/tmp/{resume_8k_heal,keep_stage_ckpts,ckpt_intact}` → `scripts/gr00t_{resume_heal,keep_stage_ckpts,ckpt_intact}.{sh,py}`，硬编路径提成 `OUT`/`MAX_STEPS`/`STAGE` env，`/tmp` 原件已删。
  - **gitignore**：`outputs/`(33)+`datasets/`(43) 整目录已忽略 → 无需改；零大文件入 git。
  - **doc**：`doc/groot_sonic_wbc_route.html` 加 §10 derisk 复盘（管线 SVG + MSE 表 0.065→0.0011 + 诚实边界 + 绿老师死路 + 下一步闭环 + 复现脚手架表）。
  - **本 repo 待 commit**：`scripts/{gear_sonic_record,gear_sonic_inject,gr00t_sonic_finetune,gr00t_dump_pred_tokens,sonic_vla_npz_to_lerobot,gr00t_resume_heal,gr00t_keep_stage_ckpts,gr00t_ckpt_intact,apply_gr00t_n17_patches}` + `patches/{gear-sonic/0003,gr00t-n17/0001}` + `doc/groot_sonic_wbc_route.html`(M) + `SONIC.ipynb`(M) + `.memory`(M)。submodule pointer `Isaac-GR00T`(m)/`lerobot`(m) 不提交。
  - **磁盘提醒**：`outputs/gr00t_sonic_8k` 83G（rolling 7000/7500/8000 + keep/2000-8000）、`outputs/gr00t_sonic_derisk` 47G（2000 步旧 run，**已被 8k 完全取代**，可清）。共 130G。
- **下一步（真问题）**：① held-out 泛化（新 prompt/新起始/新 clip 测 MSE，现 7-ep 过拟合）；② 实时 prompt 活循环；③ 多 clip+RSI 扩数据（DR 放量实测 95% 重复无用）；④ 闭环纠偏治长动作开环漂。

## ✅ 闭环活循环已跑通 + 关键发现（2026-06-04）
- **闭环桥架构**：GR00T N1.7（transformers 4.57.3 venv）与 SONIC eval（Isaac conda）**不能同进程** → GR00T 自带 ZMQ server (`gr00t.eval.run_gr00t_server --model_path <ckpt> --embodiment_tag unitree_g1_sonic --port 5555`) + Isaac 端独立 wire client。脚手架：`scripts/gear_sonic_live.sh <motion>` + `gear_sonic/data/vla_live_injector.py`(VlaLiveInjector callback，**data/ 被 gitignore 同 recorder/injector，转 patch 要 force-add**) + `doc/sonic_vla_closeloop_validation.html`。
- **wire 协议**：`PolicyClient.get_action(obs)` 包成 `call_endpoint("get_action",{"observation":obs,"options":None})`，server 端 `handler(**data)` 展开 → 必须键名 `observation`；返回 `(action,info)`，token 在 `action["motion_token"]` (无 prefix，需 `[0]` 去 batch)。序列化 = `msgpack_numpy.packb/unpackb`。conda env 有 zmq/msgpack_numpy。
- **obs 组装（1:1 复刻 converter）**：video/state/language 全 `delta_indices=[0]`（**只当前帧，无历史窗，不缓冲**）。state = `_fit(robot.data.joint_pos,43)` 按 `meta/modality.json` 切 left_leg[0:6]/right_leg[6:12]/waist[12:15]/left_arm[15:22]/left_hand[22:29]/right_arm[29:36]/right_hand[36:43] + `compute_projected_gravity(root_quat)`(3) + ego cam `scene["ego_camera"].data.output["rgb"]`(480×640) + prompt。每组 reshape (1,1,d)。**server 内部归一化 → 发原始值**。action_horizon=40 → 每 40 步查一次缓冲逐步喂。
- **🔑 核心发现（无记忆 obs 条件策略的本质）**：GR00T 单帧 obs→token，闭环里**只能延续观测到的姿态相位**，无法自发从静止重启一次性动作。逐层诊断证 GR00T 没坏（live token 正确匹配对应相位 dump token）、obs 没喂错（state L2=0.05 匹配录制帧，cam mean 对得上）。
- **P2 全 7 动作扫描（dpose=查询间关节位移 L2，判动/不动）**：**自持 ✅** squat/lunge/dance/macarena（dpose ~1.5-2.5）；**卡静止 ❌** 仅 kick(3.3s单踢→0.06)、walk(走+转身停→0.06)；jump 边缘。判别 ≠ 严格周期性，而是「是否 settle 到策略逃不出的通用站立姿」。**先前"周期vs一次性"二分被全扫描推翻，如实修正。**
- **P-fix 触发式 bootstrap（免重训救一次性动作）**：`VlaLiveInjector` 加 `bootstrap_npz`+`bootstrap_steps`，复位后前 N 步喂开环 dump token 带进动作再交 live。`BOOTSTRAP=80 bash scripts/gear_sonic_live.sh kick` → dpose 0.06→**1.42 持续踢**。证明 live GR00T 能接续中段动作。
- **踩坑**：① IsaacLab `app_launcher.py:561` 读 env var `HEADLESS` 做 `int()` → `HEADLESS=True` 泄漏崩 `int('True')`；脚本读完 `unset HEADLESS` 只用 `+headless=` arg。② headless 下 `debug_vis=True`(绿骨架)报错，只 GUI 加。③ 单 GPU 连跑多 eval，timeout 杀 wrapper 但 python 子进程 orphan 占 GPU，累积 4 个撑爆 24G(`create_articulation_view:NoneType`)→ run 间按 PID 显式 kill。④ `pkill -f eval_agent` 自杀（当前 shell 命令行含该字串）→ 按 PID 过滤排除 `$$`。

## 🚀 已发布到 HF（2026-06-04）
- **模型** [`wsagi/GR00T-N1.7-G1-SONIC`](https://huggingface.co/wsagi/GR00T-N1.7-G1-SONIC)：checkpoint-8000 推理权重（3 shard 4.7/4.7/2.5G）+ 7 个闭环 demo mp4 内嵌 model card（`<video>` + resolve full URL）+ 诚实 scope（仅训练集/kick&walk需bootstrap/依赖SONIC栈）。`base_model: nvidia/GR00T-N1.7-3B`（**实测起点，非 Cosmos**；Cosmos-Reason2-2B 只是内部冻结 VLM backbone）。22 文件。
- **数据集** [`wsagi/SONIC-VLA-LeRobot`](https://huggingface.co/datasets/wsagi/SONIC-VLA-LeRobot)：LeRobot v2.1（7 动作 3.3M）+ 7 ego mp4。**血缘**：`source_datasets: bones-studio/seed`（SONIC 训练源）+ 正文链 `nvidia/GEAR-SONIC`（直接上游：产 token 的 WBC + demo_robot_filtered.pkl 的出处）。23 文件。
- **血缘闭环**：`bones-studio/seed → nvidia/GEAR-SONIC → wsagi/SONIC-VLA-LeRobot → wsagi/GR00T-N1.7-G1-SONIC`。
- **录屏源**：`~/视频/录屏/GR00T-N1.7-G1-SONIC-*.webm`（7 条），转 mp4 用 **系统 `/usr/bin/ffmpeg`**（conda ffmpeg libx264.so.138 坏）+ `env -u LD_LIBRARY_PATH` + `-vf scale=trunc(iw/2)*2:trunc(ih/2)*2`（录屏 496x367 奇数高，H.264 要偶数）。
- **上传踩坑全记进 [[hf-upload-tricks]] + skill `hf-publish-model`**：多 GB 目录走代理别用 upload-large-folder（批量 commit 整条断），用逐文件 + **每次 `timeout` 包住**（99% 是 commit hang 不报错，retry 循环不触发）；`publish.sh` 已重写成 per-file+timeout。

## ⚠️ 闭环阶段新未提交（接着上次"你整理用户自己commit"那批之后）
- 本 repo：`scripts/gear_sonic_live.sh`(新,含 HEADLESS/BOOTSTRAP/RELAX) + `doc/sonic_vla_closeloop_validation.html`(新) + `SONIC.ipynb`(M?) + `.memory/{sonic-wbc-vla-route,hf-upload-tricks}.md`(M)。
- WBC submodule：`gear_sonic/data/vla_live_injector.py`(新,**被 data/ gitignore 吞,转 patch 要 force-add**,同 0003 套路 → 建 `patches/gear-sonic/0004-vla-live-injector.patch`)。
- 全局 skill/memory（在 `~/.claude/`，不在 repo）：`hf-publish-model` SKILL.md + scripts/publish.sh 已改;`hf-upload-tricks.md` 已补。

## 架构侧 Stage C 要点（评审给的，等路 A 过了再用）
- token 注入：`UniversalTokenModule.decode("g1_dyn",{token_flattened(64),proprioception})` 或 `forward(latent_residual_mode="pre_quantization_replace")`。in-process，免 C++。
- **GR00T 训练目标用 pre-quantization 连续 latent**（非分类、非量化后值），推理时过 SONIC 同一 FSQ quantizer 再 decode（codex+mimo 一致）。
- in-process decode 要复用 SONIC 的 obs 构造 / proprio history / 50Hz+token插值；`sonic_release` 的 `running_mean_std:false`，真正要对齐的是 obs term 布局/history/clipping。
- 14 维手部绕过 WBC（无平衡兜底），要单独处理（mimo 提）。
- 必须三曲线对照：官方 SONIC eval vs external-token playback vs direct-decode，逐帧比动作均值。

关联：[[gear-sonic-preview-setup]]、[[mimickit-to-vla-distill-plan]]（数据侧对照）、[[gr00t-multi-release-env-split]]（N1.7 env）、[[mimickit-lafan-fight-training-plan]]（LAFAN dance/fight 来源）。
