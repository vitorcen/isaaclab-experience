---
name: sonic-closeloop-freeze-rootcause
description: SONIC闭环kick/walk/jump冷站立不发起=单帧OOD观测问题(站立obs→hold token),非WBC/非架构;WBC basin探针9/10证锅在VLA;解药=aug2组合(物理rollout增广+onset-loss加权+stand),首个三动作冷启动都能起但~50-100%随机;已发布V2
metadata:
  type: project
---

# 🧊 SONIC闭环 kick/walk/jump 冷站立不发起 — 根因 + 解药(2026-06-26 收敛版)

**现象**:GR00T-N1.7-SONIC头闭环里 **kick/walk/jump 从冷站立(Isaac reset / stand 哨兵)不发起、站着不动**;squat/dance/lunge/macarena 正常。开环 token-MSE 极好(0.0011)却闭环冻 → 开环≠闭环。

## 根因(多轮单变量受控实测三方triangulate,已收敛)
**= 单帧无历史策略的 OOD 观测问题**:训练里站立帧压倒性多数且全教「→hold」,部署在站立/漂移姿(分散偏 ~0.54rad,无单点 wiring bug)单帧策略吐 hold chunk(drift 0.06 vs 真动作 0.25)→ WBC 忠实保持 → 站立死锁。**不是** WBC 起不来、**不是** conditioning、**不是** 动量/重力/图像。
- **prompt 有效但被站立 state 先验压住**:prompt×proprio 矩阵证 prompt 选动作;但冷站立 obs 下所有 prompt 都塌成 hold(state-dominant)。
- **关键反转**:从**动态态/中途切入**(序列 squat→walk→jump→kick 不reset)所有动作含 kick **都能起** → 模型学会了动作,冻结只发生在「冷站立零速度深 basin」这一窄 case(用户 1.2a「先做 work 动作再切」即绕开)。
- **🟢 WBC basin 探针(决定性,token injector 进程内喂 GT onset token)**:10 探针 **9 LAUNCH**(静止站立 + 中途动量都能起 kick/walk/jump,唯一 FELL 是 mid-dance 失衡源姿态污染)→ **WBC basin 宽,锅 100% 在 VLA 端(没吐 onset token)**,数据增广路线开绿灯。
- **onset 形状决定难度**:kick/walk onset = 突兀大跳变(首12帧 jointΔ 0→0.8)且 onset token_std≈hold → mean-loss 默认 hold 不发起;jump onset 平缓爬坡(0→0.26)→ jump 自启、kick/walk 不启。

## 解药 = aug2 组合模型(2026-06-26,首个三动作冷启动都能起)
三法互补,缺一不可:
1. **物理 WBC-rollout 增广**:`[1s 他动作动量]++[目标 onset]` token 序列经 SONIC WBC 真物理回放再录 `(token,ego_rgb,proprio)`(非帧拼接=非物理);LAUNCH 过滤、**剔所有 `*_from_kick` 源**(kick 动量 rfoot~0.9 污染目标 onset)。→ 修 walk(可靠)+ jump(部分)。
2. **onset-loss 加权**(`ONSET_LOSS_WEIGHT=4`,前 12 token 位 ×4):强迫 commit onset。→ 修最难的 abrupt kick。
3. **4 个 stand episode**(macarena 站立前段,prompt=stand):学自然 idle。
- **四模型 head-to-head 同 harness 冷启动 LAUNCH 率**:8k-baseline kick0/walk0/jump0(全冻);onsetw-6000 kick2/3 walk1/3 jump0/3(强 kick);aug-8000 kick0/3 walk3/3 jump1/3(强 walk);**aug2 kick1-2/2 walk1-2/2 jump2/2(三动作都起)**。
- **诚实**:仍**随机**(每动作 ~50-100%,非确定 3/3;同 ckpt run1 起 run2 冻)=单帧架构对 onset 的根本随机性;确定性需 history/CFG。

## 负结果档(已证伪,别再重跑)
- **输入噪声 σ=0.08**:jitter 站立教不会「发起」(缺时间信号非输入鲁棒)。
- **idle 自然微动**:留在深站立 basin 内→仍冻;决定因素=「是否离开 basin」非「是否在动」。
- **Tier-2 CFG**(`out=uncond+s·(cond−uncond)`):弱旋钮;冷 OOD 站立处 cond/uncond 都塌 hold,`|cond−uncond|<` 网格步,放大的是噪声非 kick。s=1 留作无害 no-op。
- **re-anchor**(切动作前 teleport 写回干净站立姿):A/B/C 隔离三个全冻 → 只修「摔倒/漂移→拉回」,**救不了 onset**(姿态对了预测层仍 hold)。用户实测拒绝(视觉复位+经常没生效)。
- **纯物理 aug 单独**:修 walk/jump 不修 kick(0/3)。**onsetw 单独**:修 kick 弱 walk/jump。**onsetw2 续训/更高 weight**:kick P(fire) 没提 + 害 walk(退回冻)。
- **history/velocity 架构重训**(codex):velocity 静止=0 + 3帧 OOD 仍 OOD → 只改善 sustain/一般闭环质量,**救不了冷启动首预测**(walk 大概率无效)。GR00T-N1.7 原生支持 history 但有 5 个 silent blocker(负历史时序泄漏/warm-start 非自动/config 覆盖/随机 encoder 表征断层/速度语义),MODERATE 改动。doc `LeSONIC/doc/sonic_obs_history_velocity_retrain.html`。

## eval 方法论(踩出来的硬规矩)
- **开环 token-MSE = 粗筛淘汰,不选 best**(单调最低=过训反而全冻;8k 开环 0.0011 却闭环冻);**闭环必须**。见 [[feedback-three-tier-eval-funnel]]。
- **闭环判据 = settle-then-switch**(model/哨兵 stand settle→切 prompt→量 rfoot/disp/joint_exc,`VLA_DIAG_TRACE`);**不要 basin-sweep-from-Isaac-reset**(reset 死姿 OOD artifact=误导);**不要 tok_std**(kick onset std≈hold 不可靠,用 dpose/rfoot)。baseline 冻锚:rfoot 0.038/exc 0.67;真踢:rfoot~0.77/exc~3.3。
- gravity 已证非问题(数据真值非零、部署没置零、一致)。

## 可复用脚手架(关键已 committed;/tmp 的 reboot 即失)
- 数据生成:`scripts/sonic_build_aug_transitions.py`(7动作×(cold+6源),macarena裁6s/walk2.5s,源取1s动量)→ `vla_token_injector` 录制(`VLA_RECORD_DIR`)→ `scripts/sonic_merge_aug_to_lerobot.sh`(base+LAUNCH-aug+stand,剔 kick 源,需 `PYTHONPATH=$WBC_DIR`)。
- 训练:`scripts/gr00t_sonic_finetune.sh`(`ONSET_LOSS_WEIGHT=4`);**血泪坑:editable `gr00t` 解析到伞仓 `dependencies/Isaac-GR00T` 非 LeSONIC 副本→ loss/增广改动写伞仓那份**(injector 在 LeSONIC 副本是对的,方向相反)。CUDA invalid-resource-handle / sharded-loader futex 死锁周期崩 → auto-resume supervisor;wedge 需 reboot(密存 ckpt 使只丢~0步)。
- 开环 `scripts/sonic_gr00t_openloop_mse.py`;FSQ clamp 修正(`_snap_fsq_grid` ±0.5→±1.0,GT 实达 ±0.8125 步 1/16 num_levels 32)。
- 启动机制 + GPU 纪律见 [[feedback-shared-gpu-eval-queue-orphan-discipline]](setsid&disown 会被工具回收→用 run_in_background;py3.10 import-torch 间歇 segfault→每格重试3次)。

## 部署 + 发布 + 待提交(2026-06-26)
- **当前部署**:BonesSeed.ipynb 1.1/1.2/1.2a 自动解析 `outputs/gr00t_sonic_aug2/checkpoint-8000` → 否则下载 V2;1.2a 用哨兵站立(不设 `VLA_STAND_MODEL`,aug2 的 model-driven stand 仅 4ep/600帧太少仍乱动 exc3.5);flow2 顺序 squat→jump→dance→walk→lunge→kick→macarena。
- **已发布 V2**:模型 `wsagi/GR00T-N1.7-G1-SONIC-BonesSeed-V2`(=aug2-8000,bf16 重存 6.3G 用 `scripts/gr00t_resave_bf16.py` 剔训练态)+ 数据集 `wsagi/SONIC-VLA-BonesSeed-V2`(LeRobot v2.1=标准 parquet+mp4 旁挂+`meta/modality.json`,**别拆扁平 parquet=丢 modality.json 训练管线读不了**,54ep/12630帧);录像 webm→mp4 用 `/usr/bin/ffmpeg`(conda 那个缺 libx264)+ 奇数宽须 `-vf scale=trunc(iw/2)*2:...`;大文件走 `scripts/hf_upload_v2_perfile.sh`(逐文件 timeout+retry,非 folder/upload-large-folder=TUN 下 xet 卡死)见 [[hf-upload-tricks]]。
- **提交状态**:LeSONIC 已提交 **4 笔**(aug pipeline / eval scaffolding / docs / V2 publish+notebook,全 `feat|docs(sonic):` 单行,用户自 push)。**仍待处理(各自独立 repo)**:① `dependencies/GR00T-WholeBodyControl` injector 改动 gitignored → 重生成 patch `patches/gear-sonic/0004-vla-live-injector.patch`;② `dependencies/Isaac-GR00T:gr00t_n1d7.py`(ONSET_LOSS_WEIGHT);③ 伞仓 `.memory`(本文件已精简)。
- **下一步(可选)**:多 stand episode 修 model-driven stand;history+CFG 压 onset 随机性求确定性。

关联 [[sonic-vla-critique-roadmap]]、[[sonic-masked-ce-p0-result]]、[[sonic-live-switch-idle]]、[[feedback-lesonic-similarity-leaderboard-standard]]、[[hf-collections-auto-place]]、[[feedback-hf-readme-project-links]]。
