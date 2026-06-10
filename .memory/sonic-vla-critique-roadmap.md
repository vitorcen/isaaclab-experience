---
name: sonic-vla-critique-roadmap
description: GR00T-N1.7-G1-SONIC-BonesSeed HF 模型页三模型联合评审(2026-06-08)9条共识缺陷 + P0-P4 修复 roadmap;别重复评审
metadata:
  type: project
---

# GR00T-N1.7-G1-SONIC-BonesSeed 三模型联合评审 + Roadmap(2026-06-08)

评审对象 [`wsagi/GR00T-N1.7-G1-SONIC-BonesSeed`](https://huggingface.co/wsagi/GR00T-N1.7-G1-SONIC-BonesSeed)。
Claude Opus 4.8 + codex gpt-5.5 + mimo-v2.5-pro 各独立批判再合并。文档 `LeSONIC/doc/sonic_vla_critique_roadmap.html`(单文件+SVG)。
背景见 [[sonic-wbc-vla-route]](架构侧路线已 GO/已发布)。

## 三方共识缺陷(独立全中=强信号)
- **🔴P0 #1 无 held-out**:7 轨迹×1ep,`MSE 0.0011` 是**记忆误差非泛化**;per-motion-mean=0.039 对 7-way 本就 trivial 基线。
- **🔴P0 #2 FSQ 离散码当连续回归**:每维 16 级,DiT flow-matching 连续回归落网格之间 decode 行为未定义。应 per-dim 交叉熵 / 预测 pre-quant latent 过 SONIC quantizer / 推理 snap-to-grid。
- **🔴P0 #3 无记忆单帧策略×闭环=本质缺陷**:单帧 obs→40-step chunk 无 history/phase,从静止启不动 kick/walk;bootstrap 是续播拐杖。**加数据治不了,是 observation space 设计问题**。
- 🟡 #4 不摔=SONIC WBC 托底功劳,VLA 贡献未解耦(需 ablation);#5 ego 相机 attach pelvis 看地面→视觉信号≈0;#6 标题"prompt-conditioned"oversell + bootstrap demo 混入 demonstrated;#7 F32 标注 vs bf16 训练(3shard~12G 偏大);#8 外部 WBC+ZMQ 跨进程门槛;#9 数据 3815 帧太小(params≫data)。
- P2 #10 checkpoint-8000 无 val curve 盲选。

## Opus 独有最尖锐(另两方没拿到细节)
**「pelvis 稳定不摔 0.79m」疑在 `RELAX=1`(终止 threshold=99)下测** = 把摔倒判据关掉再说自己不摔 = confound。
页面每条 demo 必须标 bootstrap?/RELAX? 并严格终止重测。

## 分歧裁决
- **ZMQ**:mimo 主张去 ZMQ 同进程;Opus/codex 判 transformers 4.57 vs SONIC trl 栈版本冲突是硬约束→**保留 ZMQ 文档化,别为它停泛化主线**。
- **loss 优先级**:采纳 mimo「改 per-dim 交叉熵这一刀可能比训 10× 数据有用」→ P0 第一刀。
- **ego cam**:先 ablation(vision-off vs on)证明无用再决定去留,别盲删/盲留。

## Roadmap(正序,三方一致:先 loss→再 history→再数据)
- **P0 修地基(1-2 周,零新数据)**:①loss→交叉熵/snap-grid ②建 held-out split(留1动作+每条留后半段)③F32→bf16 重存 ④页面去 oversell+标 bootstrap/RELAX。**held-out 是 go/no-go 闸门,过不了后面全是给查找表加表项**。
- **P1 治记忆(2-4周)**:⑤obs history N=4-8帧 ⑥phase clock(归一化时间戳/prompt phase hint)+random phase reset ⑦去 bootstrap 拐杖(判据)。
- **P2 扩数据(4-8周)**:⑧每动作10-50ep(变速度/幅度/起始相位)⑨held-out 新动作泛化 ⑩time-warp/phase-shift aug ⑪prompt paraphrase(数据多样性 ROI:prompt侧>DR>multi-clip 见 [[vla-distill-data-diversity-roi]])。
- **P3 闭环+解耦(与P2并行)**:⑫scheduled sampling/DAgger ⑬ablation 矩阵(WBC-replay/mean-token/prompt-shuffle/proprio-only/vision-off)⑭严格终止重测 fall rate/存活/phase error。
- **P4 实时(长期,依赖泛化先成立)**:⑮prompt transition 数据 ⑯live get_action ⑰组合 prompt"squat while wave"。

## 立刻可做(不需训练)
页面诚实化:标题改"POC 7-trajectory memorization+WBC base";MSE 改"train-set reconstruction(no held-out)";每 demo 标 bootstrap/RELAX;F32→bf16;补最小复现包(WBC ckpt commit/env lockfile/ZMQ schema/抽token脚本/eval命令)。

## P0 修复进度(2026-06-08,GPU占用→只做免GPU部分)
- **✅ #3 bf16重存**:`LeSONIC/scripts/gr00t_resave_bf16.py`(纯CPU,保持原3-shard分组,只cast float→bf16,拷aux文件)→`outputs/gr00t_sonic_8k_bf16/` 1030 BF16 tensor/6.29G(原12.58G)。**待上传HF**(对外发布要确认)。
- **✅ #1 snap-to-grid(推理侧免重训)**:铁证——录的`action.motion_token`精确落**17级均匀网格 k/16,k∈[-8,8],值域[-0.5,0.5]**(=FSQ量化码,decode不再量化直接吃token)。GR00T flow-matching吐连续值:实测预测**off-grid L1均值0.0158=半格的50.6%(≈格内随机)、60%维度偏离>0.01、值域出界[-0.812,0.688]**。修=`snap=clamp(round(t*16)/16,±0.5)`落两injector(`vla_live_injector.py`wrapped里snap`tok`/`vla_token_injector.py`load时snap),env`SONIC_SNAP_GRID`默认开,patch 0003/0004已重生成(reverse-apply验证==工作树)。**闭环关节增益待GPU A/B(SONIC_SNAP_GRID=0对照)**。
- **✅ #2 held-out切分工具**:`LeSONIC/scripts/make_holdout_split.py` leave-one-motion-out(LOMO):丢1动作整ep,重写meta(episodes/tasks/info连续重索引)+rewrite parquet(episode_index/task_index/global index)+symlink视频(省GB)+**故意省stats.json让GR00T`generate_stats`按6-ep子集重算**(它只算lowdim float、缺则regen,见gr00t/data/stats.py)。已GPU-free验证(6ep/索引连续/3391=3815-424帧)。finetune脚本直接指dst即可。**重训+held-out eval是真go/no-go信号,待GPU**。
- **⏳ #1' loss改per-dim CE/pre-quant latent**:训练侧根治,需重训(GPU);snap是其推理侧近似。
- **📝 #4 页面诚实化**:草稿`/tmp/sonic_card_revised.md`就绪,**待确认上传**。

关联 [[sonic-wbc-vla-route]] [[vla-distill-data-diversity-roi]] [[feedback-mimo-independent-review]] [[gr00t-multi-release-env-split]]。
