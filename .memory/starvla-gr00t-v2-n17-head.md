---
name: starvla-gr00t-v2-n17-head
description: GR00T_v2(代码名 QwenGR00T_N17) N1.7设计头 strict 20-round 13.3%/P(3)=0 vs v1 53.3%;但三方评审(Fable+codex+mimo)推翻"头不迁移"结论=混淆未控:特征取层错位(hidden_states[-1]第36层 vs 真N1.7 select_layer=12中层)+部分解冻是静默no-op bug;下一步=中层特征单变量复测,非解冻
metadata:
  type: project
---

# StarVLA GR00T_v2 头（代码名 QwenGR00T_N17）— 2026-06-12 落地

**目标**：冻结 Qwen3-VL-8B 下把 starVLA 的 N1.5 血统 QwenGR00T 头（榜上 53.3%）升级到
GR00T N1.7 头设计，训 `StarVLA-Qwen3-VL-8B-GR00T_v2-PickOrange` 冲 PI_v3(63.3%)/逼近 N1.7 SOTA(68.3%)。
设计文档（含 codex gpt-5.5 + mimo-v2.5-pro 双评审记录）：
`LeIsaac/docs/training/starvla_gr00t_v2_head_design.html`。

**N1.7 SOTA ckpt 实测配置**（非论文，读自 `gr00t-n17-leisaac-pick-orange-autodl-v2/checkpoint-6000/config.json`）：
DiT **16 层**×inner 1536（32×48）、AlternateVLDiT（图/文 token 交替 cross-attn）、VLLN、
state_dropout **0.2**（微调值；预训练 0.8 且数据+特征两层，照搬会叠成 36%）、**无 future tokens**、
`positional_embeddings: null`、`use_relative_action: true`（数据侧混淆因子）、头带机器人预训练。
N1.6→N1.7 NVIDIA 自己把 DiT 32 层砍回 16 层——赢在 VL 条件质量+正则，不在深度。

**已落地**（fork `vitorcen/StarVLA` `starVLA_dev`，commits `68af697`+`f4f1c8f`，未 push）：
- `starVLA/model/modules/action_model/GR00T_N17_ActionHeader.py`（FlowmatchingActionHeadN17 + AlternateVLDiT + 可选 SelfAttentionTransformer/vl_proj_dim，全 config 开关化）
- `starVLA/model/framework/VLM4A/QwenGR00T_N17.py`（image_mask = `input_ids==config.image_token_id`(=151655 已核验)，train 侧与 last_hidden 同步 repeat + 断言，predict 侧也传；registry 自动发现已验证）
- LeIsaac 配置 `scripts/training/starvla/configs/so101_qwen3vl8b_gr00t_v2{,_lite}.yaml`（未提交）
- CPU 冒烟 4 变体全过：**v2-full 612M ≈ PI_v3 634M（公平对比成立）**/lite 180M；state_dropout 实测 22%；漏 mask/漏 repeat 均硬断言。

**⚠️ 上游保留名坑**：上游 `.gitignore:264`（commit ea78f0e #316）精确忽略
`VLM4A/QwenGR00T_v2.py` = 上游有自己的 WIP v2。**代码名因此用 QwenGR00T_N17**，
"GR00T_v2" 只作文档/模型/榜单命名。registry 重复 key 是静默覆盖（import 顺序），别撞。

**云端正式训练 LIVE（2026-06-12 起，weste 盒子）**：
box = `ssh -p 16763 root@connect.weste.seetacloud.com`（RTX 4090-48G，kernel 5.15，env `/root/autodl-tmp/envs/starvla`）。
run = `so101_pickorange_qwen3vl8b_gr00t_v2`（v2-full：QwenGR00T_N17 头，冻 Qwen3-VL-8B）。
配方：rep_diffusion_steps=8 / bs=4 / 50000 步（=5.5ep，36293 帧÷4÷9073）/ save_interval 2500（20 均匀切片，50000÷2500 整除）/ keep-last 2。
**实测：100 步 smoke 不 OOM，正式训练峰值 VRAM 37.3G（距 48G 余 11G，rep=8 稳）；~0.62s/step → 50k≈8.7h**（远快于初估的 33-42h，evaluator 的 2.0-2.5s/step 估值偏悲观）。
脚手架全在 box `/root/`：`n17_train.sh`（formal）/`n17_smoke.sh`/`n17_watchdog.sh`（崩溃自 RESUME）/`n17_extract_heads.sh`（每 ckpt 抽 action_model.* head 防 frozen 16G 骨干占盘）；canonical 启动器 = `/root/run_train.sh`（VLM-agnostic，CONFIG/RUN_ID/MAX_STEPS/SAVE_INTERVAL/KEEP/BATCH env）。
**坑/经验**：① box 上 Qwen3-VL-8B 之前被清成空壳（只 refs/main 无 blobs），需重下；HF 下载 XET 401 + hf_transfer stall 双坑 → 解法 = `pip uninstall hf_xet` + `HF_ENDPOINT=https://hf-mirror.com` + 关 hf_transfer，标准下载器稳。② 这台 box 上一个 job 是 Cosmos-Reason2-8B（非 Qwen），清理时先抽 head 存档（reconstructable from frozen base）再删 34G fulls。③ Fable 子 agent 评审 N17 头 = GO 无 blocker，已加 per-sample image_mask NaN 断言。
本地 serve 就绪：`~/.cache/huggingface` Qwen8B 全（17G），本地 4090-24G 空，`serve_starvla.py` 泛型重建框架自动认 QwenGR00T_N17；8B 本机 24G 要 `STARVLA_VLM_8BIT=1`。
**arm-move 验证**：已通过（2026-06-12 用 step-100 head GUI demo，用户看过确认机械臂会动）。serve 路径 = 本地 `vlm_base_8b.pt`(qwen_vl_interface.*) + head(action_model.*) 经 `scripts/ckpt/merge_ckpt.py` 重建全 ckpt(999 key) → `serve_starvla.py --vlm_8bit` → `starvla_strict_eval.sh GUI=1`（GUI=1 去掉 --headless 显示 Isaac 窗口）；serve+Isaac 共存 ~18G/24G。
**自动 sweep eval**：`scripts/evaluation/starvla_n17_sweep_watcher.sh`（本地后台，从 step 10000 起每 ckpt 自动拉 0.6G head→重建→serve 8bit→**5-round** headless quick-eval→`outputs/starvla-n17-run/sweep.csv`）。快筛 = 5-round（2026-06-12 user 定，3-round variance 太大，见 [[eval-5round-mandatory]]）。**持久化坑**：`nohup &`/tmux 在 Claude Code Bash 工具里不跨轮持久（工具调用结束子进程被回收）——长活后台进程必须用 Bash 工具 `run_in_background:true`（harness-tracked 跨轮存活）；巡检发现 watcher 死了也这样重起，self-heal 补未 eval 的 head。拉 head 不拉 18G full（keep-last-2 会先删 full，且 head 抽取器留全部 head）；重建 full eval 后即删只留 head。峰位不确定（v1 是 30k/3.3ep，N17 头大 3× 可能早到 10-15k，但多正则或拉回）→ 从 10k 全扫画曲线，找峰后 top-1/2 跑 `starvla_strict_eval.sh`(strict 20-round) 上榜。密码用 `pass autodl/westd`（west 盒共用一个密码）。
巡检：ScheduleWakeup ~30min 心跳，盯训练健康 + sweep 曲线。

## 🟠 结果（2026-06-13）= 负面但三方评审后下调为"待复测"（混淆未控）

**⚠️ 重要更正（2026-06-13 三方评审 Fable+codex-gpt5.5+mimo）**：原"N1.7 头设计不迁移"是**错误归因**，已撤回。逐行核对头+框架接缝+真 Isaac-GR00T N1.7 参考后三点共识：
- **① [移植缺口·最可能根因] 特征取层错位**：fork 取 `hidden_states[-1]`（Qwen3-VL-8B 第 **36** 层／末层），真 N1.7（`qwen3_backbone.py`）**物理 pop 掉 13-36 层、头吃第 12 层中层特征**（`select_layer=12`）。末层特征为 next-token 语言预测优化、image 位置 hidden 已丢视觉语义——而 AlternateVLDiT 有 4 个 cross block **只看 image 位置**（强耦合），等于喂垃圾。静默退化不崩，对 v2 杀伤 ≫ v1。**"N1.7 头"从未在正确特征层测过。**（Fable+mimo 一致首要）
- **② [真 bug] 部分解冻是静默 no-op**（Fable+codex+代码实锤；mimo 误判）：`build_param_lr_groups` 按 `freeze_modules='qwen_vl_interface'` 把整个 VLM 剔出所有 optimizer group（`trainer_tools.py:132/140`），且 optimizer 在 `main()` 中先于 `prepare_training()` 的 freeze+unfreeze 构建（`train_starvla.py:462` vs `:473`）→ `requires_grad=True` 放开的顶层不在任何 group、永不更新。**照现状跑 unfreeze4=又一次 head-only 白烧。**
- **③ [次生 bug] 解冻层裸 bf16 无 fp32 master**（真 N1.7 有 `backbone_trainable_params_fp32`）；lr=1e-5 更新贴 ULP 静默量化为零。
- **无 bug 项（一致）**：AlternateVLDiT 路由/image_mask/repeat-lockstep/VLLN/flow-matching/state-dropout/去 future = 忠实移植无 bug；"Qwen3-VL 布局≠Eagle/Cosmos 致路由失效"不成立（Cosmos-Reason2 本就是 Qwen3-VL 架构）。

**🎯 评审裁决**：撤回"头设计不迁移"，降为"不完整移植（错特征层+5 delta 同开+推理延迟压低绝对值）下显著差于 v1"。**最高 ROI 决定性实验 = E1：把 `hidden_states[-1]` 改中层(≈12)后重跑 head-only**（仅改 QwenGR00T_N17.py，加 `select_layer` config）；回到 ≥v1=移植窟窿撤回结论，仍低→E2 跑 v2-lite 归因。**解冻(E3)须先修 ②③ + 给 v1 头跑解冻对照才有归因价值**——v1 用同一个 100% 冻结通用 Qwen3-VL 就有 53.3%，"冻结通用 VLM"解释不了 v1→v2 暴跌。详见设计文档 §6.5 评审块 + §12 实验阶梯。

### 原始负面记录（保留，但归因已被上方更正）

**sweep 找峰（10k→42.5k 全扫，覆盖 4.7ep；GUI 5-round headless 180）**：N17 是**早峰型**——峰在 **step-17500（1.9ep）quick-screen 53.3%**，22500=33.3% 次高，之后 25-42.5k 一路低位（6.7-26.7%）+ 40000 卡死 0%，**无二次爬升晚峰**（37500 的 33.3% 是噪声）。早峰符合"大头比 v1（3.3ep峰）早峰"的设计预判。
**strict 20-round 定真值（step-17500，headless 180 标准口径，8bit）= `8/60 oranges = 13.3%`，P(3)=0/20，P(≥2)=0/20**（每轮最多 1 颗、仅 8/20 round 放到 1 颗，全 wall_cap 截断）。raw=[0,0,0,1,0,1,0,0,1,0,1,1,0,0,0,0,1,1,1,0]。
**第二个 strict 点（step-37500≈4.1ep，2026-06-13）= `9/60 = 15.0%`，P(3)=0/20，P(≥2)=2/20=10%**，raw=[0,0,1,0,0,0,0,0,0,1,2,0,0,1,0,0,1,1,2,0]，**20 round 全 wall_cap 截断**（坐实 612M 大头推理延迟压低绝对值=评审 F5）。17500(1.9ep)13.3% ≈ 37500(4.1ep)15.0% → **N17 head-only 全 sweep 一致弱、非选错峰**（排除"挑错 step"这个备择解释）。
**对比**：v1 GR00T-8B(QwenGR00T/N1.5头) = 53.3%/P(3)35%；PI_v3 8B = 63.3%/40%；N1.7 SOTA = 68.3%/50%。**N17(GR00T_v2) 13.3% 垫底区**（榜上落 SmolVLA 25% 与 DP 8.3% 之间）。

**结论**：N1.7 的几个 delta（VLLN/AlternateVLDiT/state_dropout/无 future tokens）**在「冻 VLM + head-only + 60-demo PickOrange」这套设置下没带来任何提升，反而远不如 N1.5 血统的 v1 GR00T 头**。两个独立信号都负面：① strict 真值 13.3% ≪ 53.3%；② **P(3) 恒 0**（即便 GUI 360-cap 给够时间也从没放满 3 颗，只 ~2/3，vs v1 的 35%）——不是单纯推理慢的测量问题，是 policy 在完整任务上确实更弱。**quick-screen 的 53.3% 是 5-round 噪声乐观离群点**（又一次 20-round 必要性铁证，见 [[eval-5round-mandatory]]）。N1.7 头在 NVIDIA 那边赢是靠 Cosmos 骨干+机器人预训练+relative action（数据侧），纯头设计移植到冻 Qwen3-VL 上学不出来。

**踩坑（可复用）**：① **8bit 8B + N17 大头(612M)推理慢 ~1.3s/query → 120s sim 要 >180s 墙钟 → strict 180 口径每轮都 wall_cap 截断**（v1 小头 240M 快、能塞进 180s 不截断，所以同口径 N17 双重吃亏：policy 弱 + 截断）；GUI 更慢要 wall_cap 360 但仍非可比口径。② **strict_eval.sh 的 RETEST bug**：`flag_serve_hang.py`/`merge_valid_episodes.py` 引用 `$ROOT/scripts/benchmark/`（伞仓）但重构后搬到了 `LeIsaac/scripts/benchmark/` → 找不到报错被误判成"检测到 serve-hang"→ 假 RETEST 重跑 20 round（首轮结果有效，杀掉 RETEST 即可；待修 BENCH 路径）。③ headless eval 能跑（早先 sweep 一次卡死是偶发非系统）。

**🔭 部分解冻方案（已实现但有 no-op bug，前提已被评审推翻，降为 E3 备选）**：⚠️ 此节前提（"全冻通用 VLM 是根因"）已被上方评审推翻——v1 同样全冻通用 Qwen3-VL 却 53.3%。且实现是静默 no-op（见上 ②），照现状跑必白烧。**正确下一步是 E1 中层特征复测，非解冻。** 原始记录如下（含已更正的错误"在 optimizer 构建前"说法）：官方先例=GR00T `tune_top_llm_layers`（解冻 LLM 顶 N 层）：N1.6 默认 **4**（Eagle 通用骨干），N1.7 重新全冻 0（但**因为换了 Cosmos 具身预训练骨干，非冻结更好**）——我们的 Qwen3-VL 是通用骨干、对应 N1.6 情形该解冻；榜上自训 GR00T-N1.6 46.7% 正是 tune_top_llm_layers=4 训的。**已落地（config 开关化，向后兼容默认 0）**：① `QwenGR00T_N17.py` 加 `framework.qwenvl.tune_top_llm_layers` + `apply_partial_unfreeze()`（定位 `qwen_vl_interface.model.model.language_model.layers`[-N:] 放开 requires_grad，带 fallback）；② `train_starvla.py` freeze_backbones 后加通用 hook `if hasattr(model,"apply_partial_unfreeze"):model.apply_partial_unfreeze()`（在 optimizer 构建前，顶层进 optimizer；无此方法的框架自动 no-op）；③ config `so101_qwen3vl8b_gr00t_v2_unfreeze4.yaml`（top4，顶层 LR=1e-5 温和防遗忘，head 1e-4，rep=4 省 VRAM）。VRAM：解冻 4 层(~880M)+AdamW(~7G)+backward 激活，head-only 峰 37.3G/48G → 杠杆 rep8→4 / bs4→2 / top4→2，gradient_checkpointing 此时真生效。验收：拉到 ≳53.3% 且 P(3)>0 = 坐实全冻是根因；仍低=瓶颈更深(数据/骨干)可弃 N1.7-deltas 线。设计文档 §11。
**v2-lite 没做**（v2-full 既已负面，归因 lite 意义不大）。
**待办**：README 榜单 + 设计文档 §6/§10 + 本 memory 已记负面；问用户 17500 head 要不要发 HF 负面存档（参考 [[pi05-pytorch-expertonly-phase15-negative]]）。

关联：[[pi05-pytorch-expertonly-phase15-negative]]（同类负面归档先例）、[[starvla-vlm-variant-2b-4b-8b]]、[[starvla-8b-2b-sweep-result]]、[[umbrella-leisaac-repo-boundary]]（fork 维护规则）、[[feedback-mimo-independent-review]]
