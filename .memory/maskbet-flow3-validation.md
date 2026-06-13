---
name: maskbet-flow3-validation
description: MaskBeT v0.1/v0.2 flow3 实跑——data loader 接通 + SDPA/bf16 显存修复 + MSE-64 eval 巡查脚手架
metadata:
  type: project
---

**2026-06-11** MaskBeT 从 scaffold 进入真实 flow3 训练。关联 [[maskbet-route-b-submodule]]
[[starvla-sonic-ab-baseline]] [[g1-text-motion-datasets-hf]]。

## 数据真值（本地 `LeSONIC/datasets/sonic_vla_lerobot_flow3` 实测）
- **78 维 action = `action.motion_token[64]` ++ `teleop.left_hand_joints[7]` ++
  `teleop.right_hand_joints[7]`**。motion_token off-grid 率 **0/369728**（完美在 1/16 格，
  确认连续 FSQ 中心值非离散 id）；bin k∈[-15,13] 落在 [-16,16] 内；quantize↔centers 往返误差 0。
- **14 手指维在 flow3 里恒为 0** → 全锁 bin 16（零 bin）。是 embodiment 一部分（hojjunekim
  摘橙子用手），保留 78 维不为 flow3 加特例；模型瞬间学会，MSE-64 只算前 64 维不受影响。
- **46 state = `observation.state[43]` ++ `projected_gravity[3]`**，min_max 到 [-1,1]；
  state43 里 **14 个退化维 min==max**（手指）→ 归一化置 0 防除零（`_minmax` 守卫）。
- prompt = `task_index`(0..7)；8 episode / 5777 帧 → **5465 个 horizon-40 chunk**。

## 🔴 头号工程坑：3120-token attention 显存墙（两评审预警，实锤）
- `nn.TransformerEncoder` 走非融合路径物化 (B·H,S,S)，S=3120 时 **batch 32 fp32 即吃满 22G OOM**。
- **修复①**：自写 `_Block` 用 `F.scaled_dot_product_attention`（不物化 S×S）替换 nn.TransformerEncoder。
- **修复②**：**必须 bf16 autocast** —— SDPA 的 mem-efficient/flash 后端在 **fp32 下不生效**，
  退回 math 路径照样物化。加 `torch.autocast("cuda", bfloat16)` 后峰值：batch16=8.0G /
  batch32=16.1G / batch64 OOM。**结论：S=3120 训练 batch16+bf16 是 4090 甜点。**
- 启动带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 脚手架（MaskBeT/ 内，已接通）
- `maskbet/data.py SonicFlow3Dataset(synthetic=False)`：直读 flow3 parquet，已上述字段映射 + 归一化。
- `maskbet/eval.py`：开环 MSE-64（argmax/sample），de-tokenize 后只算前 64 维 + per-prompt 分解。
- `maskbet/train.py`：bf16 autocast、warmup+cosine LR、原子 ckpt、`--keep`(<=0 留全部)、`--resume`、`--seed`。
- **巡查**：`scripts/eval_watcher.sh`（poll ckpt → eval → CSV，含 trainer-dead 退出）；
  本轮用 `/tmp/maskbet_eval_all.sh`（等 train PID 退出后无 GPU 争用地 eval 全部 12 ckpt → `outputs/flow3/eval_mse64.csv`）。
- 日志要 `PYTHONUNBUFFERED=1` 否则 print 缓冲看不到（ckpt 落盘是文件操作不受影响，可用 mtime 测速）。

## 当前 run（2026-06-11 进行中）
batch16 + bf16 + 6000 步（≈17.6 epoch over 5465 chunk）+ keep0 全留 + lr 2e-4 cosine。
~5.2 step/s。loss 曲线 step50→3000：2.06→1.01→0.886→0.777，bin_acc 0.31→0.675（稳降减速）。

## 🔴 step 3000 猝死 = kernel 6.17 GPF 阵发（日志无 traceback 干净消失，非代码 bug）
- 进程信号杀死，GPU 立即空闲；StarVLA 也中过同一家族（[[starvla-sonic-ab-baseline]] 末尾）。
- **应对 = watchdog 自动 resume**：`/tmp/maskbet_watchdog.sh` 循环 `train --resume` 直到
  `ckpt_006000.pt` 存在（≤30 次重试，每次崩后 sleep 8 让 GPF 风暴过）。已从 ckpt_2500 续上。
- **eval 触发改成等 final ckpt 文件存在**，不等 train PID（看门狗会重启 PID，等 PID 死会早触发）。

## ✅ 完整结果（6000 步跑满，2026-06-11）= 强正面 + 一个解码发现
**iterative argmax MSE-64 曲线**（step 500→6000）：
0.0411/0.0446/0.0385/0.0371/0.0343/0.0303/0.0286/0.0255/0.0232/0.0221/0.0194/**0.0193**
→ 收敛到 **0.0193 ≈ CE v1 argmax 0.0174**（只高 11%）。bin_acc 收敛 ~0.713。

### 🎯 解码发现：单次全-MASK 期望值 = 0.0090（MaskBeT best，打穿所有锚）
- **soft1 = 单次从全-[MASK] 前向 + E[value]=Σp·center**（条件均值=MSE 最优估计量；argmax 取众数非最优）。
- final ckpt **0.0090**：比 CE v1 best(expected 0.0125)低 1.9×、比 CE v1 argmax 0.0174 低 1.9×，
  **离 GR00T 0.0026 只剩 3.5×**（CE v1 是 6.7×）。比自己迭代 argmax 0.0193 好一倍。
- **为何单次比迭代好**：全-MASK 输入在训练分布内（cosine mask 偏高掩码率），边际校准好；
  **反例坑**：先迭代解完整格再重喂读期望 = OOD → logits 弥散 → 期望塌向零 = 退回模板 0.0361。
  正确实现 = 单次全-MASK，已写进 `model.predict(mode="expected")`。
- **闭环 caveat（brainstorm §3）**：expected 是**离格连续值**，喂 SONIC WBC 前必须 snap 回格；
  argmax/sample 构造性在格。开环 MSE 用 expected，闭环注入用 argmax 或 expected+snap。

### 结论
**25M 从零 ≈/优于 4B 冻结 VLM CE head**（argmax 打平、expected 反超）→ 实测坐实
[[starvla-sonic-ab-baseline]] brainstorm §8 codex「非-VLA baseline 拷问」：瓶颈是数据+obs 不是
骨干/语义容量。**但都是记忆口径无 held-out**（MaskBeT 从零×17ep 记忆更重）→ 下一刀必须 held-out
（训 7 窗评第 8）。锚：GR00T 0.0026 / CE v1 0.0174argmax·0.0125E / 模板 0.0367。

## 🎮 实时推理 GUI（2026-06-11，跑通）
- **脚手架**：`LeSONIC/scripts/serve_maskbet_sonic.py`（ZMQ token server，**完全仿 serve_starvla
  wire**=msgpack-numpy `{"endpoint","data"}`，复用其 `MinMax`+置换修复；prompt 字符串→task_index
  查 tasks.jsonl；server 内 min_max；回 `{"action.motion_token":(1,40,64)}`）+
  `scripts/maskbet_sonic_live_demo.sh @flow3`（一键，GPF 重试，复用 server）。Isaac 侧
  `gear_sonic_sequence.sh`/`vla_live_injector` **零改动**，只要 `GR00T_PORT=5557`。
- **env**：base env zmq 坏（PYZMQ_DRAFT_API）→ 用 `starvla_eval_qwen35`（zmq+torch2.6，纯
  torch state_dict 照载）。decode 默认 argmax（构造性在格，snap 无操作）；expected 要 server 内 snap。
- **🔴 启动坑**：后台起 server 反复 exit 144(=信号16)=kernel 6.17 GPF「import 段」阵发
  （前台 `-c` 能跑证明代码无错）→ nohup+setsid 重试循环顶过去，attempt1 即 READY；
  顺手杀旧 serve_starvla 遗留(占 10G)腾卡。wire 冒烟：ping✓ / token(1,40,64) off-grid=0 /
  prompt 敏感(fierce vs circle mean|Δ|=0.104,没塌缩)。
- **闭环目检结果（正面，优于 CE live）**：seg0 block-push-kick 全 542 步**没倒**(pelvis z~0.71)
  +在动；**tok_std 0.169-0.196 ≈ GT 0.179**——比 CE live「幅度很小」好（CE argmax 偏保守压幅，
  MaskBeT iterative argmax 保 std）。6 段 flow3(block→circle→combat→sprint→moonwalk→jogback)
  循环。RELAX 模式（关摔倒/偏离终止）= prompt 编排非抗摔声明，同 GR00T/CE scope。

## 收尾状态（2026-06-11，compact 交接）
- **demo 已停**：server + Isaac 全杀，GPU 回 477MiB 空闲。重启=notebook cell 或
  `bash scripts/maskbet_sonic_live_demo.sh @flow3`。
- **LAFAN.ipynb 已加 MaskBeT 启动 cell**（cell 11 markdown + 12 `!bash maskbet_sonic_live_demo.sh`，
  在 StarVLA cell 10 旁）。用户目检结论：与 StarVLA 同档略好。
- **待用户 commit**（都没提）：伞仓 LeSONIC = LAFAN.ipynb + `scripts/serve_maskbet_sonic.py` +
  `scripts/maskbet_sonic_live_demo.sh`（?? 未跟踪）+ doc 改 + MaskBeT gitlink；MaskBeT 子仓 =
  maskbet/{config,model,masking,data,train,eval}.py + scripts + README + doc（已被用户首次 commit，
  本会话又改了一批，未提）；`.memory/` 本文件 + 索引。**git 全由用户做，我不碰父仓提交。**
- **下一步训练计划已成文+二轮外审修订** = `MaskBeT/doc/maskbet_training_plan.html`
  （P1 held-out 双协议+P1b 闭环初筛→P2 配方/K/温度 sweep→P3 hojjunekim 混训(0.1GPU·h
  parity 前置门)→P4 混采样预训练→P5 终评；判定门+最小效果量预注册）。codex/gpt-5.5 与
  mimo-v2.5-pro 各 12 条全采纳，最狠五刀：①stride-1 滑窗击穿朴素时间切分(训 chunk 40 帧
  伸进评区,必须全帧隔离+purge gap+disjoint assert)；②模板基线是 test-aware oracle，真下界
  =proprio k-NN 检索基线；③闭环可执行口径是 argmax 0.0193 非 expected 0.0090(统计量分层)；
  ④token 空间 fps 插值造不存在中间码(正解=关节流插值+encoder 重 tokenize,先 fps-token
  quick A/B)；⑤顺序 curriculum=灾难遗忘(P4 默认混采样,flow3 上采样 5-10%)。
  另:留窗协议必须 text_embed_dim>0+三组对照归因(投影层域偏≠泛化为零)；K=3=0.06s 上下文，
  一行 config 的 K sweep ROI 可能高于 549k 帧数据工程，P2 先测。
  坑:opencode run 接 tail 会截掉按严重度排序的前几条——全文可从
  `~/.local/share/opencode/opencode.db` 的 part 表捞回。

## P1 held-out 实施（2026-06-11，开工）= 代码落地 + 首批 train-only 基线
- **data.py `make_datasets(cfg,protocol,...)`**：一次加载 episode，产出共享底层数组的 train/eval
  两视图（`_view()` 传 `_ep_*` 给新 dataclass，`__post_init__` 见非空就不重载）。
  - **protocol="temporal"**：按 chunk 的**完整帧跨度** `_chunk_frame_span`（含 K-1 history）切，
    train last≤split-1 / eval first≥split+gap，gap<0→H+K=43。`_assert_disjoint` 逐 episode
    断言 train/eval 零共享帧——**修 stride-1 滑窗泄漏**（朴素按起点切会让 train chunk 伸进 eval 区 39 帧）。
  - **protocol="window"**：留 holdout_ep 整窗，eval=该窗全部 chunk。
  - 暴露 `.starts`/`.chunk_prompts`（shuffle=False 下与 DataLoader 对齐，给 head/steady 分桶）。
- **🔴 坑：短 episode 被 purge gap 吃空**。ep4(prompt4) T=241 最短，frac0.8+gap43 → eval=0 chunk
  → **Protocol A 只覆盖 7/8 动作**（ep2 T=542 也仅 25 eval chunk）。不缩 gap（disjoint 已由
  span 保证，gap 是抗自相关 purge 余量=协议存在的理由）；ep4 由 Protocol B 自己那折覆盖，两协议互补。
- **baselines.py `compute_baselines`（全 train-only，numpy/CPU）**：global_mean / prompt_mean
  (留窗缺失 prompt→global 回退并 flag) / **proprio_knn(K帧 state 最近邻检索=真条件下界)** /
  oracle_window_mean(只标下界永不当判定线)。
  - **首批实测（基线，非模型）**：temporal frac0.8 → global 0.0353/prompt 0.0357/**knn 0.0424
    (比 global 还差!)** /oracle 0.0275；window ep3 → global 0.0551/knn 0.0739/oracle 0.0512。
    **proprio k-NN 比 global mean 还差 = 单纯 proprio 检索抓不到对的 action，模型有真条件信号要打**。
- **eval.py 重写**：head/steady 分桶(start≤2，temporal eval 全 steady→head=nan 正常)、
  snapped-expected(闭环可执行口径)、token coverage+entropy(MaskGIT mode-collapse 检测)、
  `--baselines` 接 compute_baselines、`--protocol/--holdout-ep/--frac/--gap`。train.py 同样接 split args。
- **env=base**(torch2.7+cu128+sklearn+pandas，非 starvla_eval)。脚手架 `scripts/holdout_train.sh
  <temporal|window> <frac|ep>`(watchdog resume + 全 ckpt held-out 曲线 sweep + final baselines)。
- **关键**：旧 ckpt_006000 是全量训练的，eval 它在"held-out"区=0.0085 仍是记忆口径(训过那些帧)；
  **真 held-out 必须在 train-split-only 重训**。

## 🎯 Protocol A held-out 首个真泛化数(2026-06-11) = 记忆口径 0.0090 高估约 3×
train-split-only 重训(4307 chunk) + 评 held-out 区(521 chunk)曲线 step500→6000:
0.0347/0.0314/0.0287/**0.0279(step2000 best)**/0.0301/0.0301/0.0307/0.0301/0.0310/0.0306/0.0320/0.0314
→ **best 0.0279 @ step2000，之后过拟合上翘**(6000 回 0.0314)。
- **vs train-only 基线**：胜全局均值模板 0.0353(−21%)、per-prompt 0.0357(−22%)、proprio-kNN
  0.0424(−34%)；**只打平 oracle 窗均值 0.0275 未超过** → 模型恢复了"该窗均值轨迹"但没抓多少均值外时序结构。
- **诚实结论**：真泛化 ≈0.028，记忆口径 0.0090 高估 ~3×；但确实胜所有 train-only 模板=有真条件信号。
- **过拟合在 ~2000 步** = 直接喂 P2:early-stop 2000 不是 6000。token coverage 训练中 0.61→0.76(无塌缩)。
- 产物 `outputs/holdout_temporal_f08/{heldout_curve.csv,baselines.txt}`；写进 design §5.2。

## Protocol B 文本embedding基建(2026-06-11) + 8折窗口sweep(进行中)
- **text_embed.py**:sentence-transformers all-MiniLM-L6-v2 → 8 prompt 的 384维冻结embedding,
  缓存 `meta/prompt_text_embed_minilm.npy`(cosine off-diag 0.045-0.647 良好分离)。
- **data.py**:cfg.text_embed_dim>0 时 __getitem__ 发 embedding 行(非 int id);_view 共享 _prompt_embed。
- **train.py/eval.py 加 `--text-embed-dim`/`--state-history`(P2 K sweep 用)/eval `--zero-prompt`(3组对照)**。
- **holdout_train.sh**:window 协议默认 TEXT_DIM=384(留窗 prompt 未见→须语义向量非死查表行),
  temporal 默认 0(prompt 见过查表够)。`holdout_window_all.sh` 8 折 ep0-7 各 3000 步(A 在 2000 过拟合)。
- 跑在 base env(torch2.7+cu128+sentence_transformers 5.1.2)。CPU 冒烟过:prompt(4,384)/25.5M/forward OK。

## 🎯 Protocol B 留窗 8 折结果(2026-06-11) = 跨母题泛化真但强依赖母题
median best **0.0396** · median vs 全局 **−11%** · **7/8 折胜全局** · worst fold 0.0638(fierce swings)。
判定门(median≤−10% 且 worst-fold 不超基线+5%)**通过**。逐折(best/vs全局):
- ep5 绕圈跑 **0.0076/−73%**(连 proprio-kNN 0.0111 都超!) · ep6 jog/run 0.0110/−18% · ep7 0.0250/−25%
  → **移动类 spectacular transfer**(周期步态从其它移动窗迁移)
- ep4 fierce swings 0.0638/**+0.5%=打平模板=零迁移**(训练无近邻,孤立母题) ← 外审 codex#2 警告的"孤立动作
  被均值掩盖",逐折报告揪出
- ep0-3(moonwalk/spin/block/combat) modest −2~−14%
- **"段落起点最难"伪命题**:combat ep3 head 0.0160 vs steady 0.0535(ratio 0.30)、moonwalk 0.69-0.96
  → 起点共享 stance/setup 反而更易,steady 发散才难,phase-clock 非普适刚需
- **直接喂 P3**:移动迁移强+孤立母题零迁移 → manipulation 语料(hojjunekim)帮不到 combat/swing 窗
  (codex#7 实证),同类移动/格斗数据才会。逐折 gating 被验证为必需。
- 产物 `outputs/holdout_window_summary.csv` + `scripts/aggregate_holdout.py`(nanmedian);写进 design §5.2。
- 坑:ep3 baselines eval 被 GPF 阵发杀(只该折),手动重跑 ckpt_001000 补回(−3.3% vs 全局)。

## 🔬 P1 评判补测(2026-06-11) = 4个纯eval零重训,两个预设被推翻
1. **解码层级在 held-out 上反转**:temporal held-out argmax **0.0332**(只胜模板6%!记忆口径
   「argmax 0.0193≈CE 0.0174」是被记忆抬的) vs snapped-expected **0.0282≈expected 0.0279**。
   GT 完美在格→**snap 对 |err|<1/32 是免费纠错**。**闭环可执行口径修订为 expected+snap,
   优于 argmax**;serve_maskbet_sonic.py 默认 argmax 待切(P1b 验证)。ep5 argmax 0.0212 vs
   expected 0.0076 同向(2.8×)。
2. **zero-prompt 三组对照补齐**:ep4 zero-prompt 0.0639≈正常 0.0638=文本零贡献→失败归因=
   **母题孤立**非条件失效/投影域偏;ep5 zero-prompt expected 0.0125 vs 正常 0.0076 但
   **snapped 口径两者同为 0.0080**——可执行分辨率上跨母题迁移**全部由 proprio history 携带**,
   文本通路近乎死重→K-sweep 优先级再↑,文本塔升级=不做。
3. **P1 自身统计瑕疵(P2 起修)**:①每折 best=评测折 min-over-ckpt=轻度 peeking,median −11%
   对 −10% 门是边缘通过;②全单 seed 无误差棒;③「7/8胜全局」有水分——ep1/3/4 在±5%内≈模板,
   诚实摘要=**双峰:4/8 真迁移(移动类),4/8≈模板**。
4. **计划修订已落**(plan+design §5.2):P2 加选步纪律(val-slice/固定步2000,报告折只评一次)
   +K-sweep 最高优先+phase-clock 降级;P3 门②按母题类分桶(看孤立折 ep1/3/4 动不动);
   P1b 解码主臂=expected+snap;风险表更新;**P1b 是 P1 唯一未完成项**。

## ✅ P1 评判后续验证(2026-06-11) = 两个洞实测堵上,headline 站得住
**(A) 解码反转 8/8 折普适**(不是 temporal/ep5 孤例):全部留窗折 snapped-expected 胜 argmax:
ep0 0.0550→0.0444 / ep1 0.0574→0.0508 / ep2 0.0402→0.0353 / ep3 0.0616→0.0536 /
ep4 0.0750→0.0641 / ep5 0.0212→0.0080(2.65×) / ep6 0.0130→0.0113 / ep7 0.0301→0.0253。
幅度 13–24%(ep5 达 2.65×)。**snap 相对 raw expected ≤+0.0004**(模型期望值本就贴格)→
**闭环在格约束几乎免费**,mimo #4「expected 只是格间插值红利、闭环无收益」担忧彻底排除。
脚手架 `/tmp/decode_verify.sh`。
**(B) 协议 A 3-seed 复现**(固定 step 2000,无 peeking):seed0/1/2 = 0.0279/0.0289/0.0288 =
**0.0285 ± 0.0006**。headline 可复现 σ 紧,单 seed 批评消解。seed0 的 0.0279 是 min-over-ckpt
略乐观;seed 平均后 vs 全局模板 ≈−19%、处 oracle 水平(略高 3.6% 在噪声内)。
脚手架 `/tmp/seed_repro.sh`,产物 `outputs/seed_temporal_s{1,2}/ckpt_002000.pt`。
**结论**:P1 三个统计瑕疵中,解码反转(已 8/8 确证)+单 seed(协议 A 已补)两个堵上;
留窗折 seed + min-over-ckpt peeking 仍欠(P2 选步纪律覆盖)。**下一步仍是 P1b→P2(K-sweep)**。

## P1 收尾状态(2026-06-11)
- **P1 两协议全做完**=Protocol A(时间切分,真泛化 0.0279,记忆 0.0090 高估3×)+Protocol B(留窗,跨母题
  median −11% 7/8胜,强依赖母题)。两份结果写进 design §5.2,训练计划 P1 标完成。
- **下一步候选**:P1b RELAX-off 闭环初筛(需 Isaac,GPU 现已空)/P2 early-stop@2000+K sweep(A/B 都指向
  过拟合早,P2 该 early-stop)/P3 数据扩展(B 结果已预示 manipulation 帮不到格斗窗)。
- 全部 MaskBeT 子仓改动+`.memory` 待用户 commit。git 全由用户做。

## ✅ P2 K-sweep 完成(2026-06-11) = history 单调有用,一行 config 换 −6.8%
协议 A,**干净消融**:gap=57−K 使 4 个 K 的 eval 目标集逐 chunk 完全相同(同 4307 train/同 444
eval,只 history 长度变;否则 first=s−(K−1)≥split+gap 让大 K eval 集变小 = 污染比较)。固定
step2000 无 peeking,expected 解码:
- K=1(0.02s) 0.0304 / **K=3(0.06s) 0.0278 基线** / K=8(0.16s) 0.0271 / **K=16(0.32s) 0.0259 = −6.8%**
- **4 点单调下降(非噪声)**实证 zero-prompt 发现:proprio history 是唯一在干活的条件通路,越多越好。
  K=16 step1000 即 0.0281、2000 未平台→还能深。**但 −6.8% 未过 −10% 效果量门**。
- 脚手架 `MaskBeT/scripts/p2_ksweep.sh`+`p2_ksweep_ext.sh`,产物 `outputs/ksweep_summary.csv`。
  state_history 经 train/eval `--state-history` 传,K=16 history 下溢由 `max(0,start−k)` 钳帧0 安全。

### 🔬 P2 K-sweep 续(K=24/32+3seed,2026-06-11)=甜点 K=16 但 −5% 在显著性边缘
- **完整 K 曲线(seed0)呈 U 形**:K1=0.0304/K3=0.0278/K8=0.0271/**K16=0.0259(最低)**/K24=0.0266/
  K32=0.0274 → **history 在 K=16(0.32s)饱和并反转**,更多历史小数据上加噪。
- **3-seed 复核(白捡:K=3 复用 Protocol A 的 seed_temporal_s1/s2 在 gap=54 重评)**:
  K=3=0.0278/0.0285/0.0284 → **0.0282±0.0004**;K=16=0.0259/0.0278/0.0268 → **0.0268±0.0010**。
  **K=16 vs K=3 = −5.0%,Welch t=2.37/p≈0.10 = suggestive 但未到 p<0.05、未过 −10% 门**。
  两组三 seed 范围仅在 0.0278 端点相接(近不交叠,方向真幅度小)。
- **🔑 单 seed 的 −6.8% 是 seed0(0.0259)运气低端,3-seed 拉回 −5%**——直接印证 P2 自己的 3-seed
  纪律,单 seed sweep 高估约 2pp。K=16 可作新默认但只 modest,真杠杆仍是数据(P3)。

## 🔧 P1b 闭环 survival harness(2026-06-11) = 复用 PhysValidScreen + 组合回调坑
- **P1b 设计**:vla_live(live MaskBeT 注入)+phys_valid(测真摔 root_z<0.6/tilt>40° 记 fall_step)
  +4 个 deviation 终止设 null(留 motion_time_out,让机器人播完整窗,phys 测真摔非严格包络误触);
  逐段跑 decode=expected+snap;survival↔各段开环 MSE 关联=校准目标函数。GT 上界=现成
  `gear_sonic_phys_screen.sh`(纯回放)。脚手架 `LeSONIC/scripts/sonic_p1b_survival.sh`。
- **🔴 头号坑:eval loop `all(cb.eval_step() for cb in cbs)` 短路**。vla_live.eval_step 永远
  `return False`("never exit")→all() 在它处短路→**phys_valid.eval_step 永不被调,没 init 没 dump**
  (空 JSON)。两回调放一起根本跑不了。**修=组合回调** `scripts/vla_live_phys_probe.py`
  (VlaLivePhysProbe 内部先 vla.eval_step 注入、再 phys.eval_step 测量、返回 phys 退出信号),
  单回调挂 `eval_callbacks=[probe]`+PYTHONPATH=scripts(WBC submodule 保持干净)。
- **GPF 坑**:server+Isaac 两段都做 torch/CUDA import→都阵发 exit144(信号16)。外层 setsid+144
  重试循环顶过去(`/tmp/p1b_smoke_retry.sh`),attempt 内 server 自带 6 次重试。

## ✅ P1b 闭环 survival 结果(2026-06-11) = 闭环可用,开环MSE预测裕度不预测survival
全 8 段 live MaskBeT(ckpt_006000,expected+snap)+phys_valid:**0/8 摔,全站住**(各段播完整窗
749/999 步)。
- **① 闭环可用、不是瓶颈,可放心烧 P3**——WBC 底座吸收 token 级误差。
- **② binary survival 在 flow3 开环 MSE 区间(0.0076–0.0638)内饱和**(全站着),但**连续稳定裕度**
  与开环 MSE 中度相关:Spearman(开环MSE,max_tilt)=+0.52、(,min_root_z)=−0.52。fierce(开环最差
  0.0638)tilt 34.5° 最逼近 40° 摔线、combat 25.9°,干净移动折(低MSE)只 12–14°。
- **校准结论**:闭环鲁棒性该优化的目标量是**稳定裕度**,降开环 MSE(如 K-sweep)推裕度离包络,
  但联系松(ρ0.52,n=8 不显著 p≈0.18,单 round 确定性解码无误差棒)。
- caveats:记忆口径 ckpt(对"可用性"是对口径);RELAX/phys-valid 测真摔非严格包络误触;
  decode A/B 闭环对比留 P5(开环已实证 expected+snap 胜,闭环饱和两者都站)。
- 产物 `outputs/p1b_survival_summary.csv`。**P1+P1b+P2 K-sweep 全做完**,下一步 P2 续(K∈{24,32}+
  K=16×3seed+prev-token)或 P3 数据扩展(hojjunekim parity gate 前置)。

## 🔴 P3 parity gate(2026-06-11) = token空间过但覆盖不过,hojjunekim 救不了硬折,redirect
CPU 统计 parity(~0.01 GPU·h,只下 ~5ep/变体样本,数据在 HF cache 只有 info.json 要单独下 data/):
- **① token 空间 parity 过**:hojjunekim off-grid 0/195520(同 1/16 FSQ 格)、64 维、state/hand
  (全 0)/gravity 约定全匹配 → 混训技术上有效无 tokenizer 失配,encoder 复算可省(格证据决定性)。
- **② 覆盖是杀手**:**所有可用变体(mani/locomani/teleop)困在 |k|≤8 中央区,极端 bins |k|>7 占比
  ≤0.0005**;flow3 硬折住极端区:fierce 0.080/combat 0.055(P1-B 零迁移那几个),移动折 jogback
  0.002/circle 0.015(P1-B 强迁移)。**三证据收敛**(P1-B 迁移×token覆盖需求×hojjunekim覆盖)=
  hojjunekim 只帮已迁移好的中央折,对极端-bin 动态折零覆盖零帮助。
- **③ 多样性注水**:prompt 多样性=1("place orange on plate");**7 变体里 3 个 _psi0 的
  motion_token 全是 0**(用 psi0_18 不同动作空间不在 token 空间)→"549k 95×"虚数,可用=4变体/1prompt。
- fps=30 vs flow3 50(混训要重采样)。
- **判定**:parity 技术过语义不过→**不烧 20-40 GPU·h 混 hojjunekim 修动态折**;redirect=同域**动态**
  语料(更多 LAFAN 窗/AMASS-G1 动态 clip,访问极端 bins)才对症。产物 `outputs/p3_hojjunekim_parity.csv`。
- 脚手架坑:HF cache 里 hojjunekim 只缓存了 info.json,`hf download --include "meta/*" "data/chunk-000/episode_00000[0-4].parquet"` 下样本(别下 mp4 videos/)。

## brainstorm 经验对 MaskBeT 的适用（读 sonic_starvla_swap_brainstorm.html §3-10）
- **§8 非-VLA baseline 拷问**：MaskBeT 字面就是它，结果已答（0.0193≈0.0174，0.0090 反超）。
- **§4 flaw#1 无 held-out=记忆**：对 MaskBeT 更 severe（从零更易记），必补 held-out。
- **§3 网格点之间=latent garbage**：expected 离格，闭环前 snap（CE v1 用 SONIC_CE_SNAP=1）。
- **§8 chunk 边界/相位重置**：现 MSE-64 全 chunk 均匀平均，盲于 prompt 切换/段落起点头 N 步；
  MaskBeT 有 K=3 history 无 phase clock，段落起点最易崩，eval 待加该分解。
- **§5 归一化陷阱**：action=identity（直接 k/16 量化不碰 q99/minmax），state=min_max → 已避开。
- **§8 WBC 可行性在上游**：开环 MSE 好≠WBC 跟得住，需 RELAX-off 严格终止下闭环筛查（同 StarVLA caveat）。

## 🔍 P1b+P2 评判修订(2026-06-11) = 上面两节各砍一刀,以本节为准
- **P2「U 形甜点 K=16」不成立**:K=24 单 seed 0.0266 其实**低于** K=16 三 seed 均值 0.0268
  (−0.25σ),K8=+0.28σ/K32=+0.60σ → **K∈[8,32] 全在 K=16 seed 噪声 ~1σ 内 = 平台非 U 形**。
  单 seed 不仅高估增益 ~2pp 还虚构了曲线形状(3-seed 纪律二次印证)。K=16 仍取默认(平台内任选,
  成本零)。csv note 已改 plateau。
- **P1b Q2(目标函数校准)实际没回答**:survival 饱和(0 摔=0 信号);Spearman(MSE,max_tilt)=0.52
  p≈0.18 不显著且**混杂未控——max_tilt 同时是动作本身动力学的函数**(moonwalk/spinclap MSE 排
  5/6 名 tilt 只排 2/3 名,dance 慢动作天然小 tilt;相关由 run-vs-fight 类型差承载)。
  「该优化稳定裕度」旧结论**撤回**。去混杂需 GT-token replay 逐段 tilt 基线(live−GT Δtilt,
  ~0.5 GPU·h,gear_sonic_phys_screen.sh,脚本头注释设计的 pair 另一半没跑);survival 出信号需
  压力化(温度↑/corruption on)——两件归 P5。**「降开环 MSE 改善闭环」目前是未验证假设,P3 判定
  门不得引用闭环指标**。
- 其他 caveat 补记:P1b 实跑单 round(确定性配置);每段前 80 步录制 token bootstrap,live 覆盖
  ~89% 窗。
- **P2 收官判定**:最被看好的 K 刀只 −5%/p≈0.11 → flow3-only 配方打磨(dropout/early-stop/
  prev-token)ROI 负(P3 换数据后全要重扫);只留零重训温度 sweep 可选,其余挂起到 P3 后。
  两 doc(plan P1b/P2/P5 框+design §5.3)已同步修订。

## 🔴 GT-token phys 基线(2026-06-12) = 推翻 P1b 乐观读数,Q2 已答
补上 P1b 缺的配对另一半:`gear_sonic_phys_screen.sh`+`terrain_type=plane`(对齐 P1b live)回放 GT 录制
token。**先验:trimesh 默认地形 vs plane 几乎同结果(block/fierce 都摔)→地形不是混杂**。
- **① WBC 底座本身在 block/fierce 上站不住**:GT 录制 token 在 plane 上照摔(block 41.1°@542、
  fierce 40.2°@199)。**「WBC 吸收 token 误差、闭环不是瓶颈」撤回**——这两段可达保真上限就是「摔」,
  模型修不了,是底座天花板。
- **② live 全 8 段 tilt 都低于 GT(Δtilt −1.1~−19.6°)**:「0/8 不摔」是**平滑/向中央 bin 塌缩**的
  签名不是跟得准;block 靠比 GT 少做 19.6° 幅度活下来(没在做 push-kick-shove)。欠做最大处=最难的
  fight/dance 母题(block−19.6/combat−12.1/spinclap−10.5),最小=干净移动折(circle−1.1/jogback−2.1)。
- **③ Q2 干净答案**:Spearman(开环MSE, |Δtilt 欠做量|)=**0.62**(p≈0.10)。开环 MSE 预测**闭环保真缺口**
  (执行幅度比 GT 短多少)非 survival(平滑致饱和无信息)。原 max_tilt ρ0.52 被动作动力学混杂,减 GT 参考
  后开环 MSE 获合法闭环含义。注:Spearman(MSE,GT_tilt) 也=0.62(硬母题=高 GT tilt),|Δtilt| 继承两者。
- **P3 含义**:第三条独立证据(P1-B 零迁移×P3 token 覆盖×本节闭环欠做)收敛「硬母题缺极端-bin 动态」;
  **但 block/fierce 连 GT 都摔=WBC 不可行,数据 redirect 瞄准 WBC 稳定包络内的可达动态母题**(combat
  GT38° 站得住、跑/舞折富余),不是救这两段。压力化逼真摔(温度↑/corruption/argmax 对照)出 survival
  信号归 P5。产物 `outputs/p1b_gt_baseline_plane.csv`。两 doc(design §5.3.1 新增表+plan P1b 框转 warn)已改。

## ✅ P3 encoder parity gate = PASS（2026-06-12，决定性）+ redirect 语料就位
hojjunekim 被 parity gate 否后 redirect 到同域 LAFAN full clips。两件做实:
- **encoder parity gate 全过**:用生成 flow3 的同一管线 `gear_sonic_record_flow3.sh`(VlaTokenRecorder
  tap WBC 跟踪参考动作时 actor 发的 64 维 token)在 circle/combat/fierce 重录,与 flow3 现有 parquet
  逐列比:**全部 100.000% 逐维比特级精确,max abs diff 0.0000,长度逐段相等**(circle749/combat749/
  fierce241)。覆盖中央到极端 bin(fierce |k|>7=0.0801、combat 0.0545,与 P3 hojjunekim parity 表的
  0.080/0.055 逐字对上=交叉验证)。**token 来自参考动作未来帧 command_multi_future_*、不含仿真噪声→
  比特级而非 >99%**。含义:tokenize 管线确定性可复现、WBC ckpt=造 flow3 版本、源 pkl 对→可信 tokenize
  新 LAFAN。脚手架 `scripts/sonic_token_parity.py`(可复用 QA)+产物 `outputs/p3_encoder_parity.csv`。
- **语料源就位(诚实规模)**:`fight_full_robot.pkl`(7345帧)+`run_full_robot`(7133)+`dance_full`(3943)=
  **~18.4k 帧 ≈ flow3 3.2×**(非 95×,但同域/访问极端 bin/文本可细分)。**jumps_full 排除**——
  sonic_lafan_expansion.html 实测物理做不出 + 本轮 GT 基线坐实 WBC 对 block/fierce 连 GT 都摔。
- **下一步(task #6)**:full clips 切窗 → phys_valid(RELAX-off)WBC 能力筛查掉跟不住/逼近摔倒包络的窗
  (GT-tilt 余量门避开 block/fierce 类)→ 存活窗经 recorder tokenize → data.py 混训 → flow3 留窗评。
- 关键参照:`doc/sonic_lafan_expansion.html` 已是这事的前期 brainstorm(WBC 能力筛查=第一闸门;
  fight/run/dance 过、jumps 否;fight open-loop MSE 0.0021 已进 VLA)。

## ✅ P3 WBC 能力筛查完成+复审修正(2026-06-12)
对 143 段(fight 80+run 63)跑 phys_valid(RELAX-off,默认地形=录制条件,24 env 批)。初版 107 窗/2.60×
**复审揪出两个真 bug,全用现有 CSV 修正、零额外 GPU**:
- **bug-1 下蹲误判摔**:root_z<0.6 死阈值把 14 段下蹲母题判成摔——robot min_z 与参考 min_z 差 ≤6cm
  (seg005 0.495 vs 0.495),是正确跟踪下蹲;真摔躺地 z≈0.25。修正判据=**tilt≤33° ∧ min_root_z≥0.35**,
  回收 14 段→121 窗。fall_step=1 的"开局即摔"全是该 bug(起手蹲姿)。**教训:fall 判据对下蹲母题
  必须用真摔地板(0.35)或 ref-relative,不能用站姿阈值 0.6。**
- **bug-2 flow3 泄漏未扣**:kept 窗与 flow3 评测段同源重叠(用数组内容在 full clip 定位 flow3 偏移:
  fight 重叠 40%/run 67% of flow3 src)。encoder 确定性→重叠帧=逐比特相同 token,混训后评 flow3
  held-out=背题。修正=训练语料逐窗扣 flow3 源区间。**教训:同源切窗扩数据必先做 train/eval 区间对账。**
**修正后净语料 = 121 窗→98 净片段→合并相邻=15 条连续录制 run(fight 10/run 5,2.5-50.2s)
≈ 13.5k 帧@50Hz ≈ 2.34× flow3**(毛 2.95×,泄漏扣 0.6×)。录 15 次而非 121 次,顺带解决短窗被
~80 步 bootstrap 吃光的问题。产物:`outputs/p3_wbc_capability_keep.csv`(v2,净片段+标注) +
`outputs/p3_recording_runs.csv`(15 run)。**dance 暂未切段**,先 fight+run。
**录制前置 gate**:encoder 输入 `motion_anchor_ori_b`(base 系)可能依赖机器人状态→合并 run 切片
是否=单窗录制需先抽查 1 窗 parity;**实跑结论=不是 blocker**:每 run 作为独立段录制,(token,state,proprio)
三元组录制时天然自洽,混训学的是这个自洽映射,锚定约定训练内一致即可;flow3 泄漏已由帧区间整段排除(与锚定无关)。

## 🎯 P3 混训 = 强阳性(2026-06-12):2.34× 同域数据 → held-out MSE −18.4% + 消除过拟合
pipeline 全跑通:build_p3_runs_pkl(15 run 切自 full clip,flow3 帧整段排除)→gear_sonic_record_p3.sh
(15 次独立 Isaac 录制,GPF-retry+断点续,token 精确在格)→sonic_vla_npz_to_lerobot(data-collection venv,
**PYTHONPATH=. 才能 import gear_sonic**)→data.py `make_mix_datasets`(flow3-train temporal + 全 P3,
P3 用 flow3 stats 归一 + prompt_offset=8,n_prompts 8→10)→`scripts/heldout_watcher.sh` 每 500 步评 flow3-eval。
**train=flow3-train(frac0.8)+P3(17250 chunk,3.74×)、eval=同一 flow3-eval(521 chunk 同口径)**。
**结果(snapped-expected MSE-64)**:baseline 0.0282@2000(后过拟合升 0.0318@6000) → **mix 0.0230@12000
(−18.4%,收敛~0.023 平台无过拟合)**,coverage 0.61→0.64。
**三硬结论**:(1) 越过 flow3 oracle_window 抄近邻地板 0.0275 → 学到动态先验非抄窗;(2) 收益集中增数据最相关处——
per-prompt **run 家族(5/6/7)暴跌 0.006-0.013**(P3 加 run 最多)、fight/dance(0.027-0.032)改善小;
(3) 直接证实 P2 数据瓶颈结论——瓶颈是数据非 head。曲线末端仍微降,再训可压~0.022 但 ROI 递减(P3 收官)。
产物:`outputs/p3_mix/heldout_curve.csv`(24点)+`p3_mix_result.csv`+`p3_mix_perprompt.csv`,
best `outputs/p3_mix/ckpt_012000.pt`(余裁剪省 6.7G,曲线+config 可重建)。
**成本实测**:全程录制~1GPU·h + 混训 12000 步~15min(70ms/step),远低于先前保守报的 20-40 GPU·h。
**坑**:launcher 没传 TRAIN_PID 给 watcher→watcher 不自退要手 pkill;heldout_curve 路径是 LeSONIC/MaskBeT/outputs
(repo root 下写 MaskBeT/... 会看错空)。
**per-prompt(snapped MSE-64,mix best vs baseline,`outputs/p3_mix_perprompt.csv`)按学得好排序**:
run 家族包揽前三 = circle 0.0057(−42%)/jog-back 0.0084(−34%)/sprint 0.0130(−42%)→ P3 加 run 最多,直接+backbone 双收益;
block 0.0272(−5%)/combat 0.0306(−15%)= fight 最难(贴 WBC 翻倒包络,保真天花板);
**moonwalk 0.0316(−18%)/spin-clap 0.0317(−8%)= dance 零新数据也涨 → backbone transfer 实锤**;
fierce(prompt4)无 eval(241帧太短 temporal 切不出)。

## 🎬 GUI 闭环 sanity(2026-06-12,已关)
`maskbet_sonic_live_demo.sh @flow3` 用 mix ckpt 跑通闭环(`MASKBET_CKPT=...ckpt_012000.pt SONIC_MASKBET_DECODE=expected`)。
**serve 补丁(新待 commit)**:`scripts/serve_maskbet_sonic.py` 原默认 `MaskBeTConfig()`(n_prompts=8)→ mix ckpt 是 10
必 load_state_dict shape 崩;已加**从 `text_in.weight` shape 推断 n_prompts 重建模型**(通用,任何非8-prompt ckpt 都需)。
闭环观察:block 段(GT-baseline 里 step542 摔的硬段)mix 驱动到 step250+ 仍 root_z~0.70 站着、tok_std~0.15 无塌缩。
启停:launcher 留 server(:5557)+viewer;关=`pkill -9 -f "eval_agent_trl|gear_sonic_sequence|serve_maskbet_sonic"`(GPF-prone 用 -9)。

**P3 收官,待办**:dance 切段补语料(再加~3943帧顶到~3×);**P5 正式闭环 eval**(非 GUI sanity,要 phys_valid 量化 mix 闭环少摔/保真，A/B vs baseline ckpt);
脚手架 `/tmp/wbc_capability_screen.sh`+`/tmp/p3_*.sh` 待落 LeSONIC/scripts。

## 🔍 P2/P3 三模型评审(2026-06-12,Fable5+codex/gpt-5.5+mimo-v2.5-pro)= 一致裁决「打折阳性」
评审产物 `MaskBeT/outputs/reviews/p2p3_review_{brief,codex,mimo}.md`。三方收敛+各自独家:
- **🔴 共识1 训练预算/LR 混杂(codex 抓 LR 数学,mimo 抓分解)**:cosine schedule 依赖 total
  (baseline total=6000 vs mix=12000,同 step LR 不同);mix@2000≈0.0307 比 baseline@2000 0.0282
  **更差**——−18.4% = 数据质量效应+「能训更久不过拟合」效应之和,不能写成「同超参 data-only」。
  修=flow3-only@12000 同 schedule 对照 + 双口径报告(同步数 vs 各自best)。
- **🔴 共识2 mix 单 seed**:P2 实测单 seed 高估~2pp;fight −5%/dance −8~−18% 落噪声边缘。修=3-seed。
- **🟡 共识3 P3↔flow3-eval 无机器断言**(两外审都标结构风险);**Fable 实测坐实具体实例**:
  4/6 个 flow3 eval 区与 P3 run 起点**距离 0-6 帧(零 purge gap,内部协议是 43 步)**——
  block/fierce/jog/sprint 四处。但 per-fold 模式证非主驱动(最干净 circle −42% 最大,污染 block −5% 最小)。
  修=污染 run 掐头~26帧(30fps)重建语料 + make_mix_datasets 加带 purge 距离的 _assert_mix_disjoint。
- **🟡 共识4「backbone transfer」过强**:dance 涨有正则化/更长训练/共享步态原语三个更简单解释,未隔离。
- **🟡 共识5「越过 oracle=动态先验」过强**:oracle_window 是诊断量非理论地板。
- **Fable 独家(外审没发现)**:P3 语料 |k|>7=0.017(比 hojjunekim ≤0.0005 高 34×=redirect 部分兑现,
  但只有 combat 0.055/fierce 0.080 的 1/3)——**tilt≤33 筛查把 fight 需要的极端动态筛掉一部分**,
  与 fight 改善最小自洽;eval 权重核实诚实(dance 44%/run 38%,聚合非 run 撑的);
  per-prompt 加权聚合 0.0227/0.0280 ≈ 报告值(对账过)。
- **codex 独家**:p3_mix_result.csv 的 train_chunks 4612 应为 4307(flow3 temporal train),要修;
  P3 prompt offset 到 8/9 → 收益全走共享 trunk,文本条件语义未被验证。
- **mimo 保守估计**:控步数+3seed 后真实改善 −10~−15%(vs headline −18.4%)。
- **三方一致下一步排序**:①方法学补洞包(净化语料×3seed+12k对照+断言,~1 GPU·h)→②P5 闭环 A/B
  (~1-2 GPU·h)→③dance 补语料→④K=16+mix 复合(~0.5 GPU·h,搭车)。
- 坑:opencode 读 /tmp 被 external_directory 自动拒绝 → brief 要放 repo 内。
**待 vc commit**:LeSONIC `scripts/{build_p3_runs_pkl,gear_sonic_record_p3.sh,serve_maskbet_sonic.py(改)}`;
MaskBeT 子仓 `maskbet/{data,eval,train}.py`+`doc/*.html`+`scripts/heldout_watcher.sh`+`outputs/p3_*.csv`+`outputs/p3_mix/{heldout_curve.csv,ckpt_012000.pt}`;伞仓本文件。

## ✅ P3 方法学补洞矩阵(2026-06-12,Opus 实施)= 增益存活并扩大,机制被改写
跑通三方要的补洞包(~1.5 GPU·h 本地 4090):①净化语料 ②3seed×2臂@12k 同 schedule ③bit-identical 断言。
脚手架:`scripts/p3_purge_flow3_adjacency.py`(掐头净化)+`MaskBeT/scripts/{p3_matrix.sh,sweep_eval.py,p3_matrix_report.py}`;
产物 `outputs/{p3_purge_trims.csv,p3_matrix_result.csv,p3_matrix_perprompt.csv,p3_matrix/*/curve.csv}`;
clean 数据集 `datasets/sonic_vla_lerobot_p3_clean`(13362帧,4 run 掐 34/44/44/44 步)。
- **🔑 机制改写(最重要)**:原 headline「0.0282→0.0230 −18.4% data-only」措辞错。**@step2000 等预算下
  mix 反而 +4.8% 更差**(0.0314±0.0013 vs flow3 0.0299±0.0005)——3.74×数据小预算没收敛完。
  真增益来自**数据消除过拟合墙**:flow3-only held-out 是 **U 形**(谷底 0.0272 @ step~4500 后翻上去震荡到
  0.032,coverage 爬 0.73=记忆更多 bin 但泛化变差=教科书过拟合);**mix 单调降到 0.0211 平台无翻转**。
  两臂各自 best:**flow3 0.0272±0.0006 vs mix 0.0211±0.0004 = −22.3%**(3-seed σ 极紧)。
  **增益不是"每步拟合更好",是"数据搬走了把 flow3-only 卡在 0.027 的过拟合墙"。**
- **✅ 增益存活且扩大**:−22.3%(净化+3seed+同schedule)> 原 −18.4%(脏+单seed+混schedule)。
  mimo 保守预测 −10~−15% **过悲观**;codex/mimo 共同要的"双口径分解"做了,结论=增益真+机制澄清。
- **✅ 泄漏=软邻接非硬泄漏(实锤)**:bit-identical 断言在**脏和净数据上都 PASS**→源区间 disjoint 从无
  答案拷贝;净化只去 protocol-purge 内的近邻自相关(43步@50Hz=26帧@30fps);clean 结果≈略优于脏→
  **泄漏从不是 driver,per-fold 证据(最干净 circle 增益最大)被坐实**。断言留作永久守卫(`data.py
  _assert_mix_disjoint`)。另:flow3_train 真值 4307 chunk(原 p3_mix result.csv 笔误 4612,codex 揪出)。
- **per-prompt 带方差(各臂 @best 总效应,3-seed mean±σ,`p3_matrix_perprompt.csv`)**:
  run 全真大跌——circle(p5)0.0052/−39%·sprint(p7)0.0091/−59%·jog(p6)0.0093/−17%;
  **fight combat(p3)0.0264±0.0009 vs 0.0345±0.0010 = −23.3% 显著为真**(单seed「fight=噪声」担忧被推翻,
  combat 加数据真有用);**fight block(p2)−1.7% σ 重叠不显著**(WBC 天花板母题,GT 自己 step542 摔);
  **dance moonwalk(p0)0.0301±0.0013 vs 0.0404±0.0014 = −25.5% 显著为真**(零新 dance 数据!)、spin(p1)
  −5.8% 弱(~2σ)。dance 真增益=**数据规模的正则化**(共享 trunk 过拟合减少→无新数据母题也涨),
  **不是"语义 backbone transfer"**(mimo 的简单解释胜出)。注:per-prompt 是各臂 best 步口径,含 train-longer 成分。
- **三硬结论改判**:(1)越过 oracle 0.0275→改述「胜 train-only 窗均值诊断量」非"物理动态先验";
  (2)"集中在新数据最相关处"大体成立(run 最大)但 dance 零数据显著涨揭示全局正则化分量,"backbone
  transfer"降级为"规模正则化";(3)"瓶颈是数据非 head"**强化**——数据搬走过拟合墙,机制现在显式。
- **下一步**:②P5 闭环 A/B(mix vs flow3 best ckpt,phys_valid 量化 Δtilt/survival,验开环 −22% 是否转闭环)
  →③dance 切段补语料(验"规模正则化 vs 母题数据")→④K=16+mix 复合。
  best ckpt:`outputs/p3_matrix/{mix_s2(0.0207),flow3_s2(0.0266)}/`。
- **新待 vc commit**:LeSONIC `scripts/p3_purge_flow3_adjacency.py`;MaskBeT `maskbet/data.py`(_assert_mix_disjoint)
  +`scripts/{p3_matrix.sh,sweep_eval.py,p3_matrix_report.py}`+`outputs/{p3_purge_trims,p3_matrix_result,p3_matrix_perprompt}.csv`
  +`outputs/p3_matrix/*/{curve,perprompt}.csv`+`doc/*.html`(待改)+`outputs/reviews/*.md`;clean 数据集 gitignored。

## 🔴 P5 闭环 A/B(2026-06-12)= 开环 −22% 不转化为闭环增益(重要负面,vc 目检坐实)
mix_s2(best,开环0.0207)vs flow3_s2(best,0.0266)逐段闭环 phys_valid(RELAX-off/plane,与 §5.3.1 GT 基线同设置),
按 **|Δtilt vs GT|=欠做量**(§5.3.1 定的闭环保真口径,survival 在 RELAX 下饱和无信号)。
脚手架 `scripts/sonic_p5_closed_loop_ab.sh`(复用 P1b harness 跑两臂+对 `p1b_gt_baseline_plane.csv` join);产物 `outputs/p5_closed_loop_ab.csv`。
- **🔴 闭环是 WASH**:mean |Δtilt| **mix 9.78 vs flow3 10.04**(差 0.26°),closer-to-GT 计数 mix2/flow32/平4。
  **开环 −22.3% 在闭环保真上几乎没兑现**——两个 held-out 模型闭环行为基本一样。
- **🔴 fight/dance 普遍欠做约一半(两模型都)**:block gt41.1°→mix19/flow317、combat gt38°→mix17/flow319、
  moonwalk gt18.9→mix11/flow39、spinclap gt22.7→两者12.2。**= vc 目检「fight/dance 不好」被量化坐实**:
  不是没学会而是**高幅 token 向中央 bin 塌缩(平滑)**,做出来只有参考幅度的 ~50%。
- **✅ run 跟得准(几度内)**:circle Δ1.1、jogback Δ4、sprint Δ6 → vc 目检「run 还行」坐实。
- **🔑 方法学含义(最重要)**:**MSE-64 是 fight/dance 闭环保真的坏 proxy**。64 维里多数近零维(手指锁0+静态维)
  主导均值,高幅极端 bin 的塌缩在 MSE-64 上几乎不罚 → 降 MSE-64(P0-P3 一路优化的)只改善已经容易的 run,
  **修不了 fight/dance 欠做**。§5.3.1 的 Spearman(MSE,|Δtilt|)=0.62 是**母题难度轴**承载(硬母题既高MSE又高欠做),
  **非可利用的因果**——A/B 实证:模型间降 22% MSE 不降欠做。fierce 例外(gt40→两者35,Δ5,near-full 且不摔;
  但 fierce 开环无 eval chunk 没进 P3 对比)。全 8 段两模型 0 摔(RELAX 饱和,预期)。
- **下一步改向**:(a) 闭环保真要换**幅度感知/极端-bin 加权**的 loss 或 eval 口径(纯 MSE-64 不行);
  (b) 或承认 fight/dance 欠做是小数据+WBC 平滑的根本限——需更多 fight/dance 同母题数据(P3 dance 补语料
  现在动机更强:验"补母题数据能否降欠做"是 vs "换 loss")。**P3 数据胜在预测、不在闭环硬母题保真**。

## 🔬 第二轮三模型评审 + amp_collapse_probe 定位欠做根因(2026-06-12)
对象=Opus 的 P3 矩阵补洞 + P5 闭环 A/B。Fable5+codex/gpt-5.5+mimo(brief/appendix 在
`outputs/reviews/p3matrix_p5_review_*.md`;mimo 第一次死于越界读 LeSONIC 被 opencode 自动拒,
修法=LeSONIC 侧脚本拷成 appendix 进 MaskBeT+提示词明令禁越界)。
- **P3 矩阵裁决=收,但 headline 换口径**:codex 指 @best 有 eval-set selection bias(同集选点+报数)。
  Fable 自答:看**无选点的 final-step(12000)**——flow3 0.0321±0.0006 vs mix 0.0219±0.0012=
  **−31.8%,零选择偏置且比 −22.3% 更大**(flow3 训到底必回升,mix 训到底就是更好)。双报。
- **P5 裁决=收为强负面,但两硬伤**:①GT 锚污染——block/fierce 的 GT replay 本身摔(fall@542/199),
  gt_tilt 41/40 含摔倒姿态,这两段 |Δtilt| 失真;②单 seed(双臂 s2)+ROUNDS=1 对 WASH(差0.26°)太弱,
  但对「欠做一半」量级(Δ~20°)够强——两结论证据强度分开标。
- **🔑 amp_collapse_probe(`scripts/amp_collapse_probe.py`,新)= 回答了「为什么不转化」**:
  521 held-out chunk 上 snapped-expected 预测幅度比(std_pred/std_GT):**mix 0.744 vs flow3 0.736
  ——逐 prompt 几乎相同**;dance/fight 塌最狠(0.59-0.66),run 最轻(0.86-0.91),与闭环 |Δtilt| 逐段吻合。
  三推论:幅度塌缩**开环就有且在 token head**(GT token 喂 WBC 能打 38-41°,WBC 非天花板);
  P3 的 MSE 增益**与幅度正交**(直接解释 P5 WASH);闭环 ~50% 欠做=开环 0.74 塌缩经闭环状态反馈复合放大。
- **下一步排序(信息量/GPU·h,已写入 plan doc)**:①serve 端 α 增益旋钮(snap 前预测×α∈{1.2,1.35,1.5},
  闭环 block/combat/moonwalk,~0.2 GPU·h,杀手实验:单调逼近GT→因果坐实+白捡部署增益;摔→标定幅度-稳定边界);
  ②GT×{0.5,0.75,1.0} WBC 传递函数标定(codex #1,~0.3,顺带修脏锚);③amplitude-aware loss(评价必须报
  闭环 tilt+开环 std比);④dance 补语料**降级**(两臂塌缩相同→"补数据救幅度"先验已弱);⑤P5 多seed 仅条件触发。

## 🎯 GR00T 差距归因 + 根治线 ⑦⑧(2026-06-12 vc 拍板)
vc 问「GR00T N1.7 fight/dance 能做六七成像,StarVLA/MaskBeT 两三成都不到,核心缺哪块」。归因(已写入
plan doc 计划 v2.1):差距不是 backbone 规模,是 **(i) head 建模目标**——GR00T flow-matching 头学
「整段 40×78 联合分布采样」,CE+expected 学「逐格条件均值」,多模态高幅动作被均值化 = amp probe 0.74
塌缩的数学根源,换解码救不了(argmax 塌众数/迭代 std 68%);**(ii) 头先验**——GR00T flow expert 预训
见过海量全幅轨迹,我们 from scratch 只见 5777 帧,P3 已证此量级堆数据修预测不修幅度。
**根治线(vc 已拍板,开跑前仍需报预算)**:**⑦ flow-head swap**(25-50M 不变,MaskGIT-CE 换小型
rectified-flow 头,78 维连续格+末端 snap,trunk 复用,与 CE 臂同数据同 seed A/B;判定门 = std比
0.74→≥0.9 ∧ 闭环 fight/dance |Δtilt| 显著降 ∧ MSE 不劣 CE+10%;~3-5 GPU·h);**⑧ GR00T 教师蒸馏**
(用 GR00T-N1.7-G1-SONIC-BonesSeed 0.0026 对 (text,state) 网格批量生成 token 轨迹当语料,先验注入
绕数据墙;教师生成样本先过幅度诊断,教师自己 std比≥0.9 才有资格;生成 1-2 + 训 2-3 GPU·h)。
⑦修(i)⑧修(ii),组合=完整补上 GR00T 多的那块。④ amplitude-aware loss 降级为 ⑦ 的对照臂。
顺序:先跑已批 ①②(0.5 GPU·h 测 WBC 边界+α 上限)→ ⑦(+可选⑧)。

## ✅ ⑦ flow 头实测大胜 = 治好 fight/dance 闭环欠做(2026-06-12 Opus)
换头(MaskGIT-CE→rectified-flow,共享 trunk,25.3M→25.8M)。CE 头零改动=bit-identical A/B 对照。
脚手架 `maskbet/model_flow.py`+`scripts/{flow_ab.sh,flow_ab_report.py,flow_closed_loop_join.py}`+serve 确定性 flow
(`SONIC_MASKBET_FLOW_SEED`+`SONIC_MASKBET_TEMP`);产物 `outputs/{flow_ab_result,flow_closed_loop}.csv`;
best ckpt `outputs/flow_ab/flow_s2/ckpt_011500.pt`。架构:x1=GT/x0~N(0,I)/线性插值/预测速度场 u=x1-x0,
Euler 8 步 + clip[-1,1] + snap;ckpt 内嵌 cfg(`cfg_from_blob`+`build_model` 工厂,eval/sweep/serve 自动认 flow,
旧 ckpt 回退兼容);`eval.std_ratio` 是幅度门指标;sweep_eval 自动认 flow + 记 std_ratio。
- **开环幅度门 PASS**:std_ratio **0.736→0.941±0.005**(3 seed)。P5 塌最狠母题回升最多(spinclap 0.59→0.88、
  moonwalk 0.74→0.99、combat 0.82→1.14)。snapped MSE 0.0272→0.0328(+20%,单样本非均值最优;MSE-64 坏 proxy 不否决)。
- **🐛 闭环真坑=朴素 stochastic flow 不稳定**:temp=1.0 时 **5/8 摔**(含最易 circle,tilt 110-120 翻倒)。
  逐 chunk 独立采样→chunk 间不连贯+满幅→WBC 被打乱。两个 serve 旋钮定位**两种病因**:
  circle=噪声不连贯(确定性解码治)、spinclap=过幅(降温治)、combat=过幅+WBC 硬限。
- **✅ 部署解=确定性解码(固定噪声 seed,去重采样抖动)+temp=0.6(中幅)**:全 8 段 **0/8 摔**,
  **fight+dance 平均 |Δtilt vs GT| 13.8(CE)→7.9(flow),欠做缺口砍 43%**(moonwalk Δ9.8→1.4、
  spinclap 10.5→3.6、block 24.5→15.5、combat 18.6→13.3;fierce 平、run circle 略过冲 1.1→7.3)。
  **=vc 目检「fight/dance 不好」实证修复且站得住**。
- **三结论**:(1)CE 塌缩是可治的建模目标问题,⑦ 归因正确;(2)采样头闭环部署需两件套——确定性/低方差解码(连贯)
  +尊重 WBC 幅度-稳定上限(temp),「白捡满幅」不能朴素部署;(3)部分印证 P5:combat 满幅确定性仍翻=WBC 对最硬
  fight 母题有真实天花板,temp=0.6 退到可执行幅度即 0 摔。
- **坑账**:harness 连续启动有端口/GPU 竞态(刚结束的 wait task 没释放 5557)→立即 exit 1 空输出;清理与启动分开两调用、
  起前 pgrep 确认 CLEAN。serve `--decode` choices 要加 "flow";harness `SONIC_MASKBET_TEMP` 改 `${VAR:-1.0}` 尊重外层。
- **下一步**:⑦ 已成立,可(a)flow×3seed 跑 closer-to-GT 多 seed 复核 +(b)⑧ GR00T 教师蒸馏(flow 头已验证可当蒸馏目标)
  叠加看能否再逼近 GR00T 六七成;或先把 det+temp0.6 当 serve 默认发布。
- **live demo 默认已切 best flow**:`scripts/maskbet_sonic_live_demo.sh @flow3` 默认 ckpt 改成
  `outputs/flow_ab/flow_s2/ckpt_011500.pt` + 部署配置(decode 留 argmax→serve 见 flow 自动切 flow 采样;
  `SONIC_MASKBET_TEMP=0.6`+`SONIC_MASKBET_FLOW_SEED=0`,CE 头忽略这俩故安全)。回旧 CE 头:`MASKBET_CKPT=.../flow3/ckpt_006000.pt`。
  notebook cell `!bash scripts/maskbet_sonic_live_demo.sh @flow3` 直接跑出 0/8 摔配置;复用逻辑=5557 有 server 就复用,
  起前先 `pkill -f serve_maskbet_sonic` 防拿到旧 config。

## 三轮评判(2026-06-12,Fable5 审计 Opus 的 ⑦ 实施)——结论维持,降半档为「强阳性 n=1」
- 【品味】🟢:cfg 内嵌 ckpt+`cfg_from_blob/build_model` 工厂消掉下游特殊分支;CE 零改动=bit-identical 对照;
  零重训双旋钮(确定性噪声/温度)解剖出两种闭环病因的 2×2 表是 session 最值钱产物。
- 【证据链四缺口】(结论不翻,账如实记):🔴(a)预注册 MSE 门(≤CE+10%=0.0299)实际 **FAIL**(0.0328,+20.6%),
  被闭环终点事后改判——改判合理但须记作「门 FAIL」非「略升可接受」;🟡(b)部署口径(det seed0+temp0.6)的开环
  std_ratio **从未量过**(0.941 是 temp=1.0 随机解码,`eval.std_ratio()` 无 temp 参数);🟡(c)闭环 ROUNDS=1/段
  违「≥3 round」家规,σ 未知;🟡(d)**fierce 行非证据**:flow 与 CE max_tilt 精确同 34.5°=峰值落在共享 80 步
  bootstrap 窗内,probe 没剔 bootstrap;剔 fierce 后 fight+dance 均值 flow 8.4 vs CE 15.9(−47%)反更强。
  小账:seed=0 任意未测敏感性;temp<1 缩初始噪声=分布外 hack 经验有效非原理解。
- 【裁决】**⑦ 复核包(~2 GPU·h:部署口径 std_ratio≈0 + ROUNDS=3≈1.5h + probe 剔 bootstrap≈0 + 3 噪声 seed≈0.5h)
  先于 ⑧ 蒸馏开跑**——⑧ 判定门就是 ⑦ 的口径,尺子先复核。已写入 plan HTML(⑦ WIN box 后评判 box+§7 记录行)。

## 目检复判(2026-06-12,vc 实看 demo):「动作学得还是不像」——⑦ 修幅度没修内容
- ⑦ 两个判定门(std_ratio/|Δtilt|)全是**幅度代理**,没有一个量「内容」;门全过、眼睛不过=Goodhart。
- 「像不像」≈held-out 逐帧 MSE:GR00T 0.0026(比常数模板 0.0353 好 13.6×,目检六七成)/CE 0.0272(好 23%)/
  **flow 0.0328(只好 7%,目检两三成,且 ⑦ 在此轴是退步)**——此轴排名与眼睛完全一致。机器人=对的能量跳「即兴」,
  不是跳「编舞」。幅度(根因 i,⑦ 已修)与内容(根因 ii,5777 帧 from-scratch 学不出)是两条独立的腿,(ii) 一步没走。
- **排序改判:⑧ 蒸馏升主线(注入内容先验);开跑前先落 M0 像度硬指标(executed-vs-GT 关节 DTW/逐帧 MSE 进 p1b
  probe,~0 GPU·h,否则再被幅度代理骗);P2 phase/K(K=3 仅 0.06s,模型不知跳到第几秒)为便宜对照;⑦ 复核包并入 M0;
  demo-only 捷径=放弃 held-out 故意过拟合 8 段+phase(GR00T 的像本就是记忆)诚实摆桌。**已写入 plan HTML ⑦ 评判 box。
