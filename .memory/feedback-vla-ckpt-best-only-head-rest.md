---
name: feedback-vla-ckpt-best-only-head-rest
description: StarVLA 冻VLM训练存储纪律 — 每run只留best的full ckpt+全部head;非best full删;eval非best时临时重建full→eval完即删,绝不覆盖best
metadata:
  type: feedback
---

# StarVLA 冻-VLM ckpt 存储纪律：best 留 full，其余只留 head

**Why**：冻-VLM head-only 训练的 full ckpt 极大（4B≈10G / 9B≈20G / 8B≈18G），但里面 VLM 部分**每个 step 完全相同**（冻结），可训的只有 head（`action_model.*` + `project_layers.*`，~1-1.4G）。留一堆 full = 几百 G 死重，磁盘满了下次训练 `torch.save` ENOSPC 崩（伪装成 flash-attn）。full 随时可由 **head + vlm_base 用 `merge_head.py` 重建**，所以非-best 的 full 是纯冗余。见 [[frozen-vlm-head-extraction-sweep]]、[[starvla-checkpoint-resume-migration]]。

**How to apply**：
1. **每个 run 只留 best 那一个 step 的 full ckpt**（`checkpoints/steps_<best>_pytorch_model.pt`，做发布/直接 serve 用）+ **保留全部 head**（`heads/steps_*_head.pt`，每个 step 一份，做 resume/重建）+ vlm_base（`_head_sweep_tools/vlm_base_<backbone>.pt`，一族一份）。
2. **删掉所有非-best 的 full ckpt**（reconstructible，别占盘）。
3. **eval 非-best ckpt 时**：临时 merge 出**那个 step 自己的** full（`steps_<step>_pytorch_model.pt`，文件名带 step → 天然不和 best 同名、**不覆盖 best**）→ eval → **`rm` 掉这个临时 full**。strict 脚手架本身就这么干（merge → eval → `rm "$full"`），best 的 full 永远不动。
4. **超期/已发布的旧 run**（无 head 提取、被新家族取代）：留**已发布的 best full**（HF 上有副本）+ 删非-best full；不必为超期 run 补抽 head。
5. **其它模型族同理**：ACT/DP/SmolVLA 等非 head-分离结构 → 退化为「留 best + 删中间 sweep ckpt」（见 [[feedback-training-output-cleanup]]）。

**判据**：full ckpt 是否 = 该 run 上榜/发布的 best？是→留；否→删（head 还在就随时能重建）。删前 `pgrep` 确认没训练/eval 进程在用。

**权威脚手架 = `LeIsaac/scripts/ckpt/`（提交，代码与产物分家）**（2026-06-09 重构定型）：
- **代码进提交目录**：`LeIsaac/.gitignore` 有 `**/outputs/*` → 早先把工具放 `outputs/_head_sweep_tools/` = fresh clone 即丢、不进版本控制。工具搬到 `scripts/ckpt/`（提交），靠 `--args` 传路径位置无关；**大产物**（base / delta / ckpt）继续留 gitignored 的 `outputs/`。
- **`prune_ckpts.py`（通用，唯一权威）**：方法 = **对 base 逐张量 diff，不是按前缀切**。覆盖三类：① 前缀干净（StarVLA `action_model.*`+`project_layers.*`）—— diff 出的 delta 与前缀法**逐键完全一致**（实测 0 额外键）；② 层内交错（Wall-X expert 嵌 `model.layers.*` frozen 0.945、π0.5 嵌 `paligemma_with_expert.*` frozen 0.883）—— 没干净前缀，diff 照样精确隔离；③ 全量 FT（ACT/DP/SmolVLA/X-VLA，frozen≈0）—— **自动 REFUSE**（`--min-frozen` 默认 0.5），不产无用 delta = **防呆**。dry-run 默认，`--apply` 才删；支持 `.pt`+`.safetensors`。
- **`merge_ckpt.py`（通用重建）**：`base + delta → full`，格式按扩展名推断（两种都支持）。重建按构造字节精确。
- **验证无破坏（删 full 前必做，工具内置 GOLD 门控）**：每个 delta 重建后 `set==set(full)` 且逐张量 `torch.equal`，**全过才删**；另用独立 `merge_ckpt.py` 重建对比 kept full 端到端复验（2026-06-09 Wall-X 实测 keys✓ 张量全等✓）。
- **已处理**（2026-06-09）：3 个旧 GR00T StarVLA run（前缀法）+ Wall-X sweep/oss05（diff，base 7.9G 跨 run 共享，省 24.9G）+ π0.5-expert（diff，base 8.3G，省 9.4G）。**判定见 `scripts/ckpt/README.md` 全家族表**。
- **防呆默认**：CLAUDE.md「Post-training cleanup」已写入「每个新 run 训完无脑跑 `prune_ckpts.py`」—— 冻骨干自动塌缩，全量 FT 自动跳过。**无损 + 续训不受影响**：工具只删 `model.safetensors`，**从不碰 `training_state/optimizer_state.safetensors`**；续训被抽 head 的 step = merge 还原 model + 现成 optimizer。**optimizer_state 是续训料一律保留，别为省盘删**（用户 2026-06-09 明确）。
- GR00T N1.6/N1.7 **跳过**：已发布 HF（有副本）+ 每 run 仅 2 ckpt + HF 分片 safetensors 格式摩擦，只留 best。OpenVLA/dreamzero LoRA adapter 65–208M 已极小，无需。

关联：[[feedback-training-output-cleanup]]（一族一 dir + 3-6 ckpt）、[[feedback-cloud-env-reuse-disk-cleanup]]（起训前清死重防 ENOSPC）、[[starvla-checkpoint-resume-migration]]（ckpt 只存权重 + 迁移靠 head+vlm_base）。
