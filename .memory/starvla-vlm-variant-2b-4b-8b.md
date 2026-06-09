---
name: starvla-vlm-variant-2b-4b-8b
description: StarVLA QwenGR00T 换 VLM 骨干(2B/4B/8B)的零源码改动配方 + 每变体 bs/save 密度
metadata:
  type: project
---

StarVLA `QwenGR00T`(冻 VLM 只训 GR00T flow-matching head)换骨干 **零源码改动**:
`QwenGR00T.py:155-156` 运行时把 `cross_attention_dim` 对齐到所加载 VLM 的
`model.config.hidden_size`。config 里的 `framework.qwenvl.vl_hidden_dim` 对 QwenGR00T 是
**死字段**(只有 QwenPI/QwenAdapter/Layerwise 那些 head 才回读)。换变体 = 一个 config yaml
改 `base_vlm` + `run_id`(+ 可能 bs)即可,见 `LeIsaac/scripts/training/starvla/configs/`。

**Qwen3-VL hidden_size(实测 config.json):** 4B=**2560** / 8B=**4096** / 2B=待下载确认(疑 2048)。
⚠️ 早先误记 4B=2048,实为 **2560**(serve size-mismatch 报错 `current model 2560` 实锤)。
`vl_hidden_dim` 对 QwenGR00T 是死字段(运行时按所加载 VLM 的真 hidden 对齐),填错不影响训练,
但 **eval serve 必须用对的 base**:base 与 ckpt 的 VLM 大小不匹配 → load 时 size mismatch。
**🛡️ 防呆**:`starvla_strict_eval.sh` 现按 ckpt 路径里的 `2b/4b/8b` tag 自动解析对应本地
`models--Qwen--Qwen3-VL-<N>-Instruct` snapshot(不再硬编码 4B base);且把 "size mismatch"
和真 corruption 区分——配错 backbone 立即报 `BACKBONE MISMATCH` 退出,不白重试 4 次。

**每变体 batch / 显存(4080-32G 或 4090-48G):**
- 2B(bs=8)≈19G/32G;4B(bs=8);8B(bs=4)≈**30G/48G 不 OOM**(bs=8 会 ~40-48G 踩线 OOM,实测 bs=4 对)。

**save_interval 密度(过拟合峰是悬崖,要采够)**:峰是**样本数驱动 ~120k**(4B 15k步×bs8=120k,
8B 30k步×bs4=120k,两者峰值同样本量,**非**模型大小驱动)。崖在 bs=8 下约 3k 步宽。
- 4B = **每 1500**(20 ckpt);8B = 6000(偏稀,30k 峰靠运气采到);2B = **每 1000**(30 ckpt,
2B 特征更弱可能峰更早更尖,密一档兜住)。2B watcher 设 `MIN_STEP=3000`(<3000 欠训不 eval)。

**Qwen3 路径死字段(别误算省显存)**:`QWen3.py:53` 把 attn 强制 `sdpa`(flash-attn 配了无效);
`trainer.gradient_checkpointing` Qwen3 不读 + 冻结 VLM 无 backward 可省;单卡 ZeRO-2 不分片参数。
真显存大头 = `output_hidden_states=True` + `repeated_diffusion_steps`(默认 8)把 hidden repeat 8 倍
喂 DiT cross-attn → **OOM 第一杠杆是它降到 4,再动 batch**。

云端两台箱:westd:15528=4090-48G(8B,训完可关机)、westc:31709=4080S-32G(2B)。
都有 starvla env + starVLA repo@e8f8fbb + `leisaac-pick-orange_old` 数据集 + network_turbo;
口令在 `pass autodl/westd`(westc 同口令)。关联 [[starvla-8bit-eval-load-corruption]]
[[starvla-8b-2b-sweep-result]] [[starvla-so101-cloud-training]]。
