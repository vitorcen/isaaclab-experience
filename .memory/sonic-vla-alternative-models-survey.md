---
name: sonic-vla-alternative-models-survey
description: 2026-06-11 三方调研「除 GR00T 外哪些模型胜任 FSQ token→SONIC WBC 场景」——排序结论 + 空白生态位 + FSQ 32 级事实核查
metadata:
  type: project
---

**2026-06-11 三方头脑风暴**（Claude + codex gpt-5.5 + mimo-v2.5-pro + 两路 web 事实调研），
完整文档 `LeSONIC/doc/sonic_vla_alternative_models_survey.html`。关联 [[starvla-sonic-ab-baseline]] [[sonic-wbc-vla-route]]。

## 三方收敛结论
任务本质 = **prompt-conditioned 离散 motion-token 生成**，不是操作臂 VLA（视觉只是弱条件）。
排序：① **StarVLA CE 路线深化**（三方一致 Top-1；差距 6.7× 更可能来自无 history/逐维独立 CE
不建联合分布，而非 backbone——别花预算换更大 VLM）② **RDT2-VQ**（清华，Apache-2.0，唯一原生
「AR+CE over 离散动作 token」的大发布=目标函数同构；但 7B 24G 边缘 + proprio 推理路径是零填充
死代码）③ **小型 masked-transformer 从零搭**（MoMask/1xgpt 模板，50-200M，MaskGIT 10-12 轮
迭代去掩码——轮数与网格大小无关）。对照臂=SmolVLA(450M)/MiniVLA(1B MIT RVQ)。

## 关键事实（核查过）
- **FSQ 真实配置**：`all_mlp_v1.yaml` = `num_fsq_levels: 32` × 32 维 × `max_num_tokens: 2`
  → 64 维，**每维 32 级**（非 33）。我们 CE head 33 bins 是覆盖超集，k=+16 死 bin 无害；
  做 masked 解码时 33 = 32 + MASK 正好。
- **2560-token 红线**：64×40 逐 token AR 闭环 ~30-60s/chunk 不可行；解法=①OFT 单次并行
  （现 CE head 已是）②MaskGIT 迭代 ③FAST BPE 压缩。openpi 里已有 `FSQTokenizer` 类先例。
- **空白生态位**：「对 SONIC FSQ 网格做 AR/masked 离散建模」截至 2026-06 无人发表
  （领域收敛在连续 latent→tracker：GR00T 回归/LeVERB CVAE/FRoM-W1 自有 VQ）。
- **2026 新货**：Qwen-VLA（2026-05-29 论文，Qwen3.5-4B+1.15B DiT expert；**2026-06-11 核查：
  权重未放**——GitHub repo 只有论文/blog/benchmark 无 ckpt 无 license，HF/ModelScope 均无官方仓；
  放出后才是 backbone 升级件，调研 agent 曾误报"weights released"）；FRoM-W1（OpenMOSS，
  Apache-2.0，LLM→离散 token→tracker→真机 G1，唯一全开源同构系统，单一信源待核）；
  GR00T N2 2026 底才发布。
- **license 卫生差**：UH-1/ScaMo/NORA/Wall-X 无 LICENSE 文件；GO-1/Unitree/KungfuBot 是 NC。
  干净清单：StarVLA(MIT)/RDT2/openpi/T2M-GPT/1xgpt/FRoM-W1(Apache-2.0)/MoMask/MiniVLA/OFT(MIT)。

## 行动序（P0→P3）
P0=CE head 升级 masked 并行解码 + history（仍只改 QwenPI_CE.py，验证用 8 窗口短跑）；
P1=路线 B 小 masked transformer 并行实验；P2=backbone 换 Qwen-VLA-4B（只改 config）；
P3=RDT2-VQ 8bit 塞 24G spike。统一告诫：任何候选 stock flow/diffusion head 必塌模板，换离散 head 是前置条件。
