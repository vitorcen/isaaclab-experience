---
name: select-layer-depth-brainstorm
description: GR00T select_layer 取层深度的源码实锤 + 三模型(Opus/codex/mimo)脑暴——往中层更后(19/21)预测掉分,GR00T 无官方倒U/统计只"省算力",per-release值 N1.5=-4/N1.6=16/N1.7=12,解冻4是N1.6的值N1.7=0
metadata:
  type: project
---

# select_layer 取层深度 — 源码实锤 + 三模型脑暴（2026-06-15）

用户问：StarVLA-Qwen3.5-4B(32层)+GR00T_v2(N1.7)头当前 `select_layer=12`(37.5%深度,best 66.7%)，
改到更靠后的 **3/5≈层19 / 2/3≈层21** 会怎样？三模型(Opus 4.8 主 + `codex -m gpt-5.5` + `opencode -m xiaomi/mimo-v2.5-pro`)脑暴。
文档：`LeIsaac/docs/training/select_layer_depth_brainstorm.html`。关联 [[starvla-gr00t-v2-n17-head]] [[e1-midlayer-sweep-live-state]] [[feedback-mimo-independent-review]]。

## 🔴 GR00T 官方源码到底说了什么（实锤，先划清边界）
- **倒 U 是我们的分析框架，不是 NVIDIA 原话。** GR00T 对 select_layer 唯一注释 = `qwen3_backbone.py:86` **"needed since we don't use these layers. Also saves compute"** —— 纯省算力,无"中层更好"论证、无 per-layer 成功率、**repo 里零 ablation table**。背后消融(若有)是 NVIDIA 内部未公开。
- **12 不是从来如此,每个骨干各自调**:N1.5(Eagle2.5)=`-4`(负索引倒数第4) / N1.6(Eagle3)=`16` / N1.7(Qwen3)=`12`。骨干换了层数不同→**绝对层号不可跨骨干比**,别读成"NVIDIA主动往浅调"。源码:`gr00t_n1d7.py:47`/`gr00t_n1d6.py:28`/`configuration_eagle2_5_vl.py:43`。
- **解冻"4"是 N1.6 的值,N1.7 其实=0(全冻LLM)**:`gr00t_n1d6.py:24 tune_top_llm_layers=4` vs `gr00t_n1d7.py:43 =0`。我们配方是**混血**:select_layer=12 取自N1.7 + 解冻4取自N1.6(理由=Qwen3.5是通用骨干像Eagle,非N1.7的Cosmos机器人预训练骨干;N1.7敢全冻0正因换了具身骨干)。**解冻4是我们工程判断非N1.7原配。**
- **唯一取层统计是我们自测的**:末层(porting bug)13.3% / 中层12 frozen 48.9%~unfreeze best 66.7%(强证"取层是一阶因子",但是我们的不是GR00T的)。

## 三模型预测（强共识：往后挪掉分，机制=倒U+image-token深层塌缩）
机制:取层有用性随深度是**倒U**;GR00T_v2头的AlternateVLDiT有4个cross block**只attend image token位置**→需空间保真视觉特征;decoder-only VL-LLM里image token过~50%深度就塌向next-token语言预测(logit-lens/线性探针),视觉内容被压缩→对image-only头杀伤最大;小橙子(10-40px)空间保真比抽象语义更稀缺。
- **层8(25%)**: codex 35-45% / mimo 50-55%(略差12)
- **层12(37.5%基线)**: best 66.7%,近最优
- **层19(3/5,59%,解冻15-18)**: 一致 **~40-50% E,掉15-25点**,下行明确(Opus清晰赢仅~30%概率)
- **层21(2/3,66%)**: ~25-35%,接近failure(末层13.3%弱化版)

## 纠偏 + 下一步
- **深度比例**:N1.7的12来自36层Cosmos(12/36=33%),映射32层≈**层11**→我们的12比例上**已略偏深**,绝非"太浅该往后补"。
- **隔离归因**:换层别同改解冻集(混淆"喂哪层"vs"训哪层")→干净A/B先跑 `tune_top_llm_layers=0` frozen head-only。
- **若要验证有无更优层**:三家都建议往**浅(层8)或真中点(层16)**探,**不往后**。最省=层8 frozen head-only 5-round扫→甜点strict 20-round(~1h/4090),对照12 frozen 48.9%。除非8意外赢12,否则不值得为19/21烧~2×算力(倒U右坡论据强,大概率负)。真正提升空间更可能在**数据/骨干**非取层。
