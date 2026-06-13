---
name: maskbet-route-b-submodule
description: MaskBeT = 路线B「从零小型 masked transformer」已落为 LeSONIC/MaskBeT submodule (git@github.com:vitorcen/MaskBeT.git)
metadata:
  type: project
---

**2026-06-11** 路线 B（survey Top-3 之③）落地为独立 repo + submodule。
关联 [[sonic-vla-alternative-models-survey]] [[starvla-sonic-ab-baseline]] [[sonic-masked-ce-p0-result]]。

## 是什么
**MaskBeT (Masked Behavior Transformer)** — backbone-free 小模型（v0 默认 d=512/8层 ≈30-50M），
MaskGIT 式并行解码离散 motion token。命名谱系 ACT→DiffusionPolicy→VQ-BeT→**MaskBeT**
（VQ-BeT 自回归吐 token，MaskBeT 并行掩码预测）。目的 = 不靠 VLM 验证「6.7× 差距来自目标函数
（逐维独立 CE vs 联合 masked 建模）而非 backbone」。

## 仓库 / submodule
- 远端 `git@github.com:vitorcen/MaskBeT.git`（MIT）。首 commit + force-push 由**用户自己做**
  （我 scaffold 完已 `git update-ref -d HEAD` 退回 unborn main，文件留工作区）。
- `LeSONIC/MaskBeT` 已 `git submodule add`（.gitmodules + gitlink staged，未 commit）；用户重 commit
  后哈希变，需回 LeSONIC 重 `git add MaskBeT` 更 gitlink 再 commit 父仓。
- 结构（与 LeSONIC 主仓同构）：`maskbet/`=包代码(config/model/masking/data/train.py)、
  `scripts/`=shell 入口(smoke.sh/train.sh)、`doc/`=maskbet_design.html(中文+SVG)。
  **包目录定为 `maskbet/` 不用 `src/`**——研究仓原地 `python -m` 跑，src 布局要 `pip install -e .`
  纯摩擦(每 env 重装)；src 也不是包名，真要也是 `src/maskbet/`。
- `scripts/smoke.sh` 全过：model forward/decode OK + 合成数据 30 步 train sanity（默认 d=512/8 层
  =**25.3M 参数**，落在「小 30-50M」目标内）；init loss≈ln(33)=3.66、bin_acc≈1/33、decode 收敛 value∈[-1,1]。

## 关键对齐（与 QwenPI_CE 同口径，便于比 MSE-64）
grid = action_horizon 40 × action_dim 78 = 3120 cell；n_bins 33（k∈[-16,16]，value=(bin-16)/16，
MASK=id 33）；condition = prompt_id(8 条 LAFAN flow3 指令) + proprio state 46 维 → n_cond_tokens 个
summary token 前置到网格，双向 encoder 预测每 cell bin 分布。要打的锚：GR00T 0.0026 /
CE v1 0.0174argmax·0.0125E / 模板 0.0367。

## Fable 5 修订（2026-06-11，v0 条件路径重构 + 预训练路线落文档）
- **条件重构**：`prompt_id` 8 条查表是死胡同（预训练 1069 条文本进不来）→ 现在
  `cond={"prompt","state"}`：text 路径构造期二选一（`text_embed_dim=0`=查表基线 / `>0`=冻结
  文本塔 embedding 投影，forward 单路径无分支）；state 升级 `(B,K,46)` history K=3 内建
  （P0 教训不等 v0.3）；删掉假的 4×cond_query 广播瓶颈 → 1 text token + K state token。
- **train.py 补齐**：warmup+cosine LR、原子 ckpt（tmp+rename keep-last-3）、`--resume`；
  smoke 曾被 OOM killer 杀（exit 137）= serve_starvla 4B 拉起瞬间吃 RAM，非代码问题。
- **谱系修正**：BeT(Shafiullah 2022)→VQ-BeT→MaskBeT（不是 ACT 系）。
- **预训练路线判定（用户思路 = 对，三 gap）**：①LeVERB action=29 维关节角非 token →
  必须先过冻结 SONIC FSQ encoder tokenize（`UniversalTokenModule.encode("g1",...)`，
  `GR00T-WholeBodyControl/gear_sonic/trl/modules/universal_token_modules.py`，权重在 448M
  WBC ckpt 内，已核查可调用）；②文本条件已修；③域偏移：LeVERB 偏坐立/行走，主语料应为
  AMASS-G1，curriculum=AMASS-G1→LeVERB(Apache-2.0 唯一干净配对)→LAFAN-G1→flow3 微调。
- **模型大小**：flow3 only ≤25M（已饱和）；+LeVERB ~743k帧→50-86M；+AMASS-G1 ~8M帧→
  **86M 推荐（d=768/12L/12头）**，200M 上限——别堆参数，赌的是同空间预训练。
- 预注册预期：v0.2 flow3 直训大概率 ≈ CE v1 0.0174（P0 预示），是基线不是赌注。

## 待办
v0.1=`data.py` 对接 `sonic_lafan_flow3`（同 FSQ 量化+K 帧 history）→ v0.2 直训 8 窗口
MSE-64 → v1 tokenize 管线（LeVERB/AMASS-G1 → SONIC encoder → token 语料 + 文本 embedding
缓存）→ v2 86M 预训练+微调。
