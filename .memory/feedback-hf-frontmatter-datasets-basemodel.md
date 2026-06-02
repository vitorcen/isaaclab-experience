---
name: feedback-hf-frontmatter-datasets-basemodel
description: HF model card YAML frontmatter 必须填 datasets: 和 base_model: (无则空数组)，否则 HF UI 不会自动渲染数据集卡片 + base-model 血缘关系
metadata:
  type: feedback
---

发布到 HuggingFace Hub 的任何 model card README，YAML frontmatter **必须**填 `datasets:` 字段（列每一条用到的源数据），以及 `base_model:` 字段（不是 finetune 则给空数组 `[]`）。

**Why:** HF UI 会按 frontmatter 自动渲染两个关键 widget：
- `datasets:` → 模型卡顶部 "Datasets used to train" 卡片，点击直达数据集页面，是 HF 内部 cross-reference 的唯一机制（body 里写 markdown 链接不会被识别）
- `base_model:` → "Base model" 血缘标签 + finetune leaderboard 归类，HF 模型 graph 完全依赖这个字段

不填两者 = 模型在 HF 生态里是孤儿，没法被 dataset filter / base model tree 找到。用户在 wsagi/MimicKit-G1-LAFAN 上线时强调要补这条规则。

**How to apply:**

固定 frontmatter 模板（按场景填）：

```yaml
---
license: apache-2.0
library_name: <mimickit|lerobot|gr00t|openvla|...>
tags:
  - <task>            # motion-tracking, manipulation, ...
  - <robot>           # unitree-g1, so-101, franka, ...
  - <algo>            # ppo, diffusion-policy, act, vla, ...
  - <framework>       # isaac-lab, lerobot, ...
datasets:
  - <hf_org>/<dataset_repo>     # 每一条用到的数据集都列上
  - <upstream_chain_if_any>     # 如果有 retarget/repack 链，把直接读的也列上
base_model: []                  # 不是 finetune 给空数组
# base_model:                   # 若是 finetune:
#   - <hf_org>/<base_model_repo>
pipeline_tag: <reinforcement-learning|robotics|...>
---
```

**dataset chain 怎么填：**
- 只列**实际读取的最直接上游**（够用了，再补一两层关键节点）
- 不要把每层都列，HF UI 会显示成一堆重复卡片
- body 里用表格写完整 chain（许可证 / 帧区间 / 重打包者）

**checklist（每次 publish 前自检）：**
- [ ] `datasets:` 至少 1 条且都是 `org/repo` 格式（不是 URL）
- [ ] `base_model:` 字段存在（空就 `[]`，不能省略键）
- [ ] `library_name:` 与实际 framework 一致（影响代码示例自动渲染）
- [ ] `tags:` 至少 4 条（task / robot / algo / framework）
- [ ] body 顶部还要按 [[feedback-hf-readme-project-links]] 加 `vitorcen/isaaclab-experience` + `vitorcen/LeIsaac` 项目链接

**坑：dataset slug 必须真实存在，不存在就不渲染（沉默失败）**
- HF 不会报错也不会警告，写错的 dataset slug 静默忽略
- 验证方法：`python -c "from huggingface_hub import HfApi; HfApi().dataset_info('<org>/<name>')"`，404 就是不存在
- 常见踩坑：原作者删/改名后还引用旧 slug；用 `HfApi().list_datasets(search='...')` 查现役 fork

**已应用：**
- `wsagi/MimicKit-G1-LAFAN` (2026-06-02) — datasets: lvhaidong/LAFAN1_Retargeting_Dataset + ember-lab-berkeley/LAFAN-G1; base_model: []。注意 `unitreerobotics/LAFAN1_Retargeting_Dataset` 已下架，社区最大 fork 在 `lvhaidong/`

关联：
- [[feedback-hf-readme-project-links]] — body 顶部必带的项目链接（这条管 frontmatter，那条管 body）
- [[feedback-style]] — 中英对照 + 设计文档先行
- [[hf-upload-tricks]] — hf_transfer + upload-large-folder 配方
