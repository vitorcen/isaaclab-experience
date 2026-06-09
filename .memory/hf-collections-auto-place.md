---
name: hf-collections-auto-place
description: 发布新 HF 模型后必须加进对应 Collection — scripts/hf_make_collections.py 是权威；按任务族选 collection
metadata:
  type: feedback
---

# HF 发布后自动归入对应 Collection

**Why**：HF 个人首页(`huggingface.co/wsagi`)没有按名字自动分组，**Collection 是唯一原生分组**。
发完一个新模型若不加进 Collection，首页就是一堆散 repo，看着像没整理。用户要求：**后续每次发布自动找合适的 Collection 放进去**。

**How to apply**：
1. 权威脚本 = [`scripts/hf_make_collections.py`](../scripts/hf_make_collections.py)（幂等，`create_collection`/`add_collection_item` 都 `exists_ok=True`，随便重跑）。
   `GROUPS` = `[(title, desc, [(repo_id, "model"|"dataset"), ...]), ...]`，一族一个 Collection。
2. **发完新 repo → 把它加进 `GROUPS` 里对应族 → 跑 `python scripts/hf_make_collections.py`**（用缓存的 HF write token，别在命令行贴 token）。
3. **任务族 → Collection 映射**（按任务选，不按模型架构）：
   - **LeIsaac PickOrange**(`wsagi/leisaac-pickorange-...`) = 所有 SO-101 pick-orange-and-place 策略（ACT/DP/SmolVLA/X-VLA/OpenVLA/π0.5/GR00T-N1.6/N1.7/**所有 StarVLA 变体**含 Qwen3-VL-8B GR00T+PI_v3、Qwen3.5-2B/4B/9B PI_v3）。**新的 PickOrange 模型一律进这个**。
   - **RoboCasa** = GR00T RoboCasa 厨房操作。
   - **MimicKit** = G1 LAFAN motion-tracking。
   - **SONIC** = GR00T-in-loop 驱 SONIC WBC（含 VLA dataset）。
   - **HumanoidBench** = HumanoidBench RL baselines。
   - 任务族没有现成 Collection → 在 `GROUPS` 新建一条。
4. Collection 内顺序靠 HF 网页拖拽（API 不保证序）。

**判断"合适"**：看模型解决的**任务**（PickOrange? RoboCasa? motion-tracking?），不是骨干/算法。同一 benchmark 的不同策略族都进同一个 Collection 方便横评对照。

关联：[[three-box-sweep-live-state]] StarVLA PI_v3 家族、[[feedback-hf-readme-project-links]] model card 链接规范、[[hf-upload-tricks]] 大文件上传。
