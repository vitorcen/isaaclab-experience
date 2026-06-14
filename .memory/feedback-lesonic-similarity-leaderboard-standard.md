---
name: feedback-lesonic-similarity-leaderboard-standard
description: LeSONIC token-生成模型的权威评分=open-loop token-MSE 相似度 + macro/frame 双指标排行榜,训练时自动扫
metadata:
  type: feedback
---

# LeSONIC 标准:相似度评分 + 排行榜(对标 LeIsaac 的 benchmark 标准)

**2026-06-14 用户定为标准**:SONIC motion-token 生成模型(GR00T / StarVLA / FlowDP …)
必须有**和原始动作的相似度评分**来挑 best ckpt + 排行,**训练时自动跑**(不闷头训完再看)。
这是 SONIC 侧的 [[feedback-5round-benchmark-standard]] / [[feedback-20round-strict-benchmark]] 类比物。

**Why**:之前 FlowDP-SONIC 一上来设 15000步≈250epoch 闷训,没评分 → 既无法挑甜点也浪费。
用户指出:① 该有相似度评分;② flowdp 甜点是**几个 epoch 不是几百**——实证 LeIsaac flowdp PickOrange
sweep best 落在 **step 9800 ≈ 4.3 epoch**(14000步≈6.2ep的run;36293帧/batch16→2268步/ep),GR00T SONIC ~8ep。

## 评分口径(权威)
- **指标 = open-loop token-MSE**:teacher-force 每帧录制 obs → 预测 action chunk → 与录制 GT token
  做 MSE,**raw FSQ-grid 单位**(反归一化回 token 空间,与 GR00T 可比)。镜像 GR00T `open_loop_eval`。
- **综合 7 动作排行(用户问的)= 双指标**:
  - **macro-MSE(主排序,升序)**:7 动作各自 MSE 再**等权平均**——动作长度差 10×(kick 165帧 vs
    macarena 1375帧),等权才公平、不被长动作主导。
  - frame-MSE(次):所有帧 token SE/总 token 数=帧加权,**与 GR00T 0.0011 口径可比**。
  - skill = 模板MSE/模型MSE(>1 才算学到);beat n/7 = 几个动作低于自己 per-motion-mean 模板。
- **模板基线**:per-motion-mean(每动作自己的均值 token,**要打过的硬基线**)、global-mean。
- **参考(同任务 raw token 空间)**:GR00T N1.7 best ≈ **0.0011**(frame);per-motion-mean ≈ 0.039;global ≈ 0.048。

## 脚手架(LeSONIC/scripts/)
- `sonic_flowdp_openloop_eval.py`:`--ckpt <pm>` 单评 / `--sweep <run>` 扫全 run / `--leaderboard-from-csv`
  从 CSV 重建榜;`--csv` 追加(`--skip-existing` 去重)/`--leaderboard` 出 markdown;`--epoch-steps` 出 epoch 列。
- `sonic_flowdp_eval_watcher.sh`:**训练时自动扫**——poll checkpoints,新 ckpt 即评分→追加 CSV→重建
  `outputs/<run>/leaderboard.md`;open-loop 无需 Isaac(纯前向)可与训练共卡;训练 End of training + 全评完即退。
  kernel-6.17 import 腐蚀靠 score_one 内 retry 兜。
- 训练侧 epoch-aware:`sonic_flowdp_train.sh`/`sonic_flowdp_watchdog.sh` 默认 STEPS=2400(~40ep)
  SAVE_FREQ=120(~2ep/ckpt=20个,密覆盖低 epoch 区)。**别再设几百 epoch**。

## 用法
```bash
# 训练起来后,后台自动评分+出榜:
setsid bash scripts/sonic_flowdp_eval_watcher.sh &   # 写 outputs/flowdp_sonic/{openloop_eval.csv,leaderboard.md}
# 或训练完手动扫:
PYTHONPATH=../LeIsaac/FlowHeads python scripts/sonic_flowdp_openloop_eval.py \
    --sweep outputs/flowdp_sonic --csv .../eval.csv --leaderboard .../leaderboard.md --epoch-steps 59.6
```
关联 [[flowdp-sonic-bonesseed-live-state]] [[sonic-wbc-vla-route]] [[feedback-incremental-eval-during-training]]。
