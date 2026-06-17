---
name: feedback-headed-eval-default
description: 标准——eval 一律用"有头"(GUI 窗口可见,policy_inference 去掉 --headless 保留 --enable_cameras),不再默认无头;且必须把 wall_cap 抬到不截断,否则有头因渲染慢会假性掉分。是 benchmark 口径的一部分。
metadata:
  type: feedback
---

# Eval 标准 = 有头（2026-06-13 用户定为标准）

**标准(非可选偏好)**:PickOrange(及同类 Isaac)闭环 eval **一律有头**——`policy_inference.py` **去掉 `--headless`、保留 `--enable_cameras`**,让 Isaac 窗口显示在 `DISPLAY=:0`。隔壁 FlowDP 本来就有头跑,这台 4090 支持。属 benchmark 口径一部分,与 [[feedback-5round-benchmark-standard]] / [[feedback-20round-strict-benchmark]] 并列。

**Why**:用户要能**直接看机械臂行为**(visual sanity),无头是盲跑。

**How to apply**:
- 改 eval 启动:`DISPLAY=:0 ... policy_inference.py ... --enable_cameras`(**不要** `--headless`)。
- **必须同时把 `--max_round_wall_s` 抬到不截断**(有头渲染慢,easy 撞 wall_cap)。**默认标准(2026-06-16 用户定):quick-sweep 的 wall_cap 必须 == strict 的 180,绝不更紧**(90/150 都作废)。quick 与 strict **只差轮数(5 vs 20),不差 wall_cap** → quick 是 strict 的公平快速子采样。**血证:GR00T v9(warm头+解冻视觉)quick @150 = 26.7%,avg 精确=150=wall_cap=全截断**(慢策略每轮都被 150 砍断),而同系 v8(冻视觉,快86s)quick 不截断=真实;紧 wall_cap 系统性冤杀慢策略(=FlowHeads quick-90 截断 flowdp 的同一坑第三次重演)。脚手架 `outputs/gr00t-n17-v9-curve/quick_eval.sh` 已 @180。
- **关键不变量(回答可比性质疑)**:**只要 wall_cap 给够不截断,有头分数 == 无头分数**——分数由 sim 行为(跑满 `episode_length_s` 的 120s sim 内放几颗橙)决定,**不是渲染速度**;有头只是真实墙钟更慢。所以**有头-180 仍与榜上 v1 53.3% / PI_v3 46.7% 的 headless-180 strict 可比,不需要把它们重跑有头**。截断才是唯一会让有头假性掉分的因素 → wall_cap 务必够大。
- 代价:有头每轮真实时间更长 + GUI 渲染占额外 GPU,共享 4090 上与邻居碰撞概率略升(见 [[feedback-pull-eval-decouple-shared-gpu]] 的让位门)。
- 例外:纯自动化大规模 sweep 若 GPU 紧张/无显示,可临时无头,但默认有头。

关联:[[e1-midlayer-sweep-live-state]]、[[feedback-20round-strict-benchmark]]、[[feedback-5round-benchmark-standard]]、[[starvla-gr00t-v2-n17-head]]。
