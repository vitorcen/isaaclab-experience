---
name: umbrella-leisaac-repo-boundary
description: 伞仓 isaaclab-experience vs LeIsaac submodule 职责边界 — 通用 GR00T 基建在伞仓、PickOrange 专属在 LeIsaac;两仓只共享引擎不互调脚本
metadata:
  type: feedback
---

# 伞仓 vs LeIsaac submodule 职责边界（2026-06-09 重构确立）

**Why**：`isaaclab-experience` 是伞仓（多线：PickOrange/MimicKit/SONIC/RoboCasa）；`LeIsaac` 是它的 git submodule（独立 remote，要开源的 PickOrange VLA benchmark 子项目）。一开始 PickOrange eval 栈散在伞仓 `scripts/`+`server/`，与 MimicKit/launch 等混在一起，且 server/ 整个搬进 LeIsaac 又导致伞仓 robocasa demo 反向依赖 submodule。理清边界后两仓各自独立可跑。

**How to apply（谁放哪 / 谁能引谁）**：
- **通用 GR00T 推理基建 → 伞仓**：引擎 submodule `dependencies/Isaac-GR00T{,-N1.5,-N1.6}`（NVIDIA，跨 PickOrange/RoboCasa/motion 共享，**不能复制进 LeIsaac**）+ 启动器 `server/{start,stop,status,verify,init}_server.sh`。robocasa GR1 demo（`scripts/check_start_gr00t.sh`、`DEMO.ipynb`、`preview_gr00t_inference.sh`）用伞仓 server，**不碰 LeIsaac**。
- **PickOrange SO-101 专属 → LeIsaac**：benchmark harness（`scripts/benchmark/run_one*.sh`+baselines.tsv）、`scripts/policy_server.sh`、SO-101 策略 server（`server/serve_*.sh`、`server/{pi05,openvla,xvla,dreamzero,fastwam}_leisaac/`）、play notebook（`LeIsaac/LeIsaac.ipynb`）、ckpt 工具（`scripts/ckpt/` 见 [[feedback-vla-ckpt-best-only-head-rest]]）、资产下载 `LeIsaac/init.sh`。
- **跨仓引用铁律**：LeIsaac **只经 `../dependencies` 取共享引擎**（像取数据集/sim，可接受的外部资源依赖）；**绝不调伞仓 `../server/` 等脚本**。`policy_server.sh` 自己内联起 GR00T/lerobot server（`cd ../dependencies/Isaac-GR00T && uv run … run_gr00t_server.py` / conda lerobot），不委托伞仓。伞仓 robocasa 也不调 LeIsaac。→ 两仓各自独立运作，只共享引擎 submodule。
- **submodule init / 资产**：伞仓 `init.sh` 的 `init_submodule()` 负责 `git submodule update --init`（含 LeIsaac）；LeIsaac 资产（HF `leisaac_env`→`LeIsaac/assets`）由 `LeIsaac/init.sh` 自供给。

**坑**：脚本里 ROOT 计算的「层数/语义」per-file 不同 —— 如 `LeIsaac/scripts/finetune/gr00t/watchdog.sh` 的 `REPO_ROOT=$LEISAAC_ROOT/..` 其实是**伞仓**（不是 LeIsaac），改 `../server` 引用前必须先确认 ROOT 指向谁。

关联：[[feedback-refactor-smoke-pathhygiene]]（重构方法论+路径卫生）、[[gr00t-multi-release-env-split]]（3 个 GR00T release 环境分离）、[[feedback-vla-ckpt-best-only-head-rest]]（ckpt 工具在 scripts/ckpt）。
