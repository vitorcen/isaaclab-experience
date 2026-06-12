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
- **starVLA = vitorcen/StarVLA fork（2026-06-12 起）**：starVLA 引擎本体仅 ~10M → **不放伞仓**。用户 fork 了 [`vitorcen/StarVLA`](https://github.com/vitorcen/StarVLA)（注意首字母大写 StarVLA），所有本地改动作为 **commit 打进 fork 的 `starVLA_dev` 分支**（基于上游 `e8f8fbb`），LeIsaac/LeSONIC **各自嵌套** `dependencies/starVLA` submodule **都指向这个 fork 同一 commit**（`.gitmodules` url=fork + branch=starVLA_dev）。**不再用本仓 patch 文件**（已删 `patches/starvla/`），`--recursive` clone 即得。6 功能 commit + 1 docs commit（fork 内 `CLAUDE.md`/`AGENTS.md` 写维护准则=只打必要 patch、优先加新文件别改上游 README/降低 pull 冲突）；`.gitignore` 去掉 AGENTS.md 忽略行。SONIC 自研 `QwenPI_CE` CE head 现是 fork 里的**真实 commit 文件**（model/framework/VLM4A/，曾是 untracked 孤儿险些丢失，已根治）。data_registry kit 仍是项目专属、运行时 symlink 部署（不进 fork）。只有 **大引擎**（3×Isaac-GR00T，几 GB 级 + robocasa 共享）才留伞仓。commit SHA 两仓必须一致（fetch-share 对象，否则 user push 一个 fork 分支另一仓 gitlink 指向不存在的 SHA）。
- **跨仓引用铁律**：LeIsaac **只经 `../dependencies` 取共享引擎**（像取数据集/sim，可接受的外部资源依赖）；**绝不调伞仓 `../server/` 等脚本**。`policy_server.sh` 自己内联起 GR00T/lerobot server（`cd ../dependencies/Isaac-GR00T && uv run … run_gr00t_server.py` / conda lerobot），不委托伞仓。伞仓 robocasa 也不调 LeIsaac。→ 两仓各自独立运作，只共享引擎 submodule。
- **submodule init / 资产**：伞仓 `init.sh` 的 `init_submodule()` 负责 `git submodule update --init`（含 LeIsaac）；LeIsaac 资产（HF `leisaac_env`→`LeIsaac/assets`）由 `LeIsaac/init.sh` 自供给。

**坑**：脚本里 ROOT 计算的「层数/语义」per-file 不同 —— 如 `LeIsaac/scripts/finetune/gr00t/watchdog.sh` 的 `REPO_ROOT=$LEISAAC_ROOT/..` 其实是**伞仓**（不是 LeIsaac），改 `../server` 引用前必须先确认 ROOT 指向谁。

关联：[[feedback-refactor-smoke-pathhygiene]]（重构方法论+路径卫生）、[[gr00t-multi-release-env-split]]（3 个 GR00T release 环境分离）、[[feedback-vla-ckpt-best-only-head-rest]]（ckpt 工具在 scripts/ckpt）。
