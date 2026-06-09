---
name: feedback-refactor-smoke-pathhygiene
description: 大重构(搬迁/拆分)的方法论 — 路径卫生(无本机硬编码)+ 分层冒烟(静态→server起停→全链路1-round)+ per-file谨慎改写不盲sed
metadata:
  type: feedback
---

# 重构方法论：路径卫生 + 分层冒烟（2026-06-09 大重构沉淀）

**Why**：把 PickOrange eval 栈从伞仓搬进 LeIsaac（见 [[umbrella-leisaac-repo-boundary]]）+ 去全仓本机硬编码路径，牵动 60+ 文件。盲改两次踩坑（`leisaac.sh`→误伤 `run_leisaac.sh` 子串；`watchdog.sh` REPO_ROOT 当成 LeIsaac 实为伞仓）。总结出可复用纪律。

**路径卫生（开源面，committed 源码不写本机路径）**：
- 仓库根：`ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/<上N层>" && pwd)"`（.sh）/ `os.path.abspath(os.path.join(os.path.dirname(__file__), …))`（.py）—— **上N层 per-file 算**，别套统一值。
- conda：`$(conda info --base)/envs/X`；HF 缓存：`${HF_HOME:-$HOME/.cache/huggingface}` / `os.environ.get("HF_HOME", expanduser)`；兄弟 repo：`${VAR:-$HOME/work/X}` / `expanduser("~/work/X")`；文档示例：`~` 或 `/path/to/`。
- 配置 tsv/yaml：ckpt 路径用仓库相对（`outputs/...`，consumer 在仓根 cwd 解析）；hydra yaml 用 `${oc.env:HOME}`。
- 运维端点（云主机名/端口）：占位符 + env override（如 `master_pull_watcher.sh` 的 `PULL_BOXES` env），不提交个人 endpoint。
- 密钥：**永不硬编码**，全程 `${VAR:?}`/`$PASS`/`sshpass -e`（实测全仓 0 密码字面量是对的）。

**改写纪律（别盲 sed）**：
- 用带 `(?<!LeIsaac/)` 负向断言 / 精确 old-string，**逐行打印 diff** 复核；`replace("leisaac.sh",…)` 会误伤 `run_leisaac.sh` —— 子串替换前想清边界。
- 每个脚本的 `ROOT`/`REPO_ROOT` 指向谁（伞仓 or LeIsaac）先核实再改其 `../` 引用。
- 改完即 `bash -n`（.sh）/ `py_compile`（.py）；`import os` 别插到 `from __future__` 前。

**分层冒烟（路径重构以"接线对+起得来"为核心，便宜→贵）**：
1. **静态(零风险)**：bash -n 全脚本 + 路径解析存在性断言（每个算出的目标真实存在）+ grep 无残留旧路径 + 无 `LeIsaac/LeIsaac` 双前缀 + notebook JSON 合法。脚手架 `/tmp/smoke_refactor.sh`。
2. **server 生命周期(轻)**：起/停 lerobot server（spare port 如 18080，不载模型）—— 验证内联启动器+自包含。
3. **全链路(重)**：`run_one_strict <slug>` 跑 1-round（GR00T warm load ~85s + sim + episode）。**单 round 是方差不是质量**：GR00T N1.7 榜单 68.3%，单 episode 0/3 与 3/3 都正常 → 要确认"策略健康"得 ≥3 round（实测 `[3,0,1]` 才看出 work+variance）。每次清冒烟产物（results/logs 是 gitignored runtime）。

**判据**：路径重构若全链路能端到端 clean exit + 产出 metrics + 关键路径解析命中，即"接线没破坏"；eval 数值波动归方差，与重构无关。

关联：[[umbrella-leisaac-repo-boundary]]、[[feedback-pre-run-gpu-check]]（起 server 前查显存/端口）、[[feedback-style]]。
