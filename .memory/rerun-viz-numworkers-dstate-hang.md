---
name: rerun-viz-numworkers-dstate-hang
description: lerobot-dataset-viz 用 num_workers>0 会产生 D 状态孤儿 worker，卡死预览且让 ps/pgrep 一起挂住
metadata:
  type: project
---

`lerobot-dataset-viz`(rerun 预览)**必须 `--num-workers 0`**。2026-08-11 实测:

- 上游 `lerobot/scripts/lerobot_dataset_viz.py` 自己的注释就写了 `num_workers>0` 会
  "hanging in a blocking flush",那个 `gc.collect()` workaround **治不好**。
- 真实后果更狠:dataloader worker 退出时卡死在内核 `exit_mm`(wchan)变成 **D 状态孤儿**
  (`pt_data_worker`,ppid 被 systemd 收养)。D 状态**不可杀**,只有重启能清。
- 连带伤害:**`ps` / `pgrep` 会被这个 D 进程一起拖住不返回** —— 于是"notebook 点停止毫无反应"、
  连想查进程都查不了。表现是喂数据卡在 7/25 不动、cell 中断无效。
- 速度上 `num_workers=0` 不亏:torchcodec 内部就多线程(25 batch ≈ 9s,和 4 worker 一样)。

**判 D 状态孤儿的安全姿势**(此时不能用 ps/pgrep):扫 `/proc/<pid>/comm` + `/proc/<pid>/stat`
第 3 字段;**别读 D 进程的 `/proc/<pid>/cmdline`,会挂住**。`/proc/<pid>/wchan` = `exit_mm`
就是这个病(通常是 GPU/驱动侧释放 mapping 卡住)。

**Why:** 这类"点停止没反应"极易误判成 Jupyter/脚本 bug,实际是内核级不可杀进程污染了整个进程视图。

**How to apply:** 脚手架 `LeIsaac/scripts/dataset/dataset.sh`(`viz` 默认 `NUM_WORKERS=0`,
新增 `stop` 子命令用 /proc 扫描杀 feeder+rerun 查看器);`LeIsaac/Dataset.ipynb` 的 `viz()`
改成 detach 起进程 + 可中断轮询,停止交给独立的 ⏹️ 格 —— 因为喂数据时 Python 阻塞在 rerun 的
Rust gRPC 写里,SIGINT 进不去字节码,而且 rerun 查看器是 spawn 出去的独立进程,杀 cell 带不走。

关联 [[feedback-shared-gpu-eval-queue-orphan-discipline]] [[wallx-env-py310-torch-segfault]]
