---
name: feedback-pre-run-gpu-check
description: 每次启动推理 / 训练 / eval 命令之前先 nvidia-smi 检查是否有残留进程占用显存
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

启动任何 GPU 工作负载（policy_inference / lerobot-train / policy_server / GR00T server）**之前**先跑一遍 `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader` + `pgrep -af "policy_inference|isaac|policy_server"`，确认没有残留进程占用显存。

**Why:** SmolVLA eval 卡死案例 — 之前调试 ACT 时 timeout 杀父进程但 Isaac Sim grandchild 没被传信号，残留进程占着 GPU + 文件锁，新 run 启动后 server 处于错位状态（"Can't load processor for SmolVLM2"），表面看是 processor 缺文件，实际是旧 lerobot policy_server 进程状态污染。重启 server + 杀残留 grandchild 后立刻恢复。

**How to apply:**
- 启动前 1 行：`nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; pgrep -af "policy_inference|isaac"`
- 发现 >2 GB 异常占用 / 残留 PID → 先 `kill -9 <pid>` 清干净再启动
- LeRobot policy_server 自己也算"老进程"，**长时间运行的 server 也会状态错位**：调试间隙 server 跑了几小时 + 切换不同 ckpt + 切换 policy_type → 重启 server 比"调通现象"快
- 关联：[[act-eval-debug-roundN]] 里"grandchild 没被 timeout 杀"是反复出现的坑
