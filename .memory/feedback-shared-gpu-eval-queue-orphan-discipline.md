---
name: feedback-shared-gpu-eval-queue-orphan-discipline
description: 共享单 GPU 上多 eval 任务用 flock 文件锁排队(不要各自 N×N 互判);且绝不反复 restart eval runner——broken-bash 下 kill 没生效会堆 orphan 把 load 顶到 100+ 把工作站 wedge 死。探活只读 /proc、kill 用纯数字 PID。
metadata:
  type: feedback
---

# 共享 GPU eval：flock 队列 + 防 orphan 爆炸纪律（2026-06-14 血的教训）

## 标准：一个 flock 锁当队列，别各自互判
共享单 GPU 上多个 Isaac eval 任务(E1 20-round runner、E3a sweep、未来的)**都 `exec 9>/tmp/leisaac_gpu_eval.lock` + 起 serve 前 `flock 9`、跑完 `flock -u 9`**。内核进程死自动放锁(无僵尸锁),FIFO 串行,零 race。比"每个 watcher 各自 pgrep 对方端口/serve"强:后者有 race 窗口、漏判、且 FlowDP 这种不遵守锁的邻居单向撞不住。**gpu_free() 简化为只判非锁邻居(FlowDP)+ 显存**;E1↔E3a 互斥交给 flock。脚手架:`e1_strict_20round.sh` / `e3a_gap_sweep.sh`。

**Why**:两个 Isaac sim 同时加载必 OOM-collide 杀其一;flock 保证同一时刻只一个我的 eval 占卡。

## 🔴 血的教训：别反复 restart eval runner(会堆 orphan 把工作站 wedge 死)
**2026-06-14 事故**:E1 20-round 反复出问题,我在**这会话 bash stdout 抽风(命令大量空/截断)**下连续 kill+restart runner 十几次。broken-bash 下 `pkill`/`ps` 没真生效 → 每次 restart 的 serve_starvla + Isaac + conda 子进程**没被杀干净、累积成 ~400 个 orphan** → 本地工作站(13900KF 32 线程)**load 钉死在 142、734 进程**(正常 ~300)→ 此后**每个新 serve 一起来就被饿死/段错误**,evals 全失败(21k/24k/27k 20-round 0 成功)。orphan 多卡 D 态,`pkill` 在 load 142 下遍历 /proc 自己也挂,清不动 → 最终只能用户**重启本地工作站**(连带杀掉 Claude Code 会话本身 + FlowDP,很粗暴)。

**纪律(防再犯)**:
1. **eval runner 用 flock 单实例**:脚本头 PID 文件守卫 + flock,**确认旧的没在跑/没出数字结果才重启**,绝不盲目连续 restart。
2. **改完脚本要重启 watcher 时**:先**只读**确认旧实例 PID(`ps -eo pid,args|grep X|grep -v grep` 或读 pid 文件),**kill 用纯数字 PID**,确认死透 + serve/Isaac 子进程也清(按 :端口 PID),再起新的。一次到位,别连环重启。
3. **bash stdout 抽风/高 load 时**:探活只用**瞬时只读**——`cat /proc/loadavg`、`test -d /proc/<pid>`、`ls -d /proc/[0-9]*|wc -l`、`nvidia-smi --query-compute-apps`(驱动查询不遍历 /proc)、Read 工具读文件;**别用 `ps -eo|grep` / `pkill -f`**(遍历几百进程会挂)。
4. **pgrep/pkill self-match**:命令行里含模式串(如 `pkill -f e1_strict`)会匹配自身→可能误杀自己的 shell 致命令中途中断;列 PID 只读、kill 纯数字。
5. **load>~50 持续不降 + GPU 已空**:基本是 orphan/D 态,清不动就如实告诉用户、让其从交互终端 `htop`/`pkill` 清(交互终端比沙箱 Bash 响应快得多),**别建议重启**(那会杀会话);重启是用户的最后手段。

关联:[[feedback-pull-eval-decouple-shared-gpu]]、[[feedback-headed-eval-default]]、[[e1-midlayer-sweep-live-state]]、[[starvla-8bit-eval-load-corruption]](serve 大 ckpt load 间歇段错误,要 3×重试)。

## 🚀 长GPU任务启动机制(2026-06-26补): 别 setsid&disown,用 run_in_background
`setsid bash driver.sh & disown` 启长任务**经常启不起来**(无进程/无日志/wrapper exit1)=backgrounded setsid 随工具 shell 退出被回收。**可靠法**:①最稳 `run_in_background:true` 直接跑(阻塞到完成,harness 保活+完成通知);②次选 `nohup ... >log 2>&1 </dev/null &`。pgrep 判进程务必 `grep -v "bash -c"` 滤自身工具命令(假阳性)。本机 py3.10 import-torch 间歇 segfault([[wallx-env-py310-torch-segfault]])→ eval driver 每格重试3次(检"核心已转储/已中止"=crash 重试)。
