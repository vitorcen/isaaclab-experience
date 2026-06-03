---
name: lerobot-v040-convert-segfault-fix
description: lerobot-v040 写 LeRobotDataset 多 episode 时随机 SIGSEGV/pandas/sre/datasets 崩 = dual-ffmpeg(PyAV encode + torchcodec decode)堆损坏；修法 = encode 改 ffmpeg-CLI 子进程 + get_task_index 用 .at。另含 Isaac headless recorder teardown 卡死 os._exit 修法
metadata:
  type: project
---

把 numpy 中间格式 episode 批量写成 LeRobot 数据集（`scripts/mimickit_episodes_to_lerobot.py`，
lerobot-v040 env）时，**20+ episode 会随机崩**，且崩点/崩法每次不同：
SIGSEGV(libsvtav1) → pandas `__finalize__` `IndexError: tuple index out of range` →
`sre_compile` `REPEATING_CODES referenced before assignment` → HF `datasets`
`from_arrow_schema` `too many values to unpack`。10 ep 偶尔过，20 ep 必崩。

**根因 = 进程内 dual-ffmpeg 堆损坏。** lerobot v0.4 用 **PyAV(15.1.0) encode** +
**torchcodec(0.5) decode**（save 后算 stats 要读回视频），两者各链一份 ffmpeg；反复
in-process encode 触发 symbol/heap 冲突，把别的库的内存/globals 踩烂 → 崩在哪个库纯看运气。
线程限制(OMP/MKL=1)、换 `video_backend="pyav"`、换 codec 都只是"走更远"不根治。

**修法（已 apply 到 `scripts/mimickit_episodes_to_lerobot.py`，全是 monkey-patch 不改 lerobot 源码）：**
1. **encode 出进程** — patch `lerobot.datasets.lerobot_dataset.encode_video_frames` 改用
   **ffmpeg CLI 子进程**（`/usr/bin/ffmpeg` 6.1.1，libx264/yuv420p/+faststart）。
   输入契约：imgs_dir 下 `frame-%06d.png` 连续帧；调用形态 `(imgs_dir, video_path, fps, overwrite=True)`。
   子进程独立地址空间 → 进程内只剩 torchcodec 一个 ffmpeg **使用者**。这是治本的一招。
2. **get_task_index 绕 pandas 脆弱路径** — `LeRobotDatasetMetadata.get_task_index` 原用
   `self.tasks.loc[task].task_index`（→ `xs` → `__finalize__`，pandas 2.3.3 间歇崩），
   改 `self.tasks.at[task, "task_index"]` 标量访问。
3. shape 必须 tuple 不能 list：v0.4 `validate_frame` 用 `!=` 比 `value.shape`(tuple) vs
   feature["shape"]，list 永不相等 → features dict 里 shape 写 `(D,)`。

**另一个独立坑：Isaac headless recorder teardown 卡死。** MimicKit `mimickit/rollout_record.py`
跑完 headless rollout 后，Isaac Sim app teardown **必卡**（串行 driver 永久 block）。所有 episode
数据在 `np.savez` 时已落盘，所以 `main()` 末尾 `sys.stdout.flush(); os._exit(0)` 强退即可。
没这行：sequential sanity driver 录完 motion-1 就永远停在那、不进 motion-2。

**结果：** fight+dance 各 10ep = 20 episode / 5732 frame / h264 / 6.8MB，round-trip
`LeRobotDataset(...)` 读回 state(95)+action(29)+img(3,224,224) 正常。

**第三个坑：lerobot-v040 训练路径 flaky，ACT sanity 改用 lerobot 0.5.2。**
v0.4.0(torch 2.7.1 editable)跑 `lerobot-train` 时，`torchvision → torch._dynamo` import 阶段
**间歇性解释器级损坏**：`sre_compile` `not enough values to unpack` / `posixpath.join`
`UnboundLocalError: 'b'` / `re` `'str' object is not callable` / SIGSEGV，崩点每次不同。
RAM 48Gi 可用、磁盘 709G、无 OOM → 非环境压力，是这套 torch 2.7.1 import 路径本身不稳。
**lerobot 0.5.2(torch 2.10.0+cu128)干净训成**（loss 11.8→3.5，~50 step/s，num_workers=2 OK）。
→ **go/pivot sanity 用 0.5.2 训 ACT**；[[act-framework-drift-root-cause]] 的 v0.4-locks-ACT-quality
只对最终 benchmark 成立，sanity 阶段"能不能训起来"优先。脚手架 `scripts/mimickit_vla_act_sanity.sh`。

**第四个坑（最关键）：训练时视频 decode 在此环境两种 backend 都段错误。**
0.5.2 训 ACT：`num_workers=4` + torchcodec → step 3780 `DataLoader worker exited`；
`num_workers=0` + pyav → **step 19 段错误**。视频 decode 本身不稳（同 dual-ffmpeg 病根）。
**最终解 = image 数据集**：转换器加 `--no_video`（`dtype="image"` + `use_videos=False`），帧存 png，
训练时**零 ffmpeg**，PIL 加载稳。代价：数据集 6.8MB→170MB、加载慢（image dataset data_s≈0.6s/step，
GPU idle）→ 堆 `num_workers=12` 拉到 ~3.7 step/s。GR00T phase-2 仍要 video 版（mp4）。

**两个 bash 小坑**：① `set -u` + lerobot env 的 binutils activate.d 引用未绑定 `ADDR2LINE` → activate
前后包 `set +u`/`set -u`。② `lerobot-train` 默认想 push hub → 必带 `--policy.push_to_hub=false`。

**进程管理教训**：`lerobot-train` 实际进程名是 `python`，cmdline 含 `lerobot-train`（连字符）；
`pkill -f lerobot_train`（下划线）杀不掉，要按 output_dir 名 `pkill -f act_g1_lafan_sanity` 或 PID。

关联：
- [[mimickit-to-vla-distill-plan]] — 这条转换链是 VLA 蒸馏 sanity 的一环
- [[act-framework-drift-root-cause]] — v0.4-locks-ACT-quality 只对最终 benchmark；sanity 用 0.5.2
