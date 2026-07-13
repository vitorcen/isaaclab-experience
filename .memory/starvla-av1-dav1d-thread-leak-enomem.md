---
name: starvla-av1-dav1d-thread-leak-enomem
description: StarVLA 训练用 video_backend=torchvision_av 读 AV1(libdav1d)视频时,VideoReader 不 join 解码线程→累积到 cgroup pids.max(20480)→pthread_create 返回 EAGAIN→FFmpeg 上报 av.error.MemoryError[Errno 12];特征=每次精确同一 step 崩+内存很低(非 OOM);修复=video_backend 换 pyav + 给 pyav 分支 stream.thread_count=1
metadata:
  type: project
---

# StarVLA AV1 视频解码线程泄漏 → 确定性 ENOMEM 崩溃(2026-06-13 实锤)

**现象**:bjb1 box 上训 Qwen3.5-4B(GR00T_v2/E1),DataLoader 每次都在**精确同一 step（382）**崩,
报 `av.error.MemoryError: [Errno 12] Cannot allocate memory`,栈在 `get_frames_by_timestamps` →
`torchvision.io.VideoReader` → `av.codec.context.CodecContext.open`。

**为什么不是真 OOM**:崩时 cgroup `memory.current` 只有 **16.6G / 90G cap**(后来换法只 6G),
内存远没满。抬 `ulimit -n` 到 1048576 也没用 → 不是堆内存、也不是 FD。

**真根因 = 解码线程泄漏触顶 pid/线程上限**:
- 数据是 **AV1（libdav1d）**编码(`codec: libdav1d, pix_fmt yuv420p`),decord 读不了 AV1(`cannot find video stream`),只能 pyav/torchvision。
- libdav1d 每开一个解码上下文派生多个 worker 线程;`torchvision_av` 分支的清理
  (`reader._c=None; reader.container.close()`)**不 join/销毁这些线程**。
- 线程随 open 数累积,**到 cgroup `pids.max=20480`(含线程)时 `pthread_create` 返回 EAGAIN**,
  FFmpeg 把它转成 errno 12(ENOMEM)。
- 完美解释三件事:① 确定性同 step(同 seed→同样本序→同累计 open 数→同线程数触顶);
  ② 内存低(线程占的是 pid 配额/线程栈地址空间,不是 cgroup heap);③ 报 ENOMEM 而非真 OOM。
- 诊断锚:**盯 `cat /sys/fs/cgroup/pids.current` 是否随 step 爬向 pids.max**(泄漏时一路涨;修复后平在 ~1100)。

**修复(两步,已验证越过 382 一路到 650+,pids 平稳 1100)**:
1. `dataloader/gr00t_lerobot/video.py` 的 **pyav 分支**(两处 `stream = container.streams.video[0]` 后)
   加 `stream.thread_count = 1` —— 单线程解码,彻底不派生 dav1d 线程(实测单线程能读 AV1)。
2. config `datasets.vla_data.video_backend: torchvision_av` → **`pyav`**(pyav 分支用 `av.open`+显式
   `container.close()`,正确销毁解码器;配合 thread_count=1 零泄漏)。换 backend 对从头训无害
   (都取最近时间戳帧)。

**复用判据**:StarVLA/lerobot 视频数据集训练若"每次精确同一 step 崩 + 报 Cannot allocate memory + 内存没满" →
先看 `pids.current` 轨迹,基本就是这个;别去查 flash-attn/torch/磁盘。`resume_4b.sh` 这类 resume 脚本
存在 = 当年可能也在靠 resume 顶过周期性崩溃,是同一病。

**第二种表现 = 堆腐蚀段错误(非 ENOMEM;2026-07-10 flowdp 本机实锤)**:同一 AV1 dav1d 多线程解码,在**本机 lerobot `torchcodec`/`torchvision-VideoReader`** 路径下不是触顶 pids,而是**随机堆腐蚀**→症状=**段错误(rc=139)+ 诡异 `'_backward_hooks'`/`'NoneType' object is not callable`/`groups` 报错**(训练中途/推理中途随机崩,**py3.10 和 py3.11 都中招**——排除 py 版本因素)。**通用修 = torchcodec `VideoDecoder(file_handle, seek_mode="approximate", num_ffmpeg_threads=1)`**(`video_utils.py` 的 `VideoDecoderCache.get_decoder`)+ `OMP_NUM_THREADS=1`;实测段错误 22→2/轮、单线程 AV1 解码 8/8 绿。判据:AV1 数据 + 随机段错误/`_backward_hooks` 类怪错(不是 ENOMEM)→ 就是这个,别归咎 torch/py 版本。详见 [[so101-real-demo-recording-and-act]] flowdp 段。

关联:[[lerobot-v040-convert-segfault-fix]](dual-ffmpeg 堆损坏,另一类视频后端坑)、
[[starvla-so101-cloud-training]](av1 用 torchvision_av 的由来 + workers 调参)、
[[feedback-cloud-env-reuse-disk-cleanup]](崩因伪装成 flash-attn/ENOSPC 的同类陷阱)
