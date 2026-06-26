---
name: hf-upload-tricks
description: hf 上传实战坑。**第一铁律:大文件(≥1GB,尤其代理/TUN下)上传卡死/无流量/卡99% → 立刻 HF_HUB_DISABLE_XET=1 + HF_HUB_ENABLE_HF_TRANSFER=1 走经典LFS,别先试xet(hub≥0.36默认xet会hang)**。另:--exclude argparse陷阱、resume cache、branch SHA验证、逐文件+timeout、ffmpeg转码用/usr/bin/ffmpeg(conda缺libx264)。**数据集发布=LeRobot v2.1 原样上传(它本身已是标准parquet+mp4旁挂+meta/modality.json),别拆扁平parquet(丢modality.json训练管线读不了+丢Viewer);塞预览mp4必加frontmatter configs块**
metadata: 
  node_type: memory
  type: reference
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

# `hf upload-large-folder` 实战坑（2026-05-21 session）

发布 `wsagi/GR00T-N1.6-PickOrange` main + 3 branches 时踩到的坑。

## 0. ⚠️ 硬规则：整个目录 > 几 GB 就别用 `upload-large-folder`，改逐文件 `upload_file` + 重试循环

**2026-06-04 `wsagi/GR00T-N1.7-G1-SONIC-BonesSeed` 12G(3×4.7/4.7/2.5G shard)实测**：在 **mihomo TUN**
（`getent hosts huggingface.co` → FakeIP `198.18.0.x`）下，`upload-large-folder` 和 python
`HfApi.upload_folder` **都反复断连/卡死**，一个 shard 都没落地。**`upload-large-folder` 的批量并发
multipart 在 TUN 下 commit 阶段整条连接掉 → 注定失败**（用户原话"注定失败"）。

**正解 = 逐文件 `upload_file`，每文件独立重试循环**（脚手架 `/tmp/hf_per_file_upload.py`，值得留模板）：
- `os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"`（Rust uploader，掉线自动重连）
- 按文件大小排序**小文件先传**（config/index/statistics 秒过拿稳），大 shard 逐个
- 每文件 `for attempt in range(1,13): try upload_file… except: sleep(min(10*att,60))` —— **每次重试从 xet 断点续**，掉线只是下次接着传，不从头
- 已在 repo 的文件 `list_repo_files` 先 skip
- `ignore`/`EXCLUDE` 训练态文件（optimizer/scheduler/rng_state/trainer_state/training_args/wandb_config）

**判据**：`upload-large-folder` 的优势是省事，但它把整个 dir 当一个 commit batch，**任一文件断连可能拖垮整批**；逐文件每个是独立可续的小操作，**单点失败只影响单文件**。dir > ~几 GB（尤其有 >1.5GB shard + 走代理）一律走逐文件。

## 0.05 ⚠️🔥 "卡 99%" 头号假象 = 自己手动杀进程造成同文件并发对传（2026-06-08 实测，最易复发）

**`wsagi/GR00T-N1.7-G1-SONIC-BonesSeed` bf16 重传(3×2.49/2.49/1.31G)实测**：3 个 shard 同样大小，shard1 单进程一次过，shard2 **稳定卡 92-97% 冻死、:443 堆到 140-150**——看着就是 §5.1 "TUN 诅咒卡 99%"。**真因不是文件大小，是我自己**：用 §4 `timeout N hf upload` 重试循环时**手动 `pkill` 掉挂着的 inner hf 进程** → `timeout` wrapper 没死、循环立刻 respawn 新尝试，**而被杀那次的 100+ :443 socket 还没回收** → **两个进程同时 PUT 同一个 shard，互相卡死在 ~95%**。

**铁律**：
1. **绝不在 `timeout`+retry 循环还活着时手动杀 inner 进程**——要么杀整个循环(`pkill -f <脚本名>` + 所有 `hf upload`)，要么别碰。半路杀 inner = 制造并发。
2. **最稳上传形态 = 单进程、一次一个文件、全程 hands-off**：`HF_HUB_ENABLE_HF_TRANSFER=1 timeout 400 hf upload <repo> <file> <file>` 作为**一个**后台任务，让它自然跑完→`repo_info` 验大小→再下一个。同一个"卡死"的 2.49G shard 这样**一次过**。retry 循环看似省事，但叠加手误杀进程 = 并发地狱，单进程串行最不容易出错。
3. **判 stall 真伪**：字节进度冻结 + `:443` 爬到 140+ 且 `pgrep -af "hf upload"` **看到同一文件 ≥2 个 python 进程** = 自造并发，杀光重来；只有**单进程**且字节冻在 ~99% 才是真 §5.1 诅咒。
4. 文件间留 `sleep`，等 `:443` 排空到 <25 再传下一个，避免 §0.3 饱和导致下次 `/api/repos/create` preflight ConnectTimeout。

## 0.06 ⚠️ `HF_HUB_DISABLE_XET=1` 不是万能药——在能正常 xet 传输的 TUN 下它卡 0%（2026-06-08）

同 session：xet 默认路径**字节正常流动**(folder 模式 226MB/s、单 shard ~200MB/s)。但加 `HF_HUB_DISABLE_XET=1` 走标准 LFS → **开了 ~100 条 :443 但 120s 零字节**，死卡在 `0/1 [00:00]` 的 LFS-batch 协商阶段，一个 byte 都不传。**结论：DISABLE_XET 是 §0.2 那种"xet 传完 100% 但 commit_chunk 挂死"的专用解，不是通用解**；当 xet 本身在正常传字节时**别禁它**——禁了反而卡死 LFS batch 起步。判据：先看 xet 默认能不能流字节，能流就别动 xet；只有"字节到 100% 但 finalize 永挂"才上 DISABLE_XET。

**✅ 2026-06-06 `wsagi/StarVLA-PickOrange` 9.98GB 单 monolith `.pt` 复现 + 关键洞察**：`upload-large-folder` commit 阶段挂死 450s（io 全 idle，repo 只落地 .gitattributes）= §4 curse。杀掉→清 `.cache`→改逐文件 `timeout 360 hf upload <repo> <file> <file>` 循环 → 小文件全过 + **9.98GB monolith attempt-1 60s 落地**。**关键：问题在"批量 commit"不在"文件大小"** —— 同一 9.98GB 文件（5GB 标准 2×、StarVLA `.pt` 单文件 checkpoint 无法 safetensors-reshard）用**单文件 `hf upload` 独立 commit 一次就过**。所以 **monolith 先试单文件 `hf upload`（hf_transfer + timeout 包裹），多半直接过，reshard 是单文件也挂时才用的下策**。脚手架 `/tmp/hf_perfile_upload.sh`（小文件 list 先传 + 大文件 `for a in seq 1 15: timeout 360 hf upload && break` + `${PIPESTATUS[0]}` 判 rc）值得留模板。

**坑**：`upload-large-folder` 跑过会在目录里留 `.cache/huggingface/upload/`（resume 元数据 + `.lock`）。逐文件脚本 `os.walk` 必须 `dirs[:] = [d for d in dirs if d != ".cache"]` 跳过它，否则会试图上传 `.cache/…safetensors.lock` → HF 报 `cannot update files under a '.cache/' folder`。换方案前先 `rm -rf <ckpt>/.cache`。

**🔑 关键修正：逐文件 retry 循环还不够，每次尝试必须套超时。** 99% 诅咒是 **hung 不是 exception**：chunk 传到 100% 但 `commit_chunk` RPC 永久挂住，`upload_file` **不返回也不抛异常** → `except` 永不触发，retry 永不启动（用户连说两次"又断了"其实是 hung）。修：bash `timeout` 包住每次尝试，杀掉挂死的调用再重试（xet dedup 已传 chunk，重试秒续）。**实测 12G/4.7G-shard：两个 4.7G shard 在无超时时卡 99%，加 `timeout 240` 后各 attempt-1 落地。** 脚手架 `/tmp/hf_resume_shards.sh`：
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
for shard in model-*.safetensors; do
  for attempt in $(seq 1 10); do
    timeout 240 hf upload <repo> "$shard" "$shard" --repo-type=model && break
    echo "[retry $attempt] hung/failed, resuming..."; sleep 5
  done
done
```
（纯 python 要 `multiprocessing`/`signal.alarm` 限时每个 `upload_file`；bash `timeout hf upload <repo> <file> <file>` 循环更省事，kill 也可靠。）

## 0.07 ⚠️ XET delta 上传(替换已存在相似 blob)会**中途**卡 + 并发 puller 抢带宽是真凶(2026-06-09)

`wsagi/DiffusionPolicy-PickOrange` 用 4ep ckpt(1.07GB model.safetensors)**替换**旧 70k ckpt(相似 blob)：xet 检测到旧 blob → 走 **delta 只传差异**("New Data Upload 530MB")→ **死卡 ~869MB/81%**（注意：**不是 §0.2 的 finalize 100% 挂，也不是 §0.06 的 0% LFS-batch 挂，而是传输中途冻结**）。单文件 `hf upload`、`upload-large-folder`、hf_transfer 0/1 **全卡同一点 ~869MB**。
**两个新教训**：
1. **真凶之一 = 并发 puller 抢家宽上传带宽**。杀掉卡死的上传后 `tx` 仍 ~1.1MB/s → 是 `master_pull_watcher` 在**循环重拉**(已在本地的 head 反复 pull，rsync/ssh 占上传方向)。**§0.3 的反向**：那里是 hf 上传饿死 SSH，这里是 puller 饿死 hf 上传。**大上传前先 `kill` 所有并发 rsync/ssh/puller**，腾出家宽上行。
2. **替换已存在 blob 的 delta 上传，小到 1GB 也会中途卡**（不只 §0.2 的 17-18G finalize）→ 同样上 `HF_HUB_DISABLE_XET=1` 走**全量 LFS**（不走 delta），实测从死卡 → 7-10MB/s 稳传到 100% + 秒 commit。
**判据**：上传字节冻在中途(非 0% 非 100%) + 多方法卡同点 + 杀上传后 tx 仍高 → 先停并发传输，再 `HF_HUB_DISABLE_XET=1` 全量重传。注意此案非 mihomo TUN 场景也复现，§0.06"xet 正常流字节就别禁"指的是**全量 xet 在流**；**delta-replace 卡中途时该禁**。

> **🆕 2026-06-14 `wsagi/FlowHeads-DiffusionPolicy-PickOrange` 新 repo(非 delta) 1.07GB 单 safetensors → 铁律收紧**：mihomo TUN 下 `upload_folder`(批量+xet) 和单文件 `hf upload`(xet) **都传到 88%(934MB)中途冻结**(≈§0.07 的 869MB 同点;:443 仅 10、io idle = hang)。**全新 repo 无旧 blob,排除 delta-replace → 证明纯 xet 全量大单文件在 TUN 下也会中途挂**。`HF_HUB_DISABLE_XET=1 + hf_transfer + timeout 1800 hf upload`(标准 LFS) **attempt-1 秒落地**。**收紧:TUN 下 ≥1GB 单文件直接 DISABLE_XET,别先试 xet**(§0.06"xet 流就别禁"只适用小文件/数据集 mp4 预览)。最稳形态=**小文件+README+mp4 用 `upload_folder --exclude 大文件` 一笔小 commit 先落,大文件单独 DISABLE_XET 走**。

> **🆕 2026-06-17 `wsagi/GR00T-N1.7-V2-PickOrange` 发布(v10-4500,2×5G shard)再次实证 + recall 失败教训**:连续 4 次尝试全卡(`api.upload_folder`+xet 卡97% finalize / `upload-large-folder`+xet 早期卡 / 自以为关了 hf_transfer 的"standard"其实仍被 xet 接管也卡 / 全程误以为是"家宽 5MB/s 上行瓶颈")。**真因 = XET(hub 0.36 默认),与网络无关——`HF_HUB_DISABLE_XET=1` 一禁立刻 38MB/s 经典 LFS attempt-1 落地**。**教训:这条铁律本文档早有(§0.2/§0.06/06-14),但上传前没 recall→白绕半小时。≥1GB 上传第一步就 DISABLE_XET,别等卡了再想起**。发布流程脚手架(可复用):①`create_branch` 归档旧 main 到分支 ②`HF_HUB_DISABLE_XET=1 hf upload-large-folder`(排训练态)传模型 ③`upload_file` 传 README+mp4 ④`delete_file` 删旧图/漏网 trainer_state/training_args ⑤`HfApi.move_repo(from_id,to_id)` 改名(分支随迁,旧URL重定向)。

## 0.2 ✅ `HF_HUB_DISABLE_XET=1` = 单大文件 commit-hang 的首选干净修法（2026-06-07 实测）

`wsagi/StarVLA-Qwen3-VL-8B-PickOrange` 17.9G 单 `.pt`：**默认（xet 开）+ hf_transfer 上传，
字节传完到 100% 但卡死在最后 commit**——`ss -tnp` 看到对 commit 端点 `CLOSE-WAIT`（服务器已
FIN、客户端没回收）、进程所有线程 `futex_do_wait`、进度条还在 100% 重绘骗人“像在动”。重传后
xet dedup 秒到 100% 又卡同一处。**根因 = mihomo TUN 代理把耗时的 xet `commit_chunk` 长连接掐断**
（小文件 commit 快没事，§5.1 同源）。

**修法（比 §5.1 reshard / §4 timeout-loop 都简单）= 禁用 xet 走普通 LFS：**
```bash
pkill -9 -f "hf upload <repo-substr>"
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 hf upload <repo> <file> <path-in-repo> --repo-type model
```
普通 LFS 多段（业界 5GB 分段）传输**流量可见**、commit 是快速 pointer 注册（和小文件/mp4 同一条
能成的路径），**一次落地**。判据：传完字节后若 repo `list_repo_files` 长时间不出现该文件 + 进程
线程全 futex_wait + CLOSE-WAIT → 是 xet-commit 挂死，别等，直接 `HF_HUB_DISABLE_XET=1` 重传。

> **2026-06-08 复现 + 补充**:`wsagi/StarVLA-Qwen3-VL-8B-PI_v3` 18.9G 单 .pt 用 **python `HfApi.upload_file`(默认 xet 开)反复卡 18.7-18.8/18.9G finalize**(hf_transfer 和标准都卡)——同 §0.2 xet-commit 挂死。`upload_large_folder` 分片最终落地了但进度条 99% 假卡误判。**铁律:任何大单文件上传(CLI 或 python API)都先 `HF_HUB_DISABLE_XET=1`**。

**monolith 上传顺序更新**：①先 `HF_HUB_DISABLE_XET=1`（最稳）→ ②不行再 §0 逐文件+timeout → ③最后才 §5.1 reshard。

## 0.3 ⚠️ 大 hf_transfer 上传会饿死同代理的 cloud SSH（2026-06-07 实测）

hf_transfer 多段上传开 **129-142 条并行 :443 连接**，全走本机 mihomo 代理（FakeIP 198.18.0.x）。
**同时 ssh 到 AutoDL 云机（也走这代理）会被饿死，~全部失败/超时**——表现为 `sshpass ssh` 连环
`Connection timed out / closed`，看着像云机挂了，其实是**本机代理连接表被上传打满**。上传一结束
（`ss -tn|grep -c :443` 掉回 ~10）SSH 立即恢复。**教训**：大文件 upload 与 cloud-SSH 操作**别真并行**，
要么等 upload 完再 SSH，要么 SSH 命令保持极短 + 重试到偶然成功一次（launch 这类一次性操作可接受）。
另：短时间几十次快速 SSH 重试可能触发云机 sshd MaxStartups 限流，雪上加霜——重试间隔给够 5s+。

## 1. `--exclude` 是单 flag 多 value，**不是**重复 flag

```bash
# ❌ 错（只生效最后一个 exclude）
hf upload-large-folder repo . --exclude "optimizer.pt" --exclude "scheduler.pt" --exclude "rng_state.pth"

# ✅ 对（一个 --exclude 跟多个 pattern）
hf upload-large-folder repo . --exclude "optimizer.pt" "scheduler.pt" "rng_state.pth" "optimizer.pt.bak"
```

**Why:** `--help` 显示 `--exclude [EXCLUDE ...]`，argparse 的 `nargs="*"` 语义 — 后写的 `--exclude` 会覆盖前面整个列表。  
**如何 catch:** 看 `.cache/huggingface/upload/<file>.metadata` 是不是有 `optimizer.pt.metadata` 仍在被处理。

## 2. Resumable via `.cache/huggingface/upload/`

每个被上传的文件有：
- `<file>.metadata` — 行格式：`mtime / size / next_offset / sha_full / lfs|small / sha_hex / committed_bit / next_step`
- `<file>.lock` — 0 字节占位
- `experiment_cfg/` 等子目录同样有 sub-cache

杀进程 → 重启 `hf upload-large-folder ...` → 跳过 `committed_bit=1` 的文件，从断点续 `committed_bit=0` 的。**xet content-addressed dedup**：同 sha 的 chunk 整个仓库只上一次。

清空续传：`rm -rf <local_dir>/.cache/huggingface/upload/`。

## 3. Branches 验证：commit SHA

```python
from huggingface_hub import HfApi
api = HfApi()
for b in ["main", "ckpt-3500", "ckpt-5000"]:
    print(b, api.repo_info("wsagi/foo", revision=b).sha[:12])
```

**`create_branch(branch=...)` 后所有 branches 初始 SHA = main 的 SHA**（fork point）。upload 完成后该 branch 才会有新 SHA。如果 4 个 branches SHA 完全相同 → 还没 commit，文件可能还在传或卡了。

`api.repo_info(revision=b).siblings` 的文件数同 main（因为 branch 创建时 inherit），**不能用 file count 判断是否上传完**，必须看 SHA。

## 4. 并发与抑制 tqdm

- `--num-workers=8` 默认；`--num-workers=16` 对大 LFS shard 有提升（每个 worker 一个文件并发上传）
- `--no-bars` 抑制 tqdm 进度条 — 后台任务 / `tail -F | grep` 监控时不会被 bars 填屏
- LFS 单文件上限是 HF 的 `~10 GB` shard cap，更大要用 `safetensors` shard pattern

## 4.1 **必装 `hf_transfer` (Rust 并发 uploader) — 5-10× 提速**

```bash
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload-large-folder repo .  # 启用
```

**症状：不装的话默认 python `requests` 单连接上传**，遇到 HF Hub 限速会掉到几十 KB/s，且会留下 `CLOSE-WAIT` 半关闭连接（远端已 FIN 但本地没回收 socket）：

```bash
ss -tnp | grep "hf"
# 看到 CLOSE-WAIT 状态 + `/proc/<pid>/io` 写字节 5s 内不动 = 上传卡死，不是慢
```

**修法**：`pkill -9 -f "hf upload" → 装 hf_transfer → restart with HF_HUB_ENABLE_HF_TRANSFER=1`。Rust 多线程 chunk upload + 自动重连，遇到 HF 限速也能保持 MB/s 级别。

`HF_HUB_ENABLE_HF_TRANSFER=1` 但没装包 → 警告 fallback 到默认（不会报错），所以 install 是前提。

## 5. Branch 上传命令

```bash
# Create branches first
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
for s in [3500, 5000, 7000]:
    api.create_branch('wsagi/foo', repo_type='model', branch=f'ckpt-{s}', exist_ok=True)
"

# Then upload each ckpt dir to its branch
for step in 3500 5000 7000; do
  cd $BASE/checkpoint-${step}
  hf upload-large-folder wsagi/foo . --repo-type=model \
    --revision="ckpt-${step}" --num-workers=16 --no-bars \
    --exclude "optimizer.pt" "scheduler.pt" "rng_state.pth"
done
```

## 5.1 ✅ mihomo TUN → Hysteria 2 server 调优修复 LFS 大文件卡 99% (2026-05-24)

### TL;DR

`wsagi/Pi0.5-PickOrange` 9.35GB single safetensors 上传卡 99% / 9.22GB。**根因不在 HF，也不在 client mihomo，而在 Hysteria 2 server 配置过简**（v2.9.2，141.11.122.195）。原 config 只有 `listen + tls + auth + resolver + outbounds`，**未配 `bandwidth` / `quic`** → default `initStreamReceiveWindow=8MB` 不够，没启用 Brutal 拥塞 → 多 stream multipart upload 在 commit_chunk RPC 阶段 race。

修复（追加到 `/etc/hysteria/server.yaml` + `systemctl restart hysteria-server`）：

```yaml
bandwidth:
  up: 500 mbps       # match your server's actual uplink (per-server!)
  down: 500 mbps
quic:
  initStreamReceiveWindow: 16777216      # 16MB (default 8MB)
  maxStreamReceiveWindow: 16777216
  initConnReceiveWindow: 33554432        # 32MB (default 20MB)
  maxConnReceiveWindow: 33554432
  maxIdleTimeout: 60s
  keepAlivePeriod: 20s
udpIdleTimeout: 120s                      # default 60s; HF commit_chunk follow-up needs room
```

**验证 before/after**：
- before：9.35GB monolith 3 轮全卡 99%；11×900MB 还要 attempt 2 才过
- after：**2×5GB attempt 1 SUCCESS 36 秒**（throughput ~700-800 MB/s peak）

### 业界 shard 标准 = 5GB

- `transformers.save_pretrained` / `hf_hub.split_torch_state_dict_into_shards` default = **5GB**
- `accelerate.Accelerator.save_model` default = 10GB
- 真实 LLM repos: Llama-3.1-8B 4×5GB / Mistral-7B 3×5GB / Qwen2.5-7B 4×3.95GB / Gemma-2-9B 8×5GB — **全部 max ≤ 5GB**
- 教训：**先修代理 / 换出口，再按 5GB 标准切片**。把 shard 切到 < 1GB 绕代理 bug 是 emergency workaround 而非 release 实践，repo 给别人看一堆碎片不专业。

### 诊断 mihomo / hysteria 系统的方法

```bash
# 1. 看 client 端是否走 TUN
ip route get 198.18.0.21              # 看 dev Meta table 2022 = TUN
pgrep -af clash mihomo                # FakeIP 198.18.0.0/15 → TUN proxy

# 2. SSH 到 hysteria server 看 config 是否过简
cat /etc/hysteria/server.yaml         # 缺 bandwidth/quic 是常见过简
systemctl status hysteria-server
journalctl -u hysteria-server -n 60   # default log level info 不打 connection lifecycle

# 3. 排除 server 出口问题
curl -s -o /dev/null -w "%{time_total}s\n" https://huggingface.co/api/whoami-v2   # 在 server 上跑
# 如果 < 0.5s → 出口没问题，瓶颈在 client→server hysteria QUIC

# 4. 改完 restart 验证
systemctl restart hysteria-server && systemctl status hysteria-server
ss -ulnp | grep 38443                 # 新 PID listening
```

### 重传 + cleanup 旧 shard 标准流程

```python
# 1. local reshard
from huggingface_hub import split_torch_state_dict_into_shards
plan = split_torch_state_dict_into_shards(state_dict,
    filename_pattern="model{suffix}.safetensors",
    max_shard_size="5GB")  # industry standard

# 2. upload new shards (overwrites same-name; doesn't delete old-name)
hf upload-large-folder repo .

# 3. delete obsolete differently-named shards in single commit
from huggingface_hub import HfApi
HfApi().delete_files(
    repo_id="wsagi/foo",
    delete_patterns=[f"model-{i:05d}-of-00011.safetensors" for i in range(1, 12)],
    commit_message="Remove obsolete 11×900MB shards",
)
```

### 这次踩坑过程（已 resolved，留档）

### 业界 shard 标准 = 5GB，不要因代理 bug 切碎

- `transformers.save_pretrained` / `hf_hub.split_torch_state_dict_into_shards` default = **5GB**
- `accelerate.Accelerator.save_model` default = 10GB
- 真实 LLM repos: Llama-3.1-8B 4×5GB / Mistral-7B 3×5GB / Qwen2.5-7B 4×3.95GB / Gemma-2-9B 8×5GB — **全部 max ≤ 5GB**

**别因为本机代理扛不住就把 shard 切到 < 1GB** — 这是 emergency workaround，不是 release 标准。Repo 给别人 clone 时一堆碎片显得不专业，`index.json` 也膨胀。正确顺序：**先修代理 / 换出口，再按 5GB 标准切片**。

### 这次踩坑过程（2026-05-24 pi05 9.35GB 上传）

发布 `wsagi/Pi0.5-PickOrange` 时踩到。**症状**：

- `model.safetensors` 9.35 GB 单 monolith，上传必卡 **99% / 9.22GB**（最后 130MB）
- log 显示 `pre-uploaded: 2/3 (15.1K/9.4G)` 永不变，但单文件 progress bar 跑到 99% 然后死
- ESTAB socket `send-q=0` 长时间不动（不是 CLOSE-WAIT，是 idle）
- 8 次 retry / `--num-workers=1` / `HF_HUB_ENABLE_HF_TRANSFER=1` / `STALL_WINDOW=420s` 全救不了
- 即使 reshard 到 6×~2GB，**任何 > 1GB 的 shard 仍卡 ~98%**（shard 1 1.62GB → 卡 1.58GB）
- 终于 reshard 到 **11×~900MB** (max_shard_size="900MB") attempt 2 ✅ 47s 内 SUCCESS

**根因**：本地 `clash-verge-service` + `verge-mihomo` TUN mode (FakeIP 198.18.0.0/15) 对长 HF LFS multipart upload 的 `commit_chunk` RPC 默默失败 — chunks PUT 完成但客户端 commit_chunk 不通过，HF 端 chunk 变 orphan，下次 retry 重传同 chunk。

```bash
# 诊断 mihomo TUN
ip route get <hf-ip>   # 看是不是 dev Meta table 2022
pgrep -af clash mihomo
getent hosts huggingface.co  # 198.18.0.x → FakeIP
```

**WHY < 1GB 能过？** 单 shard 在一次连续 PUT 内完成（< CDN edge keepalive timeout / TUN session limit），不触发 commit_chunk RPC 失败。

**复现配方** — reshard 任何 > 1GB safetensors monolith：

```python
# /tmp/reshard.py
import safetensors.torch, json, shutil
from pathlib import Path
from huggingface_hub import split_torch_state_dict_into_shards

CKPT_DIR = Path("...")
SRC = CKPT_DIR / "model.safetensors"
state_dict = safetensors.torch.load_file(str(SRC), device="cpu")
plan = split_torch_state_dict_into_shards(
    state_dict, filename_pattern="model{suffix}.safetensors",
    max_shard_size="900MB",  # < 1GB to dodge mihomo TUN curse
)
filename_to_tensors = {}
for name, fn in plan.tensor_to_filename.items():
    filename_to_tensors.setdefault(fn, {})[name] = state_dict[name]
for fn, tensors in filename_to_tensors.items():
    safetensors.torch.save_file(tensors, str(CKPT_DIR/fn), metadata={"format":"pt"})
if plan.is_sharded:
    (CKPT_DIR/"model.safetensors.index.json").write_text(json.dumps(
        {"metadata":{"format":"pt"}, "weight_map":plan.tensor_to_filename}, indent=2))
shutil.move(str(SRC), str(CKPT_DIR/"model.safetensors.original"))  # 备份
```

注意 **`metadata={"format":"pt"}`** — `plan.metadata` 含 int (total_size) 触发 `safetensors` PyString 类型错；写空 dict 也不行（lerobot/transformers load 时会找 `format` 字段）。

**单 tensor > 900MB 强制 1 shard 的例外**：pi05 的 2 个 emb matrix 各 1.05GB，`split_torch_state_dict_into_shards` 无法切分单 tensor，shard 2/3 = 1.05GB。实测 1.05GB **过得去**（mihomo curse 边界在 ~1.5GB+），所以 11×~900MB 方案虽然有 2 个 > 1GB 的 shard 也成功。但若都 < 1GB 更稳。

### Phase-aware STALL detection 脚本

`/tmp/pi05_upload_loop.sh`（值得保留为模板）：
- 双阈值：`STALL_WINDOW_UPLOAD_S=90` / `STALL_WINDOW_COMMIT_S=300`，按 `rchar ≥ COMMIT_THRESHOLD` 切
- 监 python child PID 的 `/proc/<pid>/io rchar` 增长，连续 idle 触发 kill + retry
- 注意：**自动 kill 可能误杀 commit phase**（commit 8/8 单文件可能比 5min 长），需要把 success 判据改为"process exit code = 0"（脚本最新版已改）

### GR00T-N1.6 为什么"不需要 reshard"？

不是 HF 端策略不同 — 是 **HF Trainer 默认 `max_shard_size=5GB` 保存时已自动拆**：
- `wsagi/GR00T-N1.6-PickOrange` = `model-00001-of-00002.safetensors` 5.0GB + `model-00002-of-00002.safetensors` 4.4GB（训练时就是 sharded）
- `wsagi/Pi0.5-PickOrange` = lerobot/openpi trainer 不自动 shard，single `model.safetensors` 9.35GB → 中招

**Takeaway**：训练框架若是 transformers Trainer / accelerate save_model，shard 已 5GB 默认；若是自定义 trainer (lerobot pi05 / openpi) 吐 monolith，发布前用 `split_torch_state_dict_into_shards(max_shard_size="5GB")` reshard 到业界标准 5GB，**不是 900MB**。900MB 只在代理彻底救不回来时用，且要 README 注明原因。

## 6. 关联

- [[xvla-best-inference-cfg]] X-VLA HF upload 历史踩坑
- skill `hf-publish-model` 在 `~/.claude/skills/hf-publish-model/SKILL.md`（这次没用，但相关）

**🆕 2026-06-10 `wsagi/GR00T-N1.7-G1-SONIC-LAFAN` 3-shard bf16:timeout 不是越短越好,要看路径**。本机 TUN 代理下 hf_transfer 传完字节后 finalize 连接被代理掐成 CLOSE-WAIT(无流量假死)→ **关 hf_transfer 用标准上传器**。但标准路径的 **LFS commit 实测要 ~15min**(2.49G shard;shard2 成功正好 ~15min)→ 之前"timeout 240-400s"经验是 hf_transfer/xet 路径的,**对标准上传器是自伤**:timeout 900s 每轮在 commit 将成时把自己杀掉,无限循环 98%。修:**标准上传器大 shard timeout≥1800s** + 重试循环(resume cache 让重试字节秒传只剩 commit);起步偶发 SSL UNEXPECTED_EOF 秒败=瞬时抖动,重试就过。判别:进度条卡 98-100% 且连接 ESTAB+进程有 CPU=正常 commit 等着别杀;CLOSE-WAIT=死了等 timeout。

**🆕 2026-06-10 LeRobot 数据集加 README 预览视频必踩坑:Dataset Viewer 按"扩展名多数"猜 builder**。给 `wsagi/SONIC-VLA-LAFAN` 加 `media/*.mp4`(4个预览)后,mp4 总数(8 ego+4 media=12)> parquet(8) → Viewer 误判成 video 数据集只显示 12 rows;BonesSeed 当时 7:7 平手侥幸判 parquet 显示 3.82k。**修=README frontmatter 显式声明**:`configs: [{config_name: default, data_files: [{split: train, path: "data/chunk-000/*.parquet"}]}]`。规则:**凡往 LeRobot 数据集仓库塞规范外的 mp4,必须同时声明 configs**。

**🆕 2026-06-26 发布数据集 = 上传标准 Parquet,但「LeRobot v2.1 本身已是标准 Parquet」别再拆扁平**:`wsagi/SONIC-VLA-BonesSeed-V2` 发布时用户问"直接打包标准 parquet 是不是更好"——**伪命题**。LeRobot v2.1 底层 = `data/chunk-*/*.parquet`(标准 Apache Parquet) + `videos/.../*.mp4`(图像不内联,旁挂 mp4,parquet 存帧索引) + `meta/*.json(l)`(schema 旁车,含 GR00T 训练必需的 `modality.json` 把 state/action 切模态组)。**拆成扁平/单一 parquet = 净损失**:① 视频内联→文件爆炸;② 丢 `modality.json`→GR00T 数据管线加载不了;③ 丢 HF Dataset Viewer。**铁律:LeRobot 数据集就原样上传(它就是 parquet)+ frontmatter 加 configs 块声明 `data/chunk-000/*.parquet`(见下条),不要为"标准化"重打包。** 只有自定义/非 LeRobot 表格数据才需显式转 parquet 再传。

**🆕 2026-06-12 BonesSeed 复现 + 确认规则(那个"侥幸"靠不住)**:给 `wsagi/SONIC-VLA-BonesSeed` 加 7 个 `videos/BonesSeed-*.mp4` 预览(README `<video src=.../resolve/main/videos/...mp4 controls>` 嵌入)后,mp4 变 14(7 ego+7 media) > parquet(7) → Viewer 失灵,正是 06-10 那条预言的"平手不再"。同一个 configs 块(`data/chunk-000/*.parquet`)修好。**铁律固化**:LeRobot 数据集只要往里塞任何规范外 mp4(预览/封面),**当场就把 configs 块写进 frontmatter**,别赌 ego:media 数量平手。另:HF 视频内嵌确认可用 `<video controls src="https://huggingface.co/datasets/<repo>/resolve/main/<path>.mp4">`(full-path resolve URL),实测 dataset card 能渲染播放;转码用系统 `/usr/bin/ffmpeg`(conda 自带的 libx264.so.138 常缺)`-c:v libx264 -pix_fmt yuv420p -vf scale=trunc(iw/2)*2:trunc(ih/2)*2 -movflags +faststart`。
