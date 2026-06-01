---
name: hf-upload-tricks
description: hf upload-large-folder 实战坑：--exclude argparse 陷阱、resume cache、branch SHA 验证、worker 并发、mihomo TUN 卡 99% / commit_chunk → reshard 救场
metadata: 
  node_type: memory
  type: reference
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

# `hf upload-large-folder` 实战坑（2026-05-21 session）

发布 `wsagi/GR00T-N1.6-PickOrange` main + 3 branches 时踩到的坑。

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
