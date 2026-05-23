# LeIsaac Headless Eval — AutoDL Cloud Scaffold

云端无头评测 lerobot 系 baselines（SmolVLA / ACT / DP / X-VLA）的复用脚手架。GR00T
家族建议本机评（17 GB ckpt + 4090 24GB OK），云端专注小模型并行加速横评。

> See memory: `feedback-20round-strict-benchmark`, `feedback-5round-benchmark-standard`.
> Local-side leaderboard: `scripts/benchmark/run_all_strict.sh`.

## 推荐 AutoDL 镜像
- **PyTorch 2.7 / CUDA 12.8 / Ubuntu 22.04** — Blackwell-safe，与本机 venv 一致
- 单 RTX 4090 24GB 足够 lerobot baselines (per-process ~16GB peak)
- 数据盘 ≥ 60GB（Isaac Sim 5.1 ~15GB + isaaclab ~2GB + venv ~5GB + ckpts ~3GB + 工作空间）

## 部署流程（4 步）

```bash
# on local
bash scripts/cloud/autodl/headless_eval/scp_bundle.sh <host> <port> <pass>

# on cloud (ssh in)
bash /root/autodl-tmp/isaaclab-experience/scripts/cloud/autodl/headless_eval/01_install.sh
bash /root/autodl-tmp/isaaclab-experience/scripts/cloud/autodl/headless_eval/02_smoke.sh

# run baseline subset (filters to lerobot only)
ONLY=smolvla-self,smolvla-other,act-self,act-other,xvla-self \
    bash /root/autodl-tmp/isaaclab-experience/scripts/benchmark/run_all_strict.sh
```

## 文件清单

| File | Purpose |
|---|---|
| `scp_bundle.sh` | 本机 → 云端 scp 当前 repo（不含 Isaac-GR00T submodule，省 10GB） |
| `01_install.sh` | 装 Python 3.11 + PyTorch 2.7.1+cu128 + isaaclab + LeIsaac + lerobot 0.4.0 |
| `02_smoke.sh` | 跑一个 10s headless Isaac Sim + 1 episode smoke 验证 GUI off + sim 通 |
| `03_run_strict.sh` | 包装 `run_all_strict.sh ONLY=lerobot-only`，含 result scp 回本机 |
| `download_baselines.py` | HF 预下 5 个 baseline ckpt（提前 cache 避免 eval 慢启） |

## 已知坑

1. **Isaac Sim headless** 需 `KIT_HEADLESS=1` + `--/app/window/showStartup=false` + `--no-window`，否则会报 X11 missing
2. **NVIDIA driver ≥ 535** 才支持 cu128，AutoDL 默认 driver 570 OK
3. **首次 isaacsim install** 拉 ~15GB，无 hf_transfer 加速；建议用 `pip install` + 单独的 aliyun pypi mirror
4. **Vulkan + RTX rendering**：4090 OK；vGPU（数据中心改造卡）可能不支持 RT cores，跑 software 渲染会慢 3-5×
5. **ckpts cache 位置** `$HF_HOME/hub` 默认在 `/root/.cache/huggingface`，应改到 `autodl-tmp` (50GB+) — 否则 30GB 系统盘装不下
6. **Server wait timeout** `start_server.sh` 默认 120s，大模型 +cu128 cold load 可能超时；设 `GR00T_SERVER_WAIT_S=600` 提前到 10 min

## 与本机 LeIsaac 关系

云端 eval 是本机评测的 *横向 scaler* — 同一份 baselines.tsv + run_one_strict.sh，只是换台机器跑。
结果 metrics.json scp 回本机 results/benchmark/，本机 aggregate_strict_leaderboard.py 统一出榜。

无任何 cloud-specific 代码 fork — 唯一变化是 install 脚本 + Isaac Sim headless flags。
