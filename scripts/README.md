# scripts/

辅助脚本集合。两条领域线：

1. **LeIsaac SO-101 PickOrange VLA 推理**（GR00T N1.5 / SmolVLA fine-tune）
2. **GR1 双臂 robocasa tabletop 演示**（GR00T N1.6 base）

加上 LeIsaac submodule patch 维护与通用 HF 下载器。

---

## LeIsaac SO-101 PickOrange

### `download_hf_model.sh REPO_ID`

预热 HuggingFace 默认 cache（`~/.cache/huggingface/hub/`）。等价于 `huggingface-cli download REPO_ID`，原生幂等（重复跑只校验 etag）。

```bash
# GR00T N1.5 ckpt (~7.4 GB)
bash scripts/download_hf_model.sh LightwheelAI/leisaac-pick-orange-v0

# SmolVLA SO-101 PickOrange fine-tune
bash scripts/download_hf_model.sh edge-inference/smolvla-so101-pick-orange
```

下载完之后下游代码直接用 repo_id 引用即可，例如 `AutoModel.from_pretrained("LightwheelAI/leisaac-pick-orange-v0")` 或 `--policy_checkpoint_path edge-inference/smolvla-so101-pick-orange`，HF 内部走 cache。

查看缓存：`huggingface-cli scan-cache`；自定义路径：`export HF_HOME=/your/path`。

### `policy_server.sh {start|stop} {gr00t-n15|lerobot} [MODEL_PATH]`

统一管理 VLA 推理服务，幂等。

| Backend | 端口 | 协议 | 模型加载方 | 默认 MODEL_PATH | 说明 |
| --- | --- | --- | --- | --- | --- |
| `gr00t-n15` | 5555 | ZMQ | **server** | `LightwheelAI/leisaac-pick-orange-v0` | Isaac-GR00T-N1.5 inference_service.py |
| `gr00t-n16` | 5555 | ZMQ | **server** | `hi-space/GR00T-N1.6-3B-Pick-Orange` | Isaac-GR00T run_gr00t_server.py，embodiment_tag=NEW_EMBODIMENT |
| `lerobot`   | 8080 | gRPC | **client** 通过 `--policy_checkpoint_path` 指定 | — | LeRobot async-inference policy_server，client 选 ckpt（SmolVLA / ACT / pi0 …） |

⚠ `gr00t-n15` 和 `gr00t-n16` 共用 :5555，同时只能起一个。

MODEL_PATH 可以是 HF repo_id（走默认 cache）或本地绝对路径。

```bash
# GR00T N1.5 默认 ckpt
bash scripts/policy_server.sh start gr00t-n15
bash scripts/policy_server.sh stop  gr00t-n15

# GR00T N1.6 默认 ckpt
bash scripts/policy_server.sh start gr00t-n16
bash scripts/policy_server.sh stop  gr00t-n16

# 切到其它 ckpt（repo_id 或本地路径都行）
bash scripts/policy_server.sh start gr00t-n15 SomeOrg/some-other-n15-ckpt
bash scripts/policy_server.sh start gr00t-n16 /abs/path/to/local/ckpt

# LeRobot server，client 端通过 --policy_checkpoint_path 指定模型
bash scripts/policy_server.sh start lerobot
bash scripts/policy_server.sh stop  lerobot
```

已验证可用的 client checkpoint（搭 `lerobot` server）：
- `edge-inference/smolvla-so101-pick-orange`（policy_type=`lerobot-smolvla`）
- `shadowHokage/act_policy`（policy_type=`lerobot-act`）

环境变量覆盖（gr00t-n15 专用）：
- `GR00T_N15_DIR` — Isaac-GR00T-N1.5 仓库路径（默认 `../Isaac-GR00T-N1.5`）
- `GR00T_N15_PYTHON` — conda env python（默认 `~/miniconda3/envs/gr00t-n15/bin/python`）
- `GR00T_N15_HOST` / `GR00T_N15_PORT` — bind host/port

日志：`logs/gr00t_n15_server.log` / `logs/lerobot_server.log`，PID 同名 `.pid`。

---

## GR1 robocasa tabletop（GR00T N1.6 base，与 LeIsaac 无关）

- `check_start_gr00t.sh` — 检查并启动 N1.6 robocasa server (`:5555 --use-sim-policy-wrapper`)
- `preview_gr00t_inference.sh` — rollout 1 episode 录 mp4，自动 totem 全屏播放
- `realtime_gr00t_viewer.sh` / `realtime_gr00t_viewer.py` — `mujoco.viewer.launch_passive` 实时窗口

---

## LeIsaac submodule patch 维护

### `apply_leisaac_patches.sh`

把 `patches/leisaac/*.patch` 幂等地 apply 到 `LeIsaac/` submodule。已 apply 会跳过；upstream 移动了会报错。

```bash
bash scripts/apply_leisaac_patches.sh
```

---

## 命名约定

- `*.sh` — bash 入口
- 幂等：start 重复调用 = 已起则 skip；stop 重复调用 = 已停则 skip
- 日志统一进 `<repo>/logs/`，PID 文件同名 `.pid`
- 后台进程通过 `setsid + nohup` detach，确保 Jupyter cell 关闭后继续运行
