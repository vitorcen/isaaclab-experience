# scripts/

伞仓辅助脚本集合（伞仓级通用工具）。

- **GR1 双臂 robocasa tabletop 演示**（GR00T N1.6 base，与 LeIsaac 无关）
- **通用 HF 模型下载器**
- **LeIsaac / GEAR-SONIC submodule patch 维护**

> **PickOrange SO-101 的 eval/benchmark/policy server 已迁入 `LeIsaac/`** —— 见 `LeIsaac/scripts/`（benchmark、policy_server、sweep、ckpt 工具）、`LeIsaac/server/`（SO-101 策略 server）、`LeIsaac/README`。

---

## 通用 HF 模型下载

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

## GEAR-SONIC (GR00T-WholeBodyControl) submodule patch 维护

### `apply_gear_sonic_patches.sh`

把 `patches/gear-sonic/*.patch` 幂等地 apply 到 `dependencies/GR00T-WholeBodyControl/` submodule。
`gear_sonic_setup.sh` 已自动调用（`submodule update --init` 之后）。

```bash
bash LeSONIC/scripts/apply_gear_sonic_patches.sh
```

当前 patch：`0001-download-symlink-into-hf-cache.patch` —— `download_from_hf.py` 把 448MB
checkpoint 从「拷进 repo」改成「symlink 指向 `~/.cache/huggingface/hub`」，避免重复占盘、
保持 submodule 工作树干净。改了 submodule 源码后用
`git -C dependencies/GR00T-WholeBodyControl diff <file> > patches/gear-sonic/<n>-*.patch` 重生成。

---

## 命名约定

- `*.sh` — bash 入口
- 幂等：start 重复调用 = 已起则 skip；stop 重复调用 = 已停则 skip
- 日志统一进 `<repo>/logs/`，PID 文件同名 `.pid`
- 后台进程通过 `setsid + nohup` detach，确保 Jupyter cell 关闭后继续运行
