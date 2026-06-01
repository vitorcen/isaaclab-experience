---
name: gr00t-n17-hi-space
description: "hi-space/GR00T-N1.7-3B-Pick-Orange 经端到端验证是真 N1.7 (Gr00tN1d7 + Cosmos-Reason2-2B)，h=40 = 2/3 env, 8/9 oranges, avg 104s"
metadata: 
  node_type: memory
  type: project
  originSessionId: f29b6eda-afc7-4618-b533-2c8e72f8ad99
---

# hi-space GR00T-N1.7-3B-Pick-Orange 端到端验证（2026-05-21）

https://huggingface.co/hi-space/GR00T-N1.7-3B-Pick-Orange

## 架构核实结论：✅ 真 N1.7

config.json + experiment_cfg/conf.yaml 显示：

- `architectures: ["Gr00tN1d7"]`
- `model_name: nvidia/Cosmos-Reason2-2B`（≠ N1.6 的 Eagle 2.5）
- `action_horizon: 40`（≠ N1.6 的 16）
- `num_inference_timesteps: 4`
- `use_alternate_vl_dit: true`，`attend_text_every_n_blocks: 2`（N1.7 新结构）
- 3-shard ≈6GB safetensors（N1.6 是 ~3.5GB）

README 训练配方：6000 step, 1 epoch, final loss 0.0301。**README 没披露任何 success rate**。

## Eval 结果（3-round × 120s × 180s wall_cap）

| horizon | env rounds | oranges | avg round s |
|---|---|---|---|
| 16 | 2/3 | 7/9 | 95.6s |
| **40 (recommended)** | **2/3** | **8/9** | **104s** ⭐ |

**vs N1.6 ckpt-6500 self-trained (2/3, 8/9, 115s)**：oranges 持平，N1.7 快 10%。

## 接入坑（写一次）

1. **N1.7 backbone `nvidia/Cosmos-Reason2-2B` 是 gated repo** — 必须先去 HF 网页 Request access，秒批；只 `hf auth whoami` 通过不够（`model_info` 允许任何登录用户，但 `hf_hub_download` 才真测 access）。
2. **Wire 协议 N1.6 → N1.7 改了**：N1.6 用 LeIsaac 自定义 `__ndarray_class__`，N1.7 改用标准 `msgpack_numpy` 的 `{nd: True, ...}` 格式。修法：`LeIsaac/source/leisaac/leisaac/policy/gr00t/serialization.py` 重写为 `mnp.packb/unpackb`，保留 `__ndarray_class__` 反向兼容 decode。
3. **Isaac-GR00T 升级要 transformers 4.51.3 → 4.57.3**（pyproject diff）。复用旧 .venv 不重建，`uv pip install -e dependencies/Isaac-GR00T` 重指向 + `pip install -U transformers==4.57.3 huggingface-hub[cli] opencv-python-headless dm-tree`。
4. **N1.7 release 在 upstream `dependencies/Isaac-GR00T` submodule HEAD**（commit 23ace64「GR00T N1.7 Release」），本地老 fork 落后 23 commits 必须 pull。

## 关联

- [[gr00t-placement-bug-fix]] 这次同步修了 placement bug，本结果是修复后的
- [[xvla-best-inference-cfg]] X-VLA 同套 eval pipeline
