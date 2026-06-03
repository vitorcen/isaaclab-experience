---
name: gear-sonic-preview-setup
description: GEAR-SONIC (GR00T-WholeBodyControl) G1 预览跑通配方 — 走 Isaac-eval 单进程路线 + isaaclab env 6 块依赖 + trl==0.28.0
metadata:
  type: project
---

NVIDIA GEAR-SONIC（`dependencies/GR00T-WholeBodyControl`，SONIC whole-body G1 控制器）
**看动作就用单进程 Isaac-eval 路线**，2026-06-03 跑通：G1 被 SONIC 策略驱动行走
（sample motion `walk_forward_amateur_001`，40s/2002 帧），GPU ~9.5G/35%。

**两条预览路线**：
- `scripts/gear_sonic_preview.sh` ← **默认，能跑**。`eval_agent_trl.py +checkpoint=sonic_release/last.pt
  +headless=False ++num_envs=1 ...`，在 `isaaclab` conda env 跑，**无 DDS、无 C++ 二进制**。
- `scripts/gear_sonic_preview_sim2sim.sh` ← 部署保真但重：要 C++ deploy 二进制（TensorRT+cmake，
  `gear_sonic_deploy/` 只有源码没 build）+ `just` + `sudo ip link set lo multicast on`（否则
  CycloneDDS 在 loopback 上 "create domain error" 秒崩）。一般别用。

**「一启动就退」真因**：`isaaclab` env 从没装 gear_sonic 的 eval 依赖（gear_sonic 只是裸躺在
repo 根目录 path 上，没 pip install，也没 requirements 文件）。逐个炸出来，补齐这 6 块才通：
- `easydict` `loguru`（小 utils）、`open3d`（motion_lib mesh I/O 硬 import）、
  `vector_quantize_pytorch`（SONIC 的 VQ tokenizer，经 config `_target_` 动态 import，静态扫描抓不到）
- **`trl==0.28.0`**（关键）：repo 写于过渡版，`gear_sonic/trl/trainer/ppo_trainer.py` 同时
  `from trl.experimental.ppo import ...`（新）和 `from trl.trainer.ppo_trainer import *`（旧）；
  新版 trl（1.5.1）删了旧路径 → `ModuleNotFoundError: trl.trainer.ppo_trainer`。装 `--no-deps`
  不动 transformers/accelerate（repo 要 transformers>=4.56.2，本机 4.57.3 满足）。
- 虚惊：config 里 `groot.rl.trl.modules...` 不是缺包，是 `gear_sonic` 的运行时别名 shim。
- 已固化进 `gear_sonic_setup.sh` step 2/4（`conda run -n isaaclab pip install ...`）。

**submodule + patch**：WBC 现为 submodule（pin `a9d20b2`），改动走 `patches/gear-sonic/` +
`scripts/apply_gear_sonic_patches.sh`（幂等，照抄 leisaac）。
- `0001-download-symlink-into-hf-cache.patch`：`download_from_hf.py` 把 448M `last.pt` 从「拷进 repo」
  改成「symlink 指向 `~/.cache/huggingface/hub`」（`hf_hub_download` 本就下到 cache，旧脚本多 copy 一份）；
  只 symlink 大文件，`sample_data`(4M) 保持实体目录（gitignore `sample_data/` 带斜杠只匹配目录，
  symlink 会漏网污染状态）。
- `0002-g1-textured-usd-render.patch`：**修 G1 全白**。`robots/g1.py` 的
  `G1_CYLINDER_MODEL_12_DEX_CFG` 原用 `UrdfFileCfg(main.urdf)` 在线转 USD，Isaac Lab 的
  UrdfConverter 丢掉 URDF 里 per-link `<material> rgba`（0.05/0.2/0.7 灰）→ 全白（同 [[mimickit-g1-usd-material-fix]]
  的「源有色转换丢色」病，但机制是 URDF 转换不是 MJCF）。修法**不是**重绑材质，而是直接 spawn
  repo 自带的 `gear_sonic/data/robots/g1/g1_29dof_textured.usd`（同 29-DoF 拓扑 + 真灰材质
  0.105/0.575 + 完整 physics: massAPI×30/articulationRoot/colliders）→ `UsdFileCfg`，物理属性
  （rigid/articulation/contact）带过去，URDF 专属项（replace_cylinders/joint_drive）去掉。
  有 textured 用它否则回退 URDF（fresh clone 兜底）。⚠️ 注意 collider 源从 URDF-capsule 变成
  USD-baked，理论上接触动力学略变，但实测 walk 正常踏步（GPU 10G 持续）。

详见 [[feedback-mimo-independent-review]] 同期协作。

关联：[[lafan-g1-ecosystem]]（G1 motion 生态）、[[gr00t-multi-release-env-split]]（GR00T 多 env 隔离纪律）、
[[mimickit-to-vla-distill-plan]]（WBC+finetune 是 VLA 蒸馏的架构侧备选路线）。
