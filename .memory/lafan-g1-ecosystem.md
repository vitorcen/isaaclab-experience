---
name: lafan-g1-ecosystem
description: G1 humanoid LAFAN1 motion-tracking 现成 ckpt + 数据 + retargeter 的开源生态地图（2026-06-01 调研）
metadata:
  type: reference
---

## 数据（HF datasets）

| repo | 格式 | DoF | 兼容 MimicKit | 备注 |
|---|---|---|---|---|
| **`ember-lab-berkeley/LAFAN-G1`** ⭐ | Isaac Lab npz | **29** | ✅ 用我们 `scripts/lafan_g1_npz_to_mimickit.py` 转 | Berkeley 出，CC-BY-4.0；100% LAFAN1 retarget |
| `openhe/g1-retargeted-motions` | ASAP pkl (joblib) | **23** | ❌ DoF 少了 6 个手腕 | ACCAD 113 / DanceDB 13 / kungfu 8 (Bruce Lee / Roundhouse / Side kick…) / LAFAN1 40；MIT；Mink retargeter |
| `lvhaidong/LAFAN1_Retargeting_Dataset` (2.8K dl) | ? | ? | 未验证 | G1 + H1 + H1_2 三机器人 |
| Ubisoft `lafan1.zip` 144 MB | BVH 原始 | — | 需 retarget | github.com/ubisoft/ubisoft-laforge-animation-dataset |

## 现成 G1 motion-tracking ckpt

| 项目 | ckpt | pipeline | 跟 MimicKit 兼容 |
|---|---|---|---|
| **`YanjieZe/TWIST`** | `assets/twist_general_motion_tracker.pt` (4 MB) — **通用** motion tracker | legged_gym + Isaac Gym Preview | ❌ obs/action 空间不同 |
| **`NVlabs/ProtoMotions`** (1691⭐, 2026-05 最新) | README 说有 "pre-trained models today"；General Tracking Policy for G1 + BONES-SEED 142K motions | 自己的 framework（IsaacLab 兼容） | ❌ 不能塞 MimicKit |
| `LeCAR-Lab/ASAP` (2031⭐, RSS 2025) | **不发预训练**，只发训练代码 | HumanoidVerse | — |
| `MarkFzp/humanplus` (847⭐, CoRL 2024) | 旧，H1 为主 | 自己的 framework | — |
| **MimicKit 官方** | 4 个 G1 ckpt (spinkick / double_kong / lcp_walk / add_run) | MimicKit ✅ | 我们 pipeline 唯一现成可用 |

## Retargeter

- **`YanjieZe/GMR`** (2268⭐, ICRA 2026) — AMASS / OMOMO / LAFAN1 / Nokov / OptiTrack BVH / **YouTube 视频 (via GVHMR)** → 17 种 humanoid robot 实时 retargeter（含 G1 / G1+Dex31 / H1 / H1_2 / Booster T1/K1 / GR3 / Pi+ / KUAVO / Talos / Tienkung / Berkeley Lite / Galexea R1 Pro / PND Adam Lite）。MimicKit 自带 `tools/gmr_to_mimickit/` 转格式

## How to apply

- 问"有 LAFAN_fight G1 ckpt 吗？"→ **生态里 0 个**，要么 ProtoMotions 通用 policy（不在 MimicKit），要么自训（参考 [[mimickit-lafan-fight-training-plan]]）
- 想用 TWIST ckpt → 不现实，要切换整套 pipeline 到 legged_gym + Isaac Gym Preview
- 想适配新数据集 → 优先看是否有 HF 上的 retargeted 版本，再考虑 GMR 端到端 retarget
- openhe 23-DoF 数据**仅作参考**，不要塞 MimicKit pipeline（29-DoF），DoF mismatch 会直接 obs 维度爆
