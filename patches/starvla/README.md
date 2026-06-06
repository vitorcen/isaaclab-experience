# starVLA 本地 patch / local patches

submodule `dependencies/starVLA` 跟踪官方上游 `github.com/starVLA/starVLA`
（pin 在 `e8f8fbb`）。本仓库需要的本地改动以 patch 形式维护，**不污染 submodule 历史**。
_The submodule tracks upstream clean; these patches carry the repo-local changes._

应用 / Apply（在 submodule 根目录）：

```bash
cd dependencies/starVLA
for p in ../../patches/starvla/000*.patch; do git apply "$p"; done
```

| Patch | 文件 / File | 改动 / Change | 原因 / Why |
|---|---|---|---|
| 0001 | `dataloader/gr00t_lerobot/datasets.py` | `_pack_sample` resize 224→448 | 橙子 10–40px，224 是 vision death-zone |
| 0002 | `dataloader/__init__.py` | `num_workers 16→4`, `prefetch 4→2` | 默认 16 workers 爆 62G RAM-cap |
| 0003 | `training/train_starvla.py` | 原子 save (`.tmp`+`os.replace`) + keep-last-N 裁剪（`trainer.keep_last_checkpoints`，默认 1） | ckpt 无裁剪填满 60G 盘 → ENOSPC 崩 |

SO-101 PickOrange 训练脚手架（config / modality / data_registry / run script）**不在此**，
见 `LeIsaac/scripts/training/starvla/`（一个 kit + 每变体一个 config）。
