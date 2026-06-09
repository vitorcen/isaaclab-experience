---
name: autodl-hf-download-speed
description: AutoDL box 直下 HF 模型用 network_turbo + hf_transfer 可达 ~64MB/s；box→HF快 vs box→本机PULL才是~5MB/s瓶颈，别混淆，模型在box上直下别本地下再上传
metadata:
  type: reference
---

**AutoDL box 直接下 HF 模型很快,别本地下再上传。** 2026-06-07 实测 Qwen3.5-2B(~5GB)在 westc:

```bash
source /etc/network_turbo                 # AutoDL 学术加速器(对 HF/GitHub)
export HF_HUB_ENABLE_HF_TRANSFER=1         # 多线程下载,5-10×
pip install hf_transfer                    # ⚠️ 必装,否则 HF_HUB_ENABLE_HF_TRANSFER=1 直接 ValueError 下载失败
export HF_HOME=/root/autodl-tmp/hf_cache
hf download <repo> --local-dir <dir>
```
→ **实测 ≈ 64 MB/s**(20s 拉 1.28GB),和本机 hysteria 代理一个量级,5GB 模型几分钟完。

## ⚠️ 关键区分(别再搞混)
- **box → HF**(直下,network_turbo + hf_transfer)= **快,~64MB/s**。
- **box → 本机 PULL**(rsync ckpt 拉回我的机器)= **慢,~5MB/s** —— 这受**我本地从 box 的下行**限制,
  跟 box 下 HF 的速度无关。之前一直说的 "5.5MB/s 拉不动 18.9G ckpt" 是这条路,不是 box 下 HF。

## 推论
- **模型/数据集一律在 box 上直下**(64MB/s),**绝不**本地下完再上传到 box —— 上传到 box 是 ~5MB/s 上行瓶颈(5GB 要 ~17min)。
- sweep ckpt 保全仍要走 head-extraction([[frozen-vlm-head-extraction-sweep]]):因为那是 box→本机 PULL 方向,慢。
- 坑:`HF_HUB_ENABLE_HF_TRANSFER=1` 没装 hf_transfer 包 → 报 `Fast download ... 'hf_transfer' package is not available` 直接崩,先 `pip install hf_transfer`。

关联 [[hf-upload-tricks]] [[cn-pypi-mirror-aliyun]] [[feedback-autodl-cost-discipline]]。
