---
name: frozen-vlm-head-extraction-sweep
description: 冻结-VLM VLA 云端 sweep 保全标准法=box端只抽~1.4G可训head拉回，本地用一次性vlm_base合并还原，绕开18.9G全量拉不动+keep_last删峰值
metadata:
  type: feedback
---

**问题**:冻结 VLM 的 StarVLA(QwenGR00T/QwenPI_v3）训练，每个 ckpt 都是 **18.9G(8B）**，
云端 `keep_last=N` 会在你拉回前把早期 ckpt 删掉 → **漏掉 sweep 峰值**；而 18.9G × 多台
经本机代理(~5.5MB/s)**实时拉不动**，box 端 rescue-cp 全量又**撑爆盘 ENOSPC 崩训练**。

**根因洞察(2026-06-07 实测)**:ckpt 里 **冻结 VLM `qwen_vl_interface` 占 92.7%(17.5G)，
每个 ckpt 完全相同**(冻结不更新=恒等于 base_vlm)；真正变的只有可训 head:
- **QwenPI_v3**:`action_model.`(1.08G)+ `project_layers.`(0.30G)= **1.38G(7.3%)**
- **QwenGR00T**:`action_model.` only(~1G，无 project_layers）

**标准法 = box 端抽 head 拉回 + 本地合并还原**(pull 量降 13×，3 台 sweep 全可行）:
1. **box 端 head 抽取器**(tmux 常驻，每出一个 ckpt 就抽）：
   ```python
   sd=torch.load(f,'cpu',mmap=True)        # mmap! 不把 17.5G VLM 读进 RAM(62G cgroup 安全)
   head={k:v for k,v in sd.items() if k.startswith(('action_model.','project_layers.'))}
   torch.save(head, f'heads/{bn}_head.pt')  # 1.4G，**11s 完成**
   ```
   head 小 → 多个全留 box 也不 ENOSPC，**彻底绕开 keep_last 裁剪**。脚手架 `/root/extract_heads.sh`。
2. **拉 heads/**(1.4G/个 ~4min，代理扛得住）。
3. **本地合并还原**:`{**vlm_base, **head}` → 完整 18.9G ckpt → serve 照常 eval（验证 merge
   **1422/1422 keys 完整**）。`vlm_base`(17.5G)**一次性存本地**复用,两种拿法:
   - (a)从该 run 任一 full ckpt mmap 抽非-head keys；
   - (b)**零网络本地 build**(`_head_sweep_tools/extract_vlm_base.py`):`build_framework(base=本机HF VLM)`
     拿"VLM载入+head随机初始化"模型,取非`action_model.`keys=vlm_base,**不用拉 17.9G 全量 ckpt**。
     校验 `vlm_base∪训练head==全模型keys`(Cosmos 实测 750+248=998 零缺零多)。坑:read_mode_config 要 .pt
     存在(touch 空 dummy)+ config 的 flash_attn2 在 CPU build 崩(覆盖 `attn_implementation="eager"`)+ `CUDA_VISIBLE_DEVICES=""`。
     **前提**:本机有该 run 的 base VLM(HF 下好)。Cosmos 走 (b) 省了 ~1h 全量下载。

**⚠️ 坑**:
- `vlm_base` **必须匹配该 run 的冻结 VLM**:PI_v3=Qwen3-VL-8B base；**Cosmos-Reason2-8B 是不同
  VLM(后训练版)→ 要单独抽 Cosmos 的 vlm_base**，不能混用。每个 backbone 一份 vlm_base。
- `base_framework.from_pretrained` 用 **strict=True** 加载 → **不能直接喂 head-only ckpt**(缺 VLM
  keys 报错）。所以走"本地合并成完整 ckpt"路径（serve 不改代码）；或改 serve 走 strict=False
  +`build_framework` 已从 base_vlm 载好 VLM 后只 load head（省本地 18.9G 文件，但要改框架）。
- 2B ckpt 只 5G，full-pull 也还行；但 head 抽取仍省 5×，且统一管线更省心。

**判据**:任何冻结-VLM 的大 VLA(VLM≥7B）云端 sweep，**默认走 head-extraction**，别全量拉。
关联 [[feedback-training-save-policy]] [[starvla-so101-cloud-training]] [[hf-upload-tricks]]。
