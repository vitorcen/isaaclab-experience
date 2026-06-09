---
name: feedback-cloud-env-reuse-disk-cleanup
description: 云端复用 box 起训前必清死重(旧base/smoke/旧run)+ 大VLM按 KEEP×ckpt_size 算磁盘峰值，否则训到中途 ENOSPC 崩(伪装成 flash-attn)
metadata:
  type: feedback
---

**复用一台 AutoDL box 起新训练时,旧运行的死重会累积 → 训到中途 torch.save 撞 ENOSPC 崩**,
崩的样子像 flash-attn 随机错,实为磁盘满。2026-06-07 一天内 westb(PI_v3)+ westd(Cosmos)各踩一次。

## 崩溃签名(认这个=ENOSPC,不是 flash-attn/CUDA)
- `RuntimeError: PytorchStreamWriter failed writing file data/N: file write failed`
- `unexpected pos 18410261952 vs 18410261848`(写到一半位置对不上)
- `checkpoints/` 里留一个 `steps_<N>_pytorch_model.pt.tmp` **残片**(比正常小,如 18.4G vs 18.9G)
- `df -h` 100% 满。外层只看到 `torch.distributed ... ChildFailedError`,要往上翻 traceback 才见真因。

## 死重四类(复用 box 必查)
1. **切 VLM 后漏删的旧 base**:westd 换 Cosmos 后 `models/Qwen3-VL-8B-Instruct`(17G)还在,Cosmos 用
   `Cosmos-Reason2-8B`,旧 base 纯死重(atime 是昨天=本次没碰)。
2. **smoke run 输出**:`so101_*_smoke`(steps_500 + final_model = 2×18G = 36G)。
3. **已拉/已发布的旧 run 输出**:westb 的 `so101_pickorange_qwen3vl8b_gr00t`(17G,已发 HF)。
4. **别的框架基座**:Cosmos-only box 上的 `wall-oss-0.5`(7.8G,wall-x 基座)。

## 起训/续训前预检(必做)
```bash
df -h /root/autodl-tmp; du -sh /root/autodl-tmp/* | sort -rh   # 看大头
du -sh /root/autodl-tmp/starvla-outputs/*                       # 旧 run?
du -sh /root/autodl-tmp/models/*                                # 漏删的旧 base?
```
**KEEP×ckpt_size 磁盘账**(8B ckpt≈18G):save 时峰值 outputs = `(KEEP+1) × ckpt`(留 KEEP 个 +
正写的 1 个 transient)。KEEP=2→54G,KEEP=3→72G。要求 `非output(base+env+data) + output峰值 < 盘`。
- PI_v3 崩:KEEP=3(72G)+ 53G死重 / 120G盘 → step 12000 撞顶。修=删53G死重+用户扩30G+`KEEP 3→2`。
- Cosmos 险:KEEP=2(54G)+ 17G死重Qwen / 100G盘 → step 18000 会撞(算出来 used 会到101.7G)。
  修=删17G Qwen + 7.8G wall-oss → 59G空 → 峰值~85G<100G,**不用重启**(Cosmos没崩,提前清掉就行)。

## 崩后恢复
1. 删 `.tmp` 残片(腾回一个 ckpt 的空间)。2. 清死重把空间做足。3. 降 KEEP 让峰值<盘+裕量。
4. `RESUME=1`(`run_train.sh` → `--trainer.is_resume True`)从**最新完整 ckpt** 续(原子 save
   .tmp→rename,留下的都是完整的);只丢最后一个 save 之后那段。日志确认 `📦 loading checkpoint: steps_N`。

## ⚠️ 连环坑:RESUME 的 ckpt 加载会 62G cgroup OOM(大 VLM)
ENOSPC 修好 `RESUME=1` 重启后,PI_v3 在 `📦 loading checkpoint` 处**又静默 SIGKILL**(无 traceback=信号杀):
`trainer_tools.py:load_pretrained_backbones` 里 `torch.load(ckpt, map_location="cpu")` 把 **18.9G ckpt 整个读进
RAM**,叠加已 `build_framework` 建好的 8B 模型(~17G)→ 超 **AutoDL 62G cgroup**(`memory.max=66571993088`)→ OOM。
注意 `free -g` 看到的是 **host 503G** 不是 cgroup,容器里 dmesg 也看不到 host OOM,只能靠"加载处静默死+无报错"判。
**修(已验证)**:`torch.load(..., map_location="cpu", mmap=True)` —— tensor 从磁盘 mmap 不进 RAM,
`load_state_dict` 逐个拷,峰值只剩模型本身。和 head 抽取器同款 mmap。改完 RESUME 顺利越过加载、`steps=1` 开跑。
**判据**:任何 ≥7B VLM 在 62G cgroup box 上 resume(full torch.load ckpt)默认加 `mmap=True`,别等 OOM。

**判据**:任何复用 box 起训,**先 du 三处清死重 + 算 (KEEP+1)×ckpt < 盘**,别等训到一半崩。
冻结-VLM 走 head-extraction([[frozen-vlm-head-extraction-sweep]])时 KEEP 可压到 1-2(head 已拉回保全 sweep,
box 只需留最新 full 做 resume)。关联 [[feedback-training-save-policy]] [[three-box-sweep-live-state]]。
