---
name: openvla-floatingpointops-fix
description: OpenVLA 7B QLoRA 训练中 Trainer.log → floating_point_ops → _named_members 解包错的根治补丁（替代 watchdog）
metadata: 
  node_type: memory
  type: project
  originSessionId: c14e8ac0-2fe8-4b9e-96f8-0c66dd49e974
---

## 症状

OpenVLA + bnb 4-bit + PEFT LoRA 训练随机 100-500 step 崩，stack：

```
trainer.py:3878  floating_point_ops(inputs)
modeling_utils.py:1218  num_parameters(exclude_embeddings)
modeling_utils.py:1139  total_parameters = [...named_parameters...]
torch/nn/module.py:2199  for k, v in members:
ValueError: too many values to unpack (expected 0)
```

历史上 0→5k 训练崩了 32 次靠 [[compact-resume-2026-05-18]] watchdog auto-resume 拯救。

## 根因

`bnb.Linear4bit` 在 4-bit + PEFT 包装下 `_parameters` 的 `items()` 偶发返回畸形 tuple，触发 `_named_members` 解包错。

trainer 调 `floating_point_ops` 仅用于打日志里 `TFLOPS` 那一列——返回值除了日志没别的用途。

## 修法（A+B 补丁，2 个 lambda + 4 个名字 rebind）

`LeIsaac/scripts/finetune/openvla/train.py` 顶部 transformers import 后：

```python
import transformers.modeling_utils as _mu
import accelerate.utils.modeling as _am
import accelerate.utils as _au
import transformers.integrations.bitsandbytes as _tb

# A — 训练日志路径 (Trainer.log → floating_point_ops)
_mu.PreTrainedModel.floating_point_ops = lambda self, inputs, exclude_embeddings=True: 0

# B — 模型加载路径 (bnb 4-bit quantizer → accelerate.find_tied_parameters)
_noop_tied = lambda *a, **kw: []
_am.find_tied_parameters = _noop_tied
_au.find_tied_parameters = _noop_tied
_tb.find_tied_parameters = _noop_tied  # bnb 整合模块已 from accelerate.utils import 复制了引用
```

副作用：
- A：日志 TFLOPS 列恒为 0
- B：bnb 量化 lm_head（OpenVLA 走 action token，不影响）

watchdog 兜底剩余 segfault。

## 附加：trainer_state.json 冻结 save_steps

resume 时 Trainer 用 trainer_state 里的 save_steps（不是 CLI 传的），导致 `--save_steps 500` 被忽略。chunked 训练 target=5500 时实际 ckpt 在 5200/5400，不在 5500。

修法：train.py 在 `trainer.train()` 返回后强制 `trainer._save_checkpoint(model, trial=None)`，落地 `checkpoint-{global_step}`。

## 没试过的备选

- B：升 bnb 0.43→0.44 + peft + transformers（半天工作量，可能波及 X-VLA / FastWAM 共用 env）
- C：subclass Trainer 覆写 `floating_point_ops`（5 行，等同 A 但更 cleaner）

## 关联

- [[compact-resume-2026-05-18]] watchdog 32-retry 历史
- [[fastwam-qlora-finetune]] 同样 bnb + LoRA 组合（未出现此 bug，因 chunk size 不同？）
