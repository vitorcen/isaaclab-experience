---
name: lerobot-dp-async-server-bug
description: "🐛 lerobot v0.4+v0.5 async server 把 DP n_obs_steps>1 弄死的 root cause + 一行 patch — `predict_action_chunk` 不 populate_queues 导致 stack 空"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1e92da1e-d4b0-4cfa-b922-39f17de1ca36
---

# lerobot DP async server "policy not moving" 根因

## Symptom

DP via lerobot `async_inference.policy_server` + LeIsaac client:
- server log: `ERROR Error in StreamActions: stack expects a non-empty TensorList`
- client log: `[CLIENT] no actions after 8 retries (200ms); reusing last action`
- Isaac Sim arm 不动 → eval 全 0/15 stuck @ ~33s
- 影响所有 `n_obs_steps>1` 的 lerobot policy（DP default=2，VQBet 等）
- **v0.4 和 v0.5 都有这 bug**，wsagi/DiffusionPolicy-PickOrange 0/15 stuck 也是它

## Root cause

`lerobot/policies/diffusion/modeling_diffusion.py` 里 `predict_action_chunk()`（async server 入口）直接对空 deque 做 `torch.stack(list(self._queues[k]), dim=1)`，但**没调 `populate_queues`**。`select_action()` 调了 populate_queues，但 async server 用的是 `predict_action_chunk` 这条路径。

`populate_queues` 内置 cold-start 复制逻辑：首帧来时把它复制 n_obs_steps 份填满 deque。少了这一步，server 第 1 次 inference 就崩 stream。

## Fix (单文件 patch)

`lerobot/policies/diffusion/modeling_diffusion.py` `predict_action_chunk` 顶部加 select_action 的前置逻辑：

```python
@torch.no_grad()
def predict_action_chunk(self, batch, noise=None):
    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
    self._queues = populate_queues(self._queues, batch)
    batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in self._queues if k != ACTION}
    actions = self.diffusion.generate_actions(batch, noise=noise)
    return actions
```

已 apply 到 `/home/david/work/lerobot-v040/src/lerobot/policies/diffusion/modeling_diffusion.py`（editable install 立即生效）。

## Verification

ckpt-10000 patched server 实测：
- server: `Running inference for observation #N` + `Action chunk #N generated | Total time: 388ms` 连续输出
- 不再有 `stack expects a non-empty TensorList`
- Isaac 机械臂能动了

## How to apply

- v0.4 editable: 改 `/home/david/work/lerobot-v040/src/.../modeling_diffusion.py` → 重启 server 即可
- v0.5 site-packages: 直接 patch `lerobot/policies/diffusion/modeling_diffusion.py`（或上游 PR）
- Upstream: 应该提一个 PR，影响所有 n_obs_steps>1 的 lerobot policy

## 关联

- [[compact-resume-2026-05-22]] DP-v040-fullres 原本以为是 crop_shape，实际两个 bug 叠加
- [[leisaac-eval-timeout]] DDIM swap 是 latency 优化，不解决这个 stack bug
- [[feedback-incremental-eval-during-training]] 10-slice quick eval 能立刻看出 stuck（必须在 sec 1-5 内）
