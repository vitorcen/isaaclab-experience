---
name: gr00t-n17-leisaac-wire-debug
description: GR00T-N1.7 自训 ckpt sim eval arm 不动的 4 层 wire 协议根因 + service_policy_clients.py patch
metadata: 
  node_type: memory
  type: project
  originSessionId: 39899a1b-ef9c-4403-93fc-2e3491bfb440
---

# GR00T-N1.7 × LeIsaac sim eval 4 层 wire 根因（2026-05-23）

## TL;DR

自训 N1.7 ckpt-1200 sim arm 不动。Server probe 出非零 action range -28°~+21° 看似 OK，**arm 不动 root cause 全在 client wire**，不是 ckpt 不行。剥洋葱 4 层 bug 全在 `LeIsaac/source/leisaac/leisaac/policy/service_policy_clients.py:Gr00tServicePolicyClient`。

修好后 ckpt-1200 oranges 0/3（训练量不足 — 才 20% / hi-space 6000 step 收敛），但 **arm 会动了**。

**Why:** 4 个 bug 串行触发；最后一个 mnp.decode bytes vs str key 反直觉，标准 fix 思路（把 bytes key → str）反而破坏 mnp.decode。

**How to apply:** 任何 LeIsaac 接 GR00T N1.7 sim_policy_wrapper server 时，client 必须按本文 4 个 fix 同时改；详见 [[gr00t_n17_sim_wire_protocol_debug.html]] (LeIsaac/docs/training/)

## 4 层 wire mismatch

| # | Bug | Fix |
|---|-----|-----|
| 1 | LeIsaac N1.5 client 发 `data=flat_obs` 没包 `observation`；sim_policy_wrapper server 把 `**data` 当 kwargs 解包 → `'video.front'` 不是合法 Python identifier | 包 `{"observation": flat_obs}` |
| 2 | sim_policy_wrapper.check_observation 要 video (B,T,H,W,C) 5D uint8 + state (B,T,D) 3D float32；LeIsaac 给 (B,H,W,C) 4D + (B,D) 2D float64 | 每个 ndarray `v[:, None, ...]` 插 T=1 + 强转 float32/uint8 |
| 3 | N1.7 server 返回 ndarray 用 msgpack_numpy 编码成 `{b'nd':True, b'data':bytes, b'type':'<f4', b'shape':[...]}` **bytes-keyed dict**；MsgSerializer._decode_custom 只查 `obj.get("nd")` (str)，bytes 漏过 → client 拿 dict 不是 ndarray | client 端 `_to_ndarray` 兜底 |
| 4 | `mnp.decode` 内部 hard-code 用 bytes key 查 `b'nd'`；反向把 bytes→str 反而破坏它 | 保留 bytes key，或反向 re-encode str → bytes |

## 关键代码片段（patch）

```python
# service_policy_clients.py:Gr00tServicePolicyClient.get_action
wrapped_obs = {}
for k, v in obs_dict.items():
    if isinstance(v, np.ndarray):
        if k.startswith("video.") and v.ndim == 4:
            wrapped_obs[k] = v[:, None].astype(np.uint8)
        elif k.startswith("state.") and v.ndim == 2:
            wrapped_obs[k] = v[:, None].astype(np.float32)
        else:
            wrapped_obs[k] = v

action_chunk = self.call_endpoint("get_action", {"observation": wrapped_obs})
if isinstance(action_chunk, list):
    action_chunk = action_chunk[0]
arm  = _to_ndarray(action_chunk["action.single_arm"])
grip = _to_ndarray(action_chunk["action.gripper"])
if arm.ndim == 3 and arm.shape[0] == 1:
    arm, grip = arm[0], grip[0]
concat = np.concatenate([arm, grip], axis=-1)
concat = convert_lerobot_action_to_leisaac(concat)
return torch.from_numpy(concat[:, None, :])

def _to_ndarray(x):
    if isinstance(x, np.ndarray): return x
    if isinstance(x, dict):
        ks = set(x.keys())
        if "nd" in ks or b"nd" in ks:
            if "nd" in ks:  # re-encode str → bytes
                x = {(k.encode() if isinstance(k, str) else k): v for k, v in x.items()}
            return msgpack_numpy.decode(x)  # ← 必须 bytes keys
        ...
```

`server/eval_gr00t.sh` 加 `POLICY_TYPE` env, 默认 `gr00tn1.5` (与新 client 配套)。

## 次生坑（也要记）

- **5090 32GB stale eval 累积 OOM**：每次 eval timeout 杀 bash 但 conda subprocess 不退；3 次后 PhysX 拒绝 createScene。看起来像新 bug 实际是僵尸进程占 6.7GB×N
- **修法**：每次 eval 结束后 `pkill -9 -f policy_inference` 兜底；或者 cmd 用 `setsid bash ... ; trap pkill EXIT`
- **Python .pyc cache** 偶尔骗人：编辑 client 后必须 `find LeIsaac -name __pycache__ -exec rm -rf {} +` 清缓存才确保 eval 加载新代码

## 关联

- [[gr00t-n17-hi-space]] — hi-space N1.7 14/15 SOTA 标杆；wire 协议 N1.6→N1.7 改 mnp 格式是这次错配源头
- [[compact-resume-2026-05-23]] — 同一天上午对 ckpt 误判"缺 332 keys broken"（实际 698 keys = N1.7 完整 size），证伪后才发现是 wire 问题
- [[per-model-action-horizon]] — N1.7 action_horizon=40
- HTML: `LeIsaac/docs/training/gr00t_n17_sim_wire_protocol_debug.html` (4 层 bug + SVG 通信链路)
