# openvla_leisaac — OpenVLA-7B 4-bit demo server

ZMQ inference server that wraps `openvla/openvla-7b` in NF4 4-bit quantization
and exposes the same wire protocol as `pi05_leisaac` so the existing LeIsaac
`Pi05ServicePolicyClient` can hit it by just switching `--policy_port=5557`.

**Status (2026-05-17)**: base model only, NOT fine-tuned. Action space is
fundamentally mismatched (OpenVLA → BridgeData EEF deltas; SO-101 → joint
positions). See "Action remap" below — this is a demo / smoke target.

## What's here

| File | Purpose |
| --- | --- |
| `server.py` | ZMQ + msgpack inference server. Single front camera. NF4 4-bit. |
| `smoke.py`  | Offline probe: load model, run 4 prompts on one image, print actions. |
| `pyproject.toml` | Pinned deps (transformers 4.40.1, bnb 0.43.1, etc). |

## Sanity-checked numbers

- Load time: ~32 s (3 shards from HF cache)
- GPU after load: **4.38 GB** (4-bit NF4)
- Inference latency: ~440 ms cold, **~240 ms warm** (no flash_attn)
- Sensitivity probe: "move up" → dz=+0.005, "move left" → dy=+0.002 — directionally correct, magnitude tiny.
- Gripper: stuck at 0.996 across all 4 prompts on LeIsaac frame — base model has no clue about this embodiment.

## Action remap (cosmetic only)

OpenVLA returns 7-DoF `[dx, dy, dz, drx, dry, drz, gripper]` (BridgeData WidowX
EEF cartesian deltas, meters / radians). SO-101 client expects 6-DoF *joint*
*positions* `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper]`. We do:

```python
arm_abs = state[:5] + act7[:5] * ARM_DELTA_SCALE   # 0.05 default
grip    = act7[6]                                  # passthrough
```

Result: arm drifts gently, gripper response is whatever the base model decides.
Real grasp behavior requires fine-tuning on LeIsaac data.

## Usage

```bash
# 1) Smoke test (offline, no Isaac Sim needed)
conda run -n openvla python -m openvla_leisaac.smoke --image /tmp/leisaac_frame0.png

# 2) Start the demo server
bash server/serve_openvla.sh                 # foreground, port 5557
bash server/serve_openvla.sh --detach        # background + tail until ready

# 3) Run LeIsaac eval against it
POLICY_PORT=5557 ACTION_HORIZON=1 EVAL_ROUNDS=1 EPISODE_LENGTH=120 \
    bash server/eval_pi05.sh                 # reuses pi05 eval, just different port

# 4) Stop
bash server/stop_server.sh                   # or kill -SIGINT $(cat logs/openvla_server.pid)
```

## Why a separate conda env

OpenVLA pins `transformers==4.40.1` (its `modeling_prismatic.py` uses pre-tf5
internal APIs). lerobot env has tf 5.3+ which is incompatible. The `openvla`
env is built around this constraint:

| Component | Version | Why |
| --- | --- | --- |
| python | 3.10 | OpenVLA support tier |
| torch | 2.3.1+cu121 | matches triton 2.3.1 / bnb 0.43.1 prebuilts |
| transformers | 4.40.1 | OpenVLA pinned (5.x renames break trust_remote_code) |
| bitsandbytes | 0.43.1 | needs triton 2.x; 0.46+ requires triton 3+ |
| accelerate | 0.31.0 | matches tf 4.40 era; 1.x calls `model.to()` post-bnb |
| huggingface_hub | 0.23.4 | tf 4.40 expects ~0.22-0.24 API |
