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
| `pyproject.toml` | Pinned deps (transformers 4.40.1, bnb 0.46.1, etc). |

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
| torch | 2.3.1+cu121 | triton 2.3.1 prebuilts; works fine with bnb 0.46.x |
| transformers | 4.40.1 | OpenVLA pinned (5.x renames break trust_remote_code) |
| **bitsandbytes** | **0.46.1** | **0.43.x has random Linear4bit+PEFT memory corruption (see "Crash fix" below). 0.46.1 works with triton 2.3.1 — the "needs triton 3+" claim was wrong, verified empirically 2026-05-19.** |
| accelerate | 0.31.0 | matches tf 4.40 era; 1.x calls `model.to()` post-bnb |
| huggingface_hub | 0.23.4 | tf 4.40 expects ~0.22-0.24 API |

## Install from scratch

```bash
# 1) Create env (Python 3.10 + CUDA-bundled torch)
conda create -n openvla python=3.10 -y
conda activate openvla
pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Install pinned deps + this package (editable, so server.py edits take effect immediately)
cd /path/to/isaaclab-experience
pip install -e server/openvla_leisaac

# 3) Verify
conda run -n openvla python -c "import bitsandbytes, transformers, peft, accelerate, torch; \
  print(f'bnb={bitsandbytes.__version__} (must be 0.46.x)'); \
  print(f'tf={transformers.__version__} (must be 4.40.1)'); \
  print(f'peft={peft.__version__} (must be 0.11.1)'); \
  print(f'torch={torch.__version__}+cu{torch.version.cuda}')"
```

If you already have an `openvla` env on 0.43.x, **upgrade in place**:
```bash
conda run -n openvla pip install --upgrade "bitsandbytes>=0.46.1,<0.47"
```

## Crash fix · bnb 0.43.1 → 0.46.1 (2026-05-19)

**Symptoms on bnb 0.43.1**: training and server load crash randomly with three
different tracebacks, all rooted in `torch.nn.Module._named_members` tuple
unpack failing inside bnb 4-bit + PEFT wrapping:

1. `ValueError: too many/not enough values to unpack (expected N, got 2)` from
   `Trainer.log -> floating_point_ops -> named_parameters`
2. Same ValueError from `bnb_4bit_quantizer.preprocess_model -> find_tied_parameters`
3. Pure SIGSEGV ("段错误（核心已转储）"), no Python traceback
4. **Worst**: same ValueError flies into completely unrelated code like
   `sre_compile` (regex) or `pandas` import — because the C extension is
   corrupting Python heap memory, the bad tuple turns up wherever the next
   iterator unpack happens.

**Root cause**: bnb 0.43.1's `Params4bit` / `Linear4bit` doesn't play nice with
PEFT's wrapper traversal under newer transformers. Fixed in 0.46.1 via
[PR #1719](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1719)
+ [PR #1866](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1866).

**Measured impact** (RTX 4090, 5500→6000 LoRA chunk with `save_steps=100` +
20-retry watchdog):

| metric | bnb 0.43.1 | bnb 0.46.1 |
| --- | --- | --- |
| 1h progress | 5500 stuck — 0 new ckpts | 5500 → 5700+ — 2 new ckpts saved |
| Crash window | step 5527–5578 (before any save boundary) | step 5721+ (past save boundary) |
| Watchdog retries needed | 20+ exhausted, abort | 1–2 per chunk |

**Belt-and-braces patches** kept in `server.py` + `train.py` even after upgrade —
they shield the residual segfault class:

```python
import transformers.modeling_utils as _mu
import accelerate.utils.modeling as _am
import accelerate.utils as _au
import transformers.integrations.bitsandbytes as _tb
_mu.PreTrainedModel.floating_point_ops = lambda self, inputs, exclude_embeddings=True: 0
_noop_tied = lambda *a, **kw: []
_am.find_tied_parameters = _noop_tied
_au.find_tied_parameters = _noop_tied
_tb.find_tied_parameters = _noop_tied
```
