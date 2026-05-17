#!/usr/bin/env python3
"""OpenVLA-7B demo inference server (4-bit NF4 quantized).

Wire-compatible with LeIsaac ``Pi05ServicePolicyClient``: ZMQ REQ/REP + msgpack
with the custom ``__ndarray__`` envelope. Two endpoints:

    ping        -> {"status": "ok", "message": "pong"}
    get_action  -> {"status": "ok", "data": {"action.single_arm": (T, 5),
                                              "action.gripper":    (T, 1)},
                    "inference_time_ms": float}

⚠️  Action-space mismatch (demo only, not for production):
    OpenVLA outputs 7-DoF BridgeData EEF cartesian deltas
    [dx, dy, dz, drx, dry, drz, gripper] (meters / radians).
    SO-101 expects 6-DoF *joint positions*. We do a naive cosmetic remap:

        arm_abs = state[:5] + act7[:5] * ARM_DELTA_SCALE   # treat EEF Δ as joint Δ
        grip    = act7[6]                                  # passthrough [0..1]

    The arm will drift gently; gripper response is whatever the base model
    decides for the prompt+image. Fine-tuning is what makes this useful.

Run:
    bash server/serve_openvla.sh                     # default 127.0.0.1:5557
    PORT=5557 bash server/serve_openvla.sh --detach  # background
    python -m openvla_leisaac.server --host 0.0.0.0 --port 5557
"""

from __future__ import annotations

import argparse
import io
import time
import traceback
from typing import Any

import msgpack
import numpy as np
import torch
import zmq
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


DEFAULT_MODEL = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"
DEFAULT_PROMPT = "Pick up the orange and place it on the plate"
DEFAULT_ARM_DELTA_SCALE = 0.05  # OpenVLA Δ is meters; SO-101 joint Δ should be much smaller


# --- wire format -------------------------------------------------------------
def _pack_ndarray(arr: np.ndarray) -> dict:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return {
        "__ndarray__": True,
        "data": buf.getvalue(),
        "dtype": str(arr.dtype),
        "shape": arr.shape,
    }


def _unpack_ndarray(obj: Any) -> np.ndarray:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.load(io.BytesIO(obj["data"]), allow_pickle=False)
    return np.array(obj)


# --- model -------------------------------------------------------------------
class OpenVLADemoServer:
    """OpenVLA-7B NF4 4-bit, single front camera, naive joint-space remap."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        unnorm_key: str = DEFAULT_UNNORM_KEY,
        default_prompt: str = DEFAULT_PROMPT,
        arm_delta_scale: float = DEFAULT_ARM_DELTA_SCALE,
        device: str = "cuda:0",
    ) -> None:
        self.unnorm_key = unnorm_key
        self.default_prompt = default_prompt
        self.arm_delta_scale = arm_delta_scale
        self.device = device

        print(f"[openvla] loading 4-bit NF4 model: {model_name}", flush=True)
        t0 = time.time()
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print(
            f"[openvla] loaded in {time.time()-t0:.1f}s, "
            f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB",
            flush=True,
        )

    def predict_action(
        self,
        front_img: np.ndarray,
        state6: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Return (1, 6) absolute joint-position action for SO-101.

        front_img: (H, W, 3) uint8 RGB.
        state6:    (6,) current [arm5, gripper1] joint positions.
        prompt:    natural language instruction.
        """
        img = Image.fromarray(front_img.astype(np.uint8))
        fmt_prompt = f"In: What action should the robot take to {prompt}?\nOut:"
        inputs = self.processor(fmt_prompt, img).to(self.device, dtype=torch.float16)
        act7 = self.vla.predict_action(
            **inputs, unnorm_key=self.unnorm_key, do_sample=False
        )
        # act7 = [dx, dy, dz, drx, dry, drz, gripper]
        arm_delta = act7[:5].astype(np.float32) * self.arm_delta_scale
        arm_abs = state6[:5].astype(np.float32) + arm_delta
        grip = np.array([float(act7[6])], dtype=np.float32)
        return np.concatenate([arm_abs, grip])[None]  # (1, 6)

    # --- wire-compat hook for Pi05ServicePolicyClient ------------------------
    def get_action(self, obs: dict) -> dict:
        front = obs.get("video.front")
        if front is None:
            raise ValueError(
                "OpenVLA demo server needs 'video.front'. "
                f"Got keys: {sorted(obs)}"
            )
        if isinstance(front, np.ndarray) and front.ndim == 4:
            front = front[0]

        arm5 = np.asarray(obs.get("state.single_arm", np.zeros(5))).ravel()
        grip1 = np.asarray(obs.get("state.gripper", np.zeros(1))).ravel()
        state6 = np.concatenate([arm5, grip1]).astype(np.float32)

        task = obs.get("annotation.human.task_description", self.default_prompt)
        if isinstance(task, list) and task:
            task = task[0]
        if isinstance(task, bytes):
            task = task.decode()

        actions = self.predict_action(front, state6, str(task))
        return {
            "action.single_arm": actions[:, :5].astype(np.float32),
            "action.gripper":    actions[:, 5:6].astype(np.float32),
        }


# --- ZMQ loop ----------------------------------------------------------------
def serve(server: OpenVLADemoServer, host: str, port: int) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[openvla] ready, listening on tcp://{host}:{port}", flush=True)

    step = 0
    while True:
        try:
            raw = sock.recv()
            req = msgpack.unpackb(raw, raw=False)
            ep = req.get("endpoint", "")

            if ep == "ping":
                sock.send(msgpack.packb({"status": "ok", "message": "pong"}))
                continue

            if ep == "get_action":
                obs = {
                    k: (_unpack_ndarray(v) if isinstance(v, dict) and v.get("__ndarray__") else v)
                    for k, v in (req.get("data") or {}).items()
                }
                t0 = time.time()
                action = server.get_action(obs)
                infer_ms = 1000 * (time.time() - t0)
                data = {
                    k: (_pack_ndarray(v) if isinstance(v, np.ndarray) else v)
                    for k, v in action.items()
                }
                sock.send(msgpack.packb(
                    {"status": "ok", "data": data, "inference_time_ms": infer_ms}
                ))
                step += 1
                if step % 10 == 0:
                    print(
                        f"[openvla] step={step} "
                        f"action6={action['action.single_arm'][0].tolist()} "
                        f"grip={float(action['action.gripper'][0,0]):.3f} "
                        f"latency={infer_ms:.0f}ms",
                        flush=True,
                    )
                continue

            sock.send(msgpack.packb({"status": "error", "message": f"Unknown endpoint: {ep}"}))
        except KeyboardInterrupt:
            print("[openvla] interrupted, shutting down", flush=True)
            break
        except Exception as e:
            traceback.print_exc()
            try:
                sock.send(msgpack.packb({"status": "error", "message": str(e)}))
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="fallback prompt when client doesn't supply one")
    ap.add_argument("--arm-delta-scale", type=float, default=DEFAULT_ARM_DELTA_SCALE)
    args = ap.parse_args()

    server = OpenVLADemoServer(
        model_name=args.model_name,
        unnorm_key=args.unnorm_key,
        default_prompt=args.prompt,
        arm_delta_scale=args.arm_delta_scale,
    )
    serve(server, args.host, args.port)


if __name__ == "__main__":
    main()
