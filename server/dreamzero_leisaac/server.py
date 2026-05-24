"""DreamZero (Wan2.1-I2V-14B + LoRA) policy server for LeIsaac SO-101 PickOrange sim eval.

Wire protocol: ZMQ REQ/REP + msgpack-numpy (matches Gr00tServicePolicyClient).
Model: Wan2.1-I2V-14B NF4 quantized + LoRA adapter loaded on single 4090 24G.
Forward: autoregressive chunked diffusion (action_horizon=24, num_inference_timesteps=4).

USAGE:
    bash server/dreamzero_leisaac/start.sh \
        --ckpt-path /home/david/work/isaaclab-experience/LeIsaac/outputs/dreamzero-leisaac-so101-lora-r4/checkpoint-2000

Phase 2 (live): wraps `DreamZeroLeIsaacPolicy` which builds VLA on meta + NF4 stream-load + LoRA merge.
VRAM footprint after pre-warm: ~12.5GB cur / ~17.5GB peak — leaves 6+GB for Isaac Sim share.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq


# ----- ZMQ + msgpack wire protocol (mirrors dependencies/Isaac-GR00T/gr00t/policy/server_client.py) -----

class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return mnp.packb(data, default=MsgSerializer._encode_custom)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return mnp.unpackb(data, object_hook=MsgSerializer._decode_custom, raw=False)

    @staticmethod
    def _encode_custom(obj):
        if isinstance(obj, torch.Tensor):
            return mnp.encode(obj.detach().cpu().numpy())
        return obj

    @staticmethod
    def _decode_custom(obj):
        if not isinstance(obj, dict):
            return obj
        # msgpack-numpy auto-encoded ndarray: {b"nd": True, ...} or {"nd": True, ...}
        if "nd" in obj or b"nd" in obj:
            return mnp.decode(obj)
        # LeIsaac legacy: ndarray as {__ndarray_class__: True, as_npy: bytes}
        if "__ndarray_class__" in obj or b"__ndarray_class__" in obj:
            import io
            key = "as_npy" if "as_npy" in obj else b"as_npy"
            return np.load(io.BytesIO(obj[key]), allow_pickle=False)
        return obj


# ----- DreamZero policy wrapper -----

class DreamZeroPolicy:
    """Wraps DreamZeroLeIsaacPolicy from LeIsaac/scripts/inference/dreamzero/."""

    def __init__(self, ckpt_path: str, cfg_dir: str, wan_snap_dir: str,
                 action_horizon: int = 24, device: str = "cuda"):
        self.ckpt_path = Path(ckpt_path)
        self.action_horizon = action_horizon
        self.device = device

        candidates = [self.ckpt_path / "adapter_model.pt", self.ckpt_path / "model.safetensors"]
        adapter = next((p for p in candidates if p.exists()), None)
        if adapter is None:
            raise FileNotFoundError(f"no adapter file (adapter_model.pt or model.safetensors) found in {self.ckpt_path}")

        # Import the policy class (also lives in dreamzero env)
        sys.path.insert(0, "/home/david/work/isaaclab-experience/LeIsaac/scripts/inference/dreamzero")
        from dreamzero_policy import DreamZeroLeIsaacPolicy as _Policy
        print(f"[DreamZero] building inference policy from {ckpt_path}...", flush=True)
        self._policy = _Policy(
            ckpt_dir=ckpt_path,
            cfg_dir=cfg_dir,
            wan_snap_dir=wan_snap_dir,
            embodiment_str="xdof",
        )
        print(f"[DreamZero] policy ready, action_horizon={self.action_horizon}", flush=True)

    def get_action(self, obs: dict) -> dict:
        """Receive LeIsaac obs, return action chunk.

        obs (msgpack-decoded dict):
            video.front: (B, H, W, 3) uint8 (or (H,W,3))
            video.wrist: (B, H, W, 3) uint8 (or (H,W,3))
            state.joint_pos: (B, 5) float32
            state.gripper_pos: (B, 1) float32
            annotation.task: [task_str] or str

        Return:
            {"action.joint_pos": (T, 5), "action.gripper_pos": (T, 1)} with T=24.
        """
        # Normalize obs format — LeIsaac client sends `(B, T, H, W, 3)` for video, take first frame
        norm = {}
        for k in ("video.front", "video.wrist"):
            if k in obs:
                v = np.asarray(obs[k])
                # Drop batch dims down to a single (H, W, 3) frame
                while v.ndim > 3:
                    v = v[0]
                norm[k] = v
        for k in ("state.joint_pos", "state.gripper_pos"):
            if k in obs:
                v = np.asarray(obs[k], dtype=np.float32)
                while v.ndim > 1:
                    v = v[0]
                norm[k] = v
        task = obs.get("annotation.task", obs.get("task", "Pick up the orange and place it in the bowl."))
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        if isinstance(task, bytes):
            task = task.decode("utf-8", errors="replace")
        norm["annotation.task"] = task

        out = self._policy.infer(norm)
        # Convert (T, 5) + (T, 1) → return both keys
        return {
            "action.joint_pos": out["action.joint_pos"],
            "action.gripper_pos": out["action.gripper_pos"],
        }

    def reset(self) -> dict:
        self._policy.reset()
        return {"ok": True}

    def ping(self) -> dict:
        return {"pong": True, "ckpt": str(self.ckpt_path), "action_horizon": self.action_horizon}


# ----- Server main loop -----

def serve(ckpt_path: str, cfg_dir: str, wan_snap_dir: str, host: str, port: int, action_horizon: int):
    policy = DreamZeroPolicy(
        ckpt_path=ckpt_path, cfg_dir=cfg_dir, wan_snap_dir=wan_snap_dir,
        action_horizon=action_horizon,
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")
    print(f"[DreamZero] server ready on tcp://{host}:{port}", flush=True)

    endpoints = {
        "ping": (policy.ping, False),
        "get_action": (policy.get_action, True),
        "reset": (policy.reset, False),
    }

    while True:
        try:
            message = socket.recv()
            request = MsgSerializer.from_bytes(message)
            endpoint_name = request.get("endpoint", "get_action")

            if endpoint_name not in endpoints:
                socket.send(MsgSerializer.to_bytes({"error": f"unknown endpoint: {endpoint_name}"}))
                continue

            handler, needs_input = endpoints[endpoint_name]
            t0 = time.perf_counter()
            if needs_input:
                data = request.get("data", request)  # tolerate both {endpoint,data} and flat
                # Some clients send `data` flat (not wrapped); strip wire keys first.
                if "endpoint" in data:
                    data = {k: v for k, v in data.items() if k != "endpoint"}
                result = handler(data)
            else:
                result = handler()
            dt = time.perf_counter() - t0
            print(f"[DreamZero] {endpoint_name} {dt*1000:.0f}ms", flush=True)

            socket.send(MsgSerializer.to_bytes(result))
        except Exception as e:
            import traceback
            traceback.print_exc()
            socket.send(MsgSerializer.to_bytes({"error": str(e)}))


def main():
    parser = argparse.ArgumentParser(description="DreamZero policy server for LeIsaac sim eval")
    parser.add_argument("--ckpt-path", required=True,
                        help="path to checkpoint dir containing adapter_model.pt")
    parser.add_argument("--cfg-dir", default="/home/david/work/isaaclab-experience/LeIsaac/outputs/dreamzero-leisaac-so101-lora-r4/experiment_cfg",
                        help="path to experiment_cfg dir (conf.yaml + metadata.json)")
    parser.add_argument("--wan-dir",
                        default="/home/david/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279",
                        help="path to Wan2.1-I2V-14B snapshot dir")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--action-horizon", type=int, default=24)
    args = parser.parse_args()

    serve(args.ckpt_path, args.cfg_dir, args.wan_dir, args.host, args.port, args.action_horizon)


if __name__ == "__main__":
    main()
