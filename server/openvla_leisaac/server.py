#!/usr/bin/env python3
"""OpenVLA-7B inference server for LeIsaac SO-101 PickOrange.

Wire-compatible with LeIsaac ``Pi05ServicePolicyClient``: ZMQ REQ/REP + msgpack
with the custom ``__ndarray__`` envelope.  Two endpoints:

    ping        -> {"status": "ok", "message": "pong"}
    get_action  -> {"status": "ok", "data": {"action.single_arm": (T, 5),
                                              "action.gripper":    (T, 1)},
                    "inference_time_ms": float}

Modes:
  - **Demo (no --adapter)**: loads `openvla/openvla-7b` 4-bit, uses Bridge
    EEF stats and a naive Δ→joint hack.  Behaviour will be roughly correct
    geometry but is NOT the trained model — use only to sanity-check the
    server wiring.
  - **Finetuned (--adapter <dir>)**: loads our LoRA adapter on top of the
    4-bit base, injects `norm_stats["leisaac"]` from
    `<adapter>/dataset_statistics.json` (or `<adapter>/../dataset_statistics.json`
    for a checkpoint inside an output dir), reads the canonical prompt from
    the same stats file, and emits 6-DOF *joint positions* directly with no
    Δ→joint hack.

Run:
    bash server/serve_openvla.sh                                  # demo
    ADAPTER=...../checkpoint-N bash server/serve_openvla.sh       # finetuned
"""
from __future__ import annotations

# Belt-and-braces patches against the bnb 4-bit + PEFT _named_members tuple
# corruption crash class.  Same fix as train.py — see openvla-floatingpointops-fix
# memory.  Without these, server load crashes inside accelerate.find_tied_parameters
# with random "'Linear' object has no attribute 'set'" / "expected N, got 2".
import transformers.modeling_utils as _mu
import accelerate.utils.modeling as _am
import accelerate.utils as _au
import transformers.integrations.bitsandbytes as _tb
_mu.PreTrainedModel.floating_point_ops = lambda self, inputs, exclude_embeddings=True: 0
_noop_tied = lambda *a, **kw: []
_am.find_tied_parameters = _noop_tied
_au.find_tied_parameters = _noop_tied
_tb.find_tied_parameters = _noop_tied

# bnb 0.46.x removed MatmulLtState.memory_efficient_backward (PEFT 0.11.x still
# references it on 8-bit load).  Patch attr at class level — match train.py.
from bitsandbytes.autograd._functions import MatmulLtState as _MatmulLtState
_MatmulLtState.memory_efficient_backward = False

import argparse
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import msgpack
import numpy as np
import torch
import zmq
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


DEFAULT_MODEL = "openvla/openvla-7b"
DEFAULT_DEMO_UNNORM_KEY = "bridge_orig"
LEISAAC_UNNORM_KEY = "leisaac"
DEFAULT_DEMO_PROMPT = "Pick up the orange and place it on the plate"
DEFAULT_DEMO_ARM_DELTA_SCALE = 0.05  # Bridge EEF Δ (m) → SO-101 joint Δ scale


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


def _resolve_openvla(model):
    """Walk PeftModel → LoraModel → OpenVLAForActionPrediction.

    DO NOT use ``hasattr(model, "norm_stats")`` as a guard — PeftModel's
    ``__getattr__`` forwards reads to the base model so the check is always
    True but *writes* to wrapper attrs don't propagate.  Always traverse.
    """
    inner = model
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    if hasattr(inner, "model"):
        inner = inner.model
    if not hasattr(inner, "predict_action"):
        raise RuntimeError(
            f"Did not land on OpenVLAForActionPrediction; got {type(inner).__name__}"
        )
    return inner


# --- model -------------------------------------------------------------------
class OpenVLAServer:
    """OpenVLA-7B 4-bit NF4 inference server.  Supports demo + finetuned modes."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        adapter_dir: Optional[str] = None,
        unnorm_key: Optional[str] = None,
        default_prompt: Optional[str] = None,
        arm_delta_scale: float = DEFAULT_DEMO_ARM_DELTA_SCALE,
        device: str = "cuda:0",
        quant: str = "8bit",
    ) -> None:
        self.device = device
        self.adapter_dir = adapter_dir
        self.is_finetuned = adapter_dir is not None
        self.arm_delta_scale = arm_delta_scale

        # ----- load base (quant ∈ {4bit, 8bit, bf16}) -----
        # Default switched to 8bit after 4-bit Params4bit + PyTorch 2.3 + PEFT
        # crashes (see openvla_crash_diagnosis HTML).  Match train.py --quant.
        print(f"[openvla] loading {quant} base: {model_name}", flush=True)
        t0 = time.time()
        if quant == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:  # bf16
            bnb_cfg = None
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs = dict(
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if bnb_cfg is not None:
            model_kwargs["quantization_config"] = bnb_cfg
        model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)

        # ----- optional adapter + stats -----
        if self.is_finetuned:
            from peft import PeftModel
            print(f"[openvla] loading LoRA adapter: {adapter_dir}", flush=True)
            model = PeftModel.from_pretrained(model, adapter_dir)

            stats_path = self._locate_stats_file(adapter_dir)
            with open(stats_path) as f:
                raw = json.load(f)
            stats = raw["action"]
            canonical_prompt = raw.get("prompt", "Grab orange and place into plate")
            print(f"[openvla] loaded stats from {stats_path}", flush=True)
            print(f"[openvla] canonical_prompt={canonical_prompt!r}", flush=True)

            inner = _resolve_openvla(model)
            inner.norm_stats = dict(inner.norm_stats) if inner.norm_stats else {}
            inner.norm_stats[LEISAAC_UNNORM_KEY] = {
                "action": {
                    "q01": list(stats["q01"]),
                    "q99": list(stats["q99"]),
                    "mask": list(stats.get("mask", [True] * len(stats["q01"]))),
                }
            }
            self.unnorm_key = unnorm_key or LEISAAC_UNNORM_KEY
            self.default_prompt = default_prompt or canonical_prompt
            # Hard self-check — if this fails, the rest is theatre.
            assert inner.get_action_dim(self.unnorm_key) == len(stats["q01"]), (
                f"action_dim mismatch: get_action_dim('{self.unnorm_key}')="
                f"{inner.get_action_dim(self.unnorm_key)} but stats has "
                f"{len(stats['q01'])} dims"
            )
            print(f"[openvla] action_dim self-check passed "
                  f"(dim={inner.get_action_dim(self.unnorm_key)}) ✓", flush=True)
        else:
            self.unnorm_key = unnorm_key or DEFAULT_DEMO_UNNORM_KEY
            self.default_prompt = default_prompt or DEFAULT_DEMO_PROMPT

        self.vla = model
        self.inner = _resolve_openvla(model)

        # Cache vision conv dtype so we don't recompute per request
        try:
            self._pixel_dtype = self.inner.vision_backbone.featurizer.patch_embed.proj.weight.dtype
        except AttributeError:
            self._pixel_dtype = torch.bfloat16

        print(
            f"[openvla] loaded in {time.time()-t0:.1f}s, "
            f"finetuned={self.is_finetuned}  unnorm_key={self.unnorm_key}  "
            f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB",
            flush=True,
        )

    @staticmethod
    def _locate_stats_file(adapter_dir: str) -> Path:
        """Find dataset_statistics.json — first in the adapter dir, then in
        its parent (for output_dir/checkpoint-N layouts)."""
        adapter = Path(adapter_dir)
        for cand in (adapter / "dataset_statistics.json",
                     adapter.parent / "dataset_statistics.json"):
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"No dataset_statistics.json found at {adapter / 'dataset_statistics.json'} "
            f"or {adapter.parent / 'dataset_statistics.json'}"
        )

    def predict_action(
        self,
        front_img: np.ndarray,
        state6: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Return (1, 6) absolute joint-position action for SO-101.

        Finetuned mode: model predicts 6 joint positions directly, unnorm via
        our `leisaac` q01/q99 stats.  Demo mode: Bridge 7-DOF EEF with a naive
        Δ→joint hack (will drift; only to verify server wiring).
        """
        img = Image.fromarray(front_img.astype(np.uint8))
        fmt_prompt = f"In: What action should the robot take to {prompt.strip().rstrip('.')}?\nOut:"
        inputs = self.processor(fmt_prompt, img).to(self.device)
        if inputs["pixel_values"].dtype != self._pixel_dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._pixel_dtype)

        act = self.inner.predict_action(
            **inputs, unnorm_key=self.unnorm_key, do_sample=False
        )
        act = np.asarray(act, dtype=np.float32).flatten()

        if self.is_finetuned:
            # Direct 6-DOF joint positions
            assert act.shape[0] == 6, f"finetuned predict_action returned {act.shape}, expected 6"
            return act[None]  # (1, 6)

        # Demo mode: Bridge EEF Δ + naive scale into current joint frame
        assert act.shape[0] == 7, f"demo predict_action returned {act.shape}, expected 7 (Bridge EEF+grip)"
        arm_delta = act[:5] * self.arm_delta_scale
        arm_abs = state6[:5].astype(np.float32) + arm_delta
        grip = np.array([float(act[6])], dtype=np.float32)
        return np.concatenate([arm_abs, grip])[None]

    # --- wire-compat hook for Pi05ServicePolicyClient ------------------------
    def get_action(self, obs: dict) -> dict:
        front = obs.get("video.front")
        if front is None:
            raise ValueError(
                f"OpenVLA server needs 'video.front'. Got keys: {sorted(obs)}"
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
def serve(server: OpenVLAServer, host: str, port: int) -> None:
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
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir (a HF Trainer checkpoint).  When set, "
                         "loads stats + canonical prompt from <adapter>/dataset_statistics.json "
                         "(falling back to <adapter>/../dataset_statistics.json).")
    ap.add_argument("--unnorm-key", default=None,
                    help="Override unnorm_key.  Default: 'leisaac' if --adapter, else 'bridge_orig'.")
    ap.add_argument("--prompt", default=None,
                    help="Fallback prompt when client omits annotation.human.task_description.  "
                         "Default: canonical prompt from stats file in finetuned mode, "
                         "or the demo prompt otherwise.")
    ap.add_argument("--arm-delta-scale", type=float, default=DEFAULT_DEMO_ARM_DELTA_SCALE,
                    help="Demo mode only: Bridge EEF Δ → joint Δ scale.")
    ap.add_argument("--quant", choices=["4bit", "8bit", "bf16"], default="8bit",
                    help="Base precision (must match training).  Default 8bit "
                         "after 4-bit Params4bit crash class — see crash_diagnosis HTML.")
    args = ap.parse_args()

    server = OpenVLAServer(
        model_name=args.model_name,
        adapter_dir=args.adapter,
        unnorm_key=args.unnorm_key,
        default_prompt=args.prompt,
        quant=args.quant,
        arm_delta_scale=args.arm_delta_scale,
    )
    serve(server, args.host, args.port)


if __name__ == "__main__":
    main()
