#!/usr/bin/env python3
"""X-VLA inference server for LeIsaac SO-101 PickOrange.

Wire-compatible with LeIsaac ``Pi05ServicePolicyClient``: ZMQ REQ/REP + msgpack
with the custom ``__ndarray__`` envelope.  Endpoints:

    ping        -> {"status": "ok", "message": "pong"}
    reset       -> clears the chunked-action queue (call between episodes)
    get_action  -> {"status": "ok", "data": {"action.single_arm": (1, 5),
                                              "action.gripper":    (1, 1)},
                    "inference_time_ms": float}

Loads a LeRobot XVLAPolicy checkpoint (Florence2 + SoftPromptedTransformer +
rectified-flow head, 0.9B params).  Uses the training-time preprocessor /
postprocessor saved alongside the checkpoint so normalization stats are exact.

Trained with action_mode=auto (real_dim=6, max_dim=20); the AutoActionSpace
post-processes back to 6 dims [shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper] before exiting the model.

Camera-key rename:
  client sends 'video.front'/'video.wrist'  → fed as
  'observation.images.image'/'observation.images.image2'.  image3 is auto-filled
  via empty_cameras=1 baked into the model config.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Side-effect import: registers SingleArmSO101ActionSpace into lerobot's
# ACTION_REGISTRY so XVLAPolicy.from_pretrained can resolve our trained ckpts
# (config.json action_mode='so101_single').  Kept in our repo (not the lerobot
# submodule) so the submodule stays patch-free.
_LEISAAC_XVLA = Path(__file__).resolve().parent.parent.parent / "LeIsaac" / "scripts" / "finetune" / "xvla"
if str(_LEISAAC_XVLA) not in sys.path:
    sys.path.insert(0, str(_LEISAAC_XVLA))
import action_spaces  # noqa: F401  ← registers so101_single

import msgpack
import numpy as np
import torch
import zmq


DEFAULT_PROMPT = "Grab orange and place into plate"


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
class XVLAServer:
    """X-VLA ckpt inference server."""

    def __init__(
        self,
        ckpt_dir: str,
        default_prompt: str = DEFAULT_PROMPT,
        device: str = "cuda",
        n_action_steps_override: int | None = None,
        ema_alpha: float | None = None,
        tae_buffer: int | None = None,
        tae_m: float = 0.1,
        denoising_steps_override: int | None = None,
    ) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        self.device = device
        self.default_prompt = default_prompt
        self.real_action_dim = 6
        # EMA smoothing: out_t = α·new + (1-α)·prev.  None = disabled (Opencode #2).
        self.ema_alpha = ema_alpha
        self._ema_prev: np.ndarray | None = None
        if ema_alpha is not None:
            print(f"[xvla] EMA action smoothing enabled, alpha={ema_alpha}", flush=True)

        # TAE (Temporal Action Ensembling, ALOHA 2304.13705 §4.3).
        # Each step: predict full chunk, buffer last K chunks, ensemble all
        # predictions for current step with exponential weights w_i = exp(-m·age).
        # Bypasses select_action queue entirely.  None = disabled.
        self.tae_buffer_size = tae_buffer
        self.tae_m = tae_m
        # buffer entries: (start_step:int, chunk:np.ndarray of shape (chunk_size, action_dim))
        self._tae_buffer: list[tuple[int, np.ndarray]] = []
        self._tae_step = 0
        if tae_buffer is not None:
            print(f"[xvla] TAE enabled, buffer={tae_buffer} m={tae_m}", flush=True)

        t0 = time.time()
        print(f"[xvla] loading config from {ckpt_dir}", flush=True)
        self.policy_cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
        self.policy_cfg.device = device
        # Inference-time override: smaller n_action_steps reduces chunk-staleness
        # (re-plans more frequently); useful for diagnosing pick-but-no-place
        # failures where the model commits to a stale chunk after grasp.
        if n_action_steps_override is not None:
            print(
                f"[xvla] OVERRIDE n_action_steps: {self.policy_cfg.n_action_steps} → "
                f"{n_action_steps_override}",
                flush=True,
            )
            self.policy_cfg.n_action_steps = n_action_steps_override
        if denoising_steps_override is not None:
            print(
                f"[xvla] OVERRIDE num_denoising_steps: {self.policy_cfg.num_denoising_steps} → "
                f"{denoising_steps_override}",
                flush=True,
            )
            self.policy_cfg.num_denoising_steps = denoising_steps_override

        print("[xvla] loading XVLAPolicy weights", flush=True)
        self.policy = XVLAPolicy.from_pretrained(ckpt_dir, config=self.policy_cfg)
        self.policy.to(device)
        self.policy.eval()
        # Also patch the loaded policy's config (config is shared via .config attr
        # but the loaded one may have been re-read from disk during from_pretrained).
        if n_action_steps_override is not None:
            self.policy.config.n_action_steps = n_action_steps_override
        if denoising_steps_override is not None:
            self.policy.config.num_denoising_steps = denoising_steps_override

        print("[xvla] loading pre/post processors", flush=True)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=ckpt_dir,
            preprocessor_overrides={
                "device_processor": {"device": device},
            },
        )

        print(
            f"[xvla] loaded in {time.time()-t0:.1f}s  "
            f"n_action_steps={self.policy_cfg.n_action_steps}  "
            f"chunk_size={self.policy_cfg.chunk_size}  "
            f"denoise_steps={self.policy_cfg.num_denoising_steps}  "
            f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB",
            flush=True,
        )

    def reset_queues(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._ema_prev = None
        self._tae_buffer = []
        self._tae_step = 0

    def predict_action(
        self,
        front: np.ndarray,
        wrist: np.ndarray,
        state6: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        # uint8 HWC → uint8 CHW tensor; preprocessor scales to float + ImageNet-normalizes.
        def hwc_to_chw_u8(img: np.ndarray) -> torch.Tensor:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        batch = {
            "observation.images.image":  hwc_to_chw_u8(front),
            "observation.images.image2": hwc_to_chw_u8(wrist),
            "observation.state":         torch.from_numpy(state6.astype(np.float32)).unsqueeze(0),
            "task":                      [prompt],
        }

        batch = self.preprocessor(batch)

        with torch.inference_mode():
            if self.tae_buffer_size is not None:
                # TAE path: get full chunk every step, ensemble with buffered chunks.
                chunk = self.policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
                chunk_post = self.postprocessor(chunk.reshape(-1, chunk.shape[-1]))
                chunk_np = chunk_post.detach().float().cpu().numpy().reshape(chunk.shape[1], -1)
                if chunk_np.shape[1] > self.real_action_dim:
                    chunk_np = chunk_np[:, : self.real_action_dim]
                t = self._tae_step
                self._tae_buffer.append((t, chunk_np))
                if len(self._tae_buffer) > self.tae_buffer_size:
                    self._tae_buffer.pop(0)
                preds, weights = [], []
                cs = chunk_np.shape[0]
                for s_i, c_i in self._tae_buffer:
                    idx = t - s_i
                    if 0 <= idx < cs:
                        preds.append(c_i[idx])
                        weights.append(float(np.exp(-self.tae_m * idx)))
                preds = np.asarray(preds)
                weights = np.asarray(weights)
                weights = weights / weights.sum()
                act = (preds * weights[:, None]).sum(axis=0)
                self._tae_step += 1
                # Skip EMA when TAE enabled (orthogonal smoothers).
                return act[None]
            act = self.policy.select_action(batch)  # tensor (1, ?)

        # postprocessor un-normalizes the action back to physical units (joint deg).
        act = self.postprocessor(act)
        act = act.detach().float().cpu().numpy().reshape(-1)

        # AutoActionSpace already trimmed to real_dim, but be defensive.
        if act.shape[0] > self.real_action_dim:
            act = act[: self.real_action_dim]
        if self.ema_alpha is not None:
            if self._ema_prev is None:
                self._ema_prev = act.copy()
            else:
                act = self.ema_alpha * act + (1.0 - self.ema_alpha) * self._ema_prev
                self._ema_prev = act.copy()
        return act[None]  # (1, 6)

    # --- wire-compat hook ---------------------------------------------------
    def get_action(self, obs: dict) -> dict:
        front = obs.get("video.front")
        wrist = obs.get("video.wrist")
        if front is None or wrist is None:
            raise ValueError(
                f"XVLA server needs 'video.front' and 'video.wrist'. Got keys: {sorted(obs)}"
            )
        if isinstance(front, np.ndarray) and front.ndim == 4:
            front = front[0]
        if isinstance(wrist, np.ndarray) and wrist.ndim == 4:
            wrist = wrist[0]

        arm5 = np.asarray(obs.get("state.single_arm", np.zeros(5))).ravel()
        grip1 = np.asarray(obs.get("state.gripper", np.zeros(1))).ravel()
        state6 = np.concatenate([arm5, grip1]).astype(np.float32)

        task = obs.get("annotation.human.task_description", self.default_prompt)
        if isinstance(task, list) and task:
            task = task[0]
        if isinstance(task, bytes):
            task = task.decode()

        actions = self.predict_action(front, wrist, state6, str(task))
        return {
            "action.single_arm": actions[:, :5].astype(np.float32),
            "action.gripper":    actions[:, 5:6].astype(np.float32),
        }


# --- ZMQ loop ----------------------------------------------------------------
def serve(server: XVLAServer, host: str, port: int) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[xvla] ready, listening on tcp://{host}:{port}", flush=True)

    step = 0
    while True:
        try:
            raw = sock.recv()
            req = msgpack.unpackb(raw, raw=False)
            ep = req.get("endpoint", "")

            if ep == "ping":
                sock.send(msgpack.packb({"status": "ok", "message": "pong"}))
                continue

            if ep == "reset":
                server.reset_queues()
                sock.send(msgpack.packb({"status": "ok", "message": "reset"}))
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
                        f"[xvla] step={step} "
                        f"action6={action['action.single_arm'][0].tolist()} "
                        f"grip={float(action['action.gripper'][0,0]):.3f} "
                        f"latency={infer_ms:.0f}ms",
                        flush=True,
                    )
                continue

            sock.send(msgpack.packb({"status": "error", "message": f"Unknown endpoint: {ep}"}))
        except KeyboardInterrupt:
            print("[xvla] interrupted, shutting down", flush=True)
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
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--ckpt", required=True,
                    help="Path to a LeRobot ckpt dir, "
                         "e.g. .../checkpoints/last/pretrained_model")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="Fallback prompt when client omits annotation.human.task_description")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="Override config.n_action_steps for inference. "
                         "Smaller value (e.g. 1) forces re-planning every step, "
                         "useful for diagnosing chunk-staleness in chunked policies.")
    ap.add_argument("--ema-alpha", type=float, default=None,
                    help="Action smoothing EMA: out = α·new + (1-α)·prev. "
                         "Lower α = more smoothing.  Cleared on reset().")
    ap.add_argument("--tae-buffer", type=int, default=None,
                    help="Temporal Action Ensembling buffer size K (ALOHA 2304.13705). "
                         "Bypasses select_action: every step computes full chunk, "
                         "ensembles last K chunks via exp weights.")
    ap.add_argument("--tae-m", type=float, default=0.1,
                    help="TAE exponential decay rate (default 0.1). "
                         "Higher m = newer chunks weight more.")
    ap.add_argument("--num-denoising-steps", type=int, default=None,
                    help="Override RF head denoising steps (default 10 from config).")
    args = ap.parse_args()

    server = XVLAServer(
        ckpt_dir=args.ckpt,
        default_prompt=args.prompt,
        n_action_steps_override=args.n_action_steps,
        ema_alpha=args.ema_alpha,
        tae_buffer=args.tae_buffer,
        tae_m=args.tae_m,
        denoising_steps_override=args.num_denoising_steps,
    )
    serve(server, args.host, args.port)


if __name__ == "__main__":
    main()
