#!/usr/bin/env python3
"""π0.5 PyTorch inference server.

Speaks the same ZMQ REQ/REP + msgpack wire protocol the LeIsaac
``Pi05ServicePolicyClient`` expects (custom ``__ndarray__`` envelope, two
endpoints: ``ping`` and ``get_action``).

Reuses the full **lerobot pre/post-processor pipeline** so train and
inference share one contract:

  1. real PaliGemma tokenizer on the task prompt (no zero-token hack)
  2. real state input via ``Pi05PrepareStateTokenizerProcessorStep``
     (6-DOF joint → 256-bin discretization → spliced onto tokens)
  3. ``NormalizerProcessorStep`` (QUANTILES) before the model, and
     ``UnnormalizerProcessorStep`` after — actions come out in real
     joint units, not normalized ``[-1, 1]``.

Usage:
    pi05-server \\
        --lora-npz path/to/final_lora.npz \\
        --dataset-root path/to/leisaac-pick-orange  # for stats + features

(The dataset root only has to be reachable at startup; we use its
``meta.stats`` for the normalizer and ``meta.features`` for the policy.
No frames are loaded.)
"""

from __future__ import annotations

import argparse
import io
import os as _os
import sys
import time
import traceback
from pathlib import Path

import msgpack
import numpy as np
import torch
import zmq

_LEROBOT_SRC = _os.environ.get(
    "LEROBOT_SRC", str(Path.home() / "work/lerobot-experience/lerobot/src")
)
if _os.path.isdir(_LEROBOT_SRC) and _LEROBOT_SRC not in sys.path:
    sys.path.insert(0, _LEROBOT_SRC)

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402

from .lora import load_lora_npz  # noqa: E402


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


def _unpack_ndarray(obj) -> np.ndarray:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.load(io.BytesIO(obj["data"]), allow_pickle=False)
    return np.array(obj)


# --- model wrapper -----------------------------------------------------------
class Pi05TorchServer:
    """PI05Policy + lerobot pre/post processors + trained LoRA."""

    DEFAULT_PROMPT = "Pick up the orange and place it on the plate"

    def __init__(
        self,
        *,
        policy_id: str,
        lora_npz: str,
        dataset_repo_id: str,
        dataset_root: str | None,
        device: str,
        dtype: torch.dtype,
        default_prompt: str | None = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.default_prompt = default_prompt or self.DEFAULT_PROMPT

        # --- dataset metadata: features + normalization stats -------------
        print(f"[pi05-pt] loading dataset metadata {dataset_repo_id} "
              f"(root={dataset_root or 'HF cache'}) ...", flush=True)
        ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
        # All camera keys the model was trained on. Anything we don't
        # forward gets zero-padded with mask=False inside _preprocess_images,
        # which is a train/test mismatch — so we send every camera we got.
        self.model_camera_keys = [
            k for k in ds_meta.features if k.startswith("observation.images.")
        ]
        # Map wire keys (client uses "video.<cam>") → model keys.
        # Convention: dataset key "observation.images.<cam>" ↔ wire key "video.<cam>".
        self.wire_to_model_img: dict[str, str] = {
            f"video.{k.rsplit('.', 1)[-1]}": k for k in self.model_camera_keys
        }
        print(f"[pi05-pt]   model camera keys: {self.model_camera_keys}", flush=True)
        print(f"[pi05-pt]   wire→model image map: {self.wire_to_model_img}", flush=True)

        # --- policy: make_policy populates input_features from ds_meta ----
        cfg = PI05Config()
        cfg.pretrained_path = policy_id
        cfg.device = str(self.device)
        print(f"[pi05-pt] loading {policy_id} ...", flush=True)
        self.policy = make_policy(cfg, ds_meta=ds_meta).to(
            device=self.device, dtype=dtype
        ).eval()

        # lerobot's denoise_step hardcodes fp32 for the diffusion timestep;
        # patch when serving in bf16/fp16 so it casts to model dtype.
        if dtype != torch.float32:
            self._patch_dtype_entrypoints(dtype)

        # --- LoRA injection -------------------------------------------------
        print(f"[pi05-pt] applying LoRA from {lora_npz} ...", flush=True)
        report = load_lora_npz(self.policy, lora_npz)
        print(
            f"[pi05-pt]   loaded={len(report['loaded'])} "
            f"missing={len(report['missing'])} skipped={len(report['skipped'])}",
            flush=True,
        )
        if report["missing"]:
            raise RuntimeError(f"LoRA layers missing weights: {report['missing'][:5]}...")

        # --- pre/post processors (uses dataset_stats for QUANTILES) -------
        print("[pi05-pt] building pre/post processors ...", flush=True)
        self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(
            self.policy.config, dataset_stats=ds_meta.stats
        )

        # --- warm up + free cached blocks for Isaac Sim -------------------
        print("[pi05-pt] warming up ...", flush=True)
        self._warmup()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 2**20
            res = torch.cuda.memory_reserved() / 2**20
            print(f"[pi05-pt]   gpu: allocated={mem:.0f}MiB reserved={res:.0f}MiB", flush=True)
        print("[pi05-pt] ready", flush=True)

    def _patch_dtype_entrypoints(self, dtype: torch.dtype) -> None:
        model = self.policy.model
        orig_denoise = model.denoise_step

        def denoise_step_cast(*, prefix_pad_masks, past_key_values, x_t, timestep):
            if timestep.dtype != dtype:
                timestep = timestep.to(dtype)
            if x_t.dtype != dtype:
                x_t = x_t.to(dtype)
            return orig_denoise(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )

        model.denoise_step = denoise_step_cast

    @torch.no_grad()
    def _warmup(self) -> None:
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_state = np.zeros(6, dtype=np.float32)
        obs = {
            "state.single_arm": dummy_state[:5],
            "state.gripper": dummy_state[5:6],
            "annotation.human.task_description": [self.default_prompt],
        }
        for wire_key in self.wire_to_model_img:
            obs[wire_key] = dummy_img
        for _ in range(2):
            t0 = time.time()
            self.get_action(obs)
            print(f"[pi05-pt]   warmup inf: {1000 * (time.time() - t0):.0f}ms", flush=True)

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def get_action(self, obs: dict) -> dict:
        """Run one chunk inference. ``obs`` uses the client's wire keys.

        Required keys (single-arm SO-101):
            video.front, video.wrist:   (H, W, 3) uint8 (at least one)
            state.single_arm:           (5,) float32
            state.gripper:              (1,) float32
            annotation.human.task_description: [str] (or str)
        """
        # 1) images: forward every camera the model was trained on. Anything
        # missing on the wire would be zero-padded with mask=False inside
        # _preprocess_images, which is a train/test mismatch — refuse instead.
        batch: dict[str, object] = {}
        for wire_key, model_key in self.wire_to_model_img.items():
            arr = obs.get(wire_key)
            if arr is None:
                continue
            img = torch.from_numpy(np.ascontiguousarray(arr))
            if img.ndim == 4:
                img = img.squeeze(0)
            if img.dtype != torch.float32:
                img = img.to(torch.float32) / 255.0
            if img.ndim == 3 and img.shape[-1] == 3:  # HWC → CHW (SigLIP conv)
                img = img.permute(2, 0, 1).contiguous()
            batch[model_key] = img
        if not batch:
            raise ValueError(
                f"No camera images in observation. Expected at least one of "
                f"{list(self.wire_to_model_img)}; got keys={list(obs)}"
            )

        # 2) state: concat single_arm (5,) + gripper (1,) → (6,) float32.
        arm = obs.get("state.single_arm")
        grip = obs.get("state.gripper")
        if arm is None or grip is None:
            raise ValueError("Missing 'state.single_arm' / 'state.gripper'")
        state = np.concatenate([np.asarray(arm).ravel(), np.asarray(grip).ravel()])
        batch["observation.state"] = torch.from_numpy(state.astype(np.float32))

        # 3) task: prefer client's annotation, fall back to default prompt.
        task_field = obs.get("annotation.human.task_description")
        if isinstance(task_field, list) and task_field:
            task_str = str(task_field[0])
        elif isinstance(task_field, (bytes, str)):
            task_str = task_field.decode() if isinstance(task_field, bytes) else task_field
        else:
            task_str = self.default_prompt
        batch["task"] = task_str

        # 4) feed through lerobot preprocessor (rename → batchify → normalize
        #    → Pi05 state-tokenize → PaliGemma tokenize → to device).
        batch = self.preprocessor(batch)

        # 5) inference: predict_action_chunk handles _preprocess_images +
        #    sample_actions + 6-dim truncation, returns (B, T, 6) normalized.
        chunk = self.policy.predict_action_chunk(batch)

        # 6) postprocessor: unnormalize (QUANTILES inverse) + abs actions +
        #    move to cpu. PolicyAction pipeline expects a tensor in/out.
        chunk = self.postprocessor(chunk)

        actions_np = chunk[0].to(torch.float32).cpu().numpy()  # (T, 6)
        return {
            "action.single_arm": actions_np[:, :5].astype(np.float32),
            "action.gripper": actions_np[:, 5:6].astype(np.float32),
        }


# --- ZMQ server loop --------------------------------------------------------
def serve(server: Pi05TorchServer, host: str, port: int) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[pi05-pt] listening on tcp://{host}:{port}", flush=True)

    while True:
        try:
            raw = sock.recv()
            request = msgpack.unpackb(raw, raw=False)
            endpoint = request.get("endpoint", "")

            if endpoint == "ping":
                sock.send(msgpack.packb({"status": "ok", "message": "pong"}))
                continue

            if endpoint == "get_action":
                obs_data = request.get("data", {}) or {}
                obs = {
                    k: (_unpack_ndarray(v) if isinstance(v, dict) and v.get("__ndarray__") else v)
                    for k, v in obs_data.items()
                }
                t0 = time.time()
                action = server.get_action(obs)
                infer_ms = 1000 * (time.time() - t0)

                data = {
                    k: (_pack_ndarray(v) if isinstance(v, np.ndarray) else v)
                    for k, v in action.items()
                }
                sock.send(
                    msgpack.packb(
                        {"status": "ok", "data": data, "inference_time_ms": infer_ms}
                    )
                )
                print(f"[pi05-pt]   action: {infer_ms:.0f}ms", flush=True)
                continue

            sock.send(
                msgpack.packb({"status": "error", "message": f"Unknown endpoint: {endpoint}"})
            )

        except KeyboardInterrupt:
            print("\n[pi05-pt] shutting down", flush=True)
            break
        except Exception as e:
            print(f"[pi05-pt] error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            try:
                sock.send(msgpack.packb({"status": "error", "message": str(e)}))
            except Exception:
                pass

    sock.close()
    ctx.term()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-id", default="lerobot/pi05_base")
    parser.add_argument(
        "--lora-npz",
        required=True,
        help="LoRA weights .npz (e.g. outputs/.../final_lora.npz)",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="LightwheelAI/leisaac-pick-orange",
        help="HF dataset id (used for input_features + normalization stats)",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Local v3.0 dataset root (skips HF fetch if present)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--default-prompt",
        default=None,
        help="Fallback task prompt when the client doesn't send annotation.human.task_description",
    )
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    server = Pi05TorchServer(
        policy_id=args.policy_id,
        lora_npz=args.lora_npz,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        device=args.device,
        dtype=dtype,
        default_prompt=args.default_prompt,
    )
    serve(server, args.host, args.port)


if __name__ == "__main__":
    main()
