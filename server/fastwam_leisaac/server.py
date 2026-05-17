#!/usr/bin/env python3
"""FastWAM (Wan2.2-5B + 1B action expert) demo inference server.

Wire-compatible with LeIsaac ``Pi05ServicePolicyClient``: ZMQ REQ/REP + msgpack
with the ``__ndarray__`` envelope. Two endpoints:

    ping        -> {"status": "ok", "message": "pong"}
    get_action  -> {"status": "ok",
                    "data": {"action.single_arm": (T, 5),
                             "action.gripper":    (T, 1)},
                    "inference_time_ms": float}

Three HF ckpts are fused into one FastWAM instance:
    Wan-AI/Wan2.2-TI2V-5B   -> Video DiT backbone
    Wan-AI/Wan2.1-T2V-1.3B  -> UMT5 text encoder + WanVAE (redirect_common_files)
    yuanty/fastwam          -> trained ActionDiT/Video DiT finetune ckpt

Action-space mismatch (demo only, fine-tune required for real use):
    LIBERO ckpt outputs 7-DoF Franka EEF deltas + gripper:
        [dx, dy, dz, drx, dry, drz, grip]
    SO-101 expects 6-DoF *joint positions*. Cosmetic remap:

        arm_abs = state[:5] + act7[:5] * ARM_DELTA_SCALE   # joint Δ
        grip    = act7[6]                                  # passthrough [0..1]

Run:
    bash server/serve_fastwam.sh                     # default 127.0.0.1:5559
    PORT=5559 bash server/serve_fastwam.sh --detach  # background
"""

from __future__ import annotations

import argparse
import io
import os
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import zmq
from omegaconf import OmegaConf


DEFAULT_CONFIG = "configs/model/fastwam.yaml"
DEFAULT_CKPT = "libero_uncond_2cam224.pt"
DEFAULT_PROMPT = "Pick up the orange and place it on the plate"
DEFAULT_ARM_DELTA_SCALE = 0.05
DEFAULT_ACTION_HORIZON = 24
DEFAULT_INFERENCE_STEPS = 10  # paper uses 20, drop for latency
DEFAULT_IMG_SIZE = 224  # per-camera; concat → (224, 448)


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


def _resize_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop + resize (H,W,3) uint8 -> (size, size, 3) uint8."""
    from PIL import Image
    h, w = img.shape[:2]
    scale = max(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil = Image.fromarray(img.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)
    left = max((new_w - size) // 2, 0)
    top = max((new_h - size) // 2, 0)
    return np.asarray(pil.crop((left, top, left + size, top + size)), dtype=np.uint8)


def _state6_to_proprio8(state6: np.ndarray) -> np.ndarray:
    """SO-101 6-DoF joint state -> LIBERO 8-D proprio (cosmetic pad).

    LIBERO proprio = [eef_pos(3), axisangle(3), gripper_qpos(2)].
    We have joint positions, not EEF; the mapping is meaningless on the base
    ckpt. We put state[:3] in eef_pos slots, state[3:5] in axisangle, and the
    gripper in both qpos slots. Fine-tuning is what makes this useful.
    """
    s = state6.astype(np.float32).ravel()
    proprio = np.zeros(8, dtype=np.float32)
    proprio[0:3] = s[0:3]
    proprio[3:5] = s[3:5]  # leave proprio[5]=0
    proprio[6] = s[5]
    proprio[7] = s[5]
    return proprio


# --- model -------------------------------------------------------------------
class FastWAMDemoServer:
    """FastWAM bf16, 2-cam concat, LIBERO 7-DoF EEF action -> SO-101 6-DoF joint remap."""

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        default_prompt: str = DEFAULT_PROMPT,
        arm_delta_scale: float = DEFAULT_ARM_DELTA_SCALE,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
        img_size: int = DEFAULT_IMG_SIZE,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ) -> None:
        from fastwam.runtime import create_fastwam  # local import → defer GPU init

        self.default_prompt = default_prompt
        self._cached_prompt: str | None = None
        self._cached_context: torch.Tensor | None = None
        self._cached_context_mask: torch.Tensor | None = None
        self.arm_delta_scale = float(arm_delta_scale)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.img_size = int(img_size)
        self.device = device
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]

        print(f"[fastwam] loading model from {config_path}", flush=True)
        cfg = OmegaConf.load(config_path)

        # The yaml has `${data.train.processor.action_output_dim}` / proprio_dim
        # interpolations that resolve from the training data cfg. Hard-bake to
        # LIBERO 2cam values so we don't need to drag the full hydra cfg in.
        cfg.proprio_dim = 8
        cfg.video_dit_config.action_dim = 7
        cfg.action_dit_config.action_dim = 7
        # Resolve text_dim / freq_dim / num_heads / attn_head_dim / num_layers cross-refs.
        cfg.action_dit_config.text_dim = int(cfg.video_dit_config.text_dim)
        cfg.action_dit_config.freq_dim = int(cfg.video_dit_config.freq_dim)
        cfg.action_dit_config.num_heads = int(cfg.video_dit_config.num_heads)
        cfg.action_dit_config.attn_head_dim = int(cfg.video_dit_config.attn_head_dim)
        cfg.action_dit_config.num_layers = int(cfg.video_dit_config.num_layers)
        cfg.video_dit_config.use_gradient_checkpointing = False
        cfg.action_dit_config.use_gradient_checkpointing = False

        t0 = time.time()
        # Load Wan2.2 5B DiT loads to GPU in fp32 (transient ~20GB) then converts
        # to bf16 (10GB). If UMT5 (5GB) is loaded in the same call, ActionDiT
        # init OOMs. Workaround: build the model WITHOUT text encoder first, then
        # load UMT5 separately, encode the prompt once, and drop UMT5 to CPU.
        self._tokenizer_model_id = str(cfg.tokenizer_model_id)
        self._tokenizer_max_len = int(cfg.tokenizer_max_len)
        self._torch_dtype = torch_dtype
        self.model = create_fastwam(
            model_id=cfg.model_id,
            tokenizer_model_id=cfg.tokenizer_model_id,
            video_dit_config=cfg.video_dit_config,
            tokenizer_max_len=int(cfg.tokenizer_max_len),
            load_text_encoder=False,
            proprio_dim=int(cfg.proprio_dim),
            action_dit_config=cfg.action_dit_config,
            action_dit_pretrained_path=cfg.get("action_dit_pretrained_path"),
            skip_dit_load_from_pretrain=bool(cfg.get("skip_dit_load_from_pretrain", False)),
            video_scheduler=cfg.video_scheduler,
            action_scheduler=cfg.action_scheduler,
            loss=cfg.get("loss"),
            mot_checkpoint_mixed_attn=False,
            redirect_common_files=bool(cfg.get("redirect_common_files", True)),
            model_dtype=torch_dtype,
            device=device,
        )
        print(f"[fastwam] model built in {time.time()-t0:.1f}s, "
              f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

        if ckpt_path:
            t0 = time.time()
            print(f"[fastwam] loading checkpoint {ckpt_path}", flush=True)
            self.model.load_checkpoint(str(ckpt_path))
            print(f"[fastwam] ckpt loaded in {time.time()-t0:.1f}s, "
                  f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        self.model.eval()

        # Now that DiT + ActionDiT + VAE are in place, attach text encoder on GPU
        # just long enough to encode the prompt, then move it back to CPU to free
        # ~5GB. The prompt is fixed for our SO-101 PickOrange demo.
        self._precompute_prompt_context(default_prompt)

    @torch.no_grad()
    def _precompute_prompt_context(self, prompt: str) -> None:
        """Lazy-load UMT5 + tokenizer on GPU, encode `prompt`, evict UMT5 to CPU.

        Saves ~5GB GPU vs keeping UMT5 resident, since prompts are static.
        """
        from fastwam.models.wan22.helpers.loader import _load_registered_model
        from fastwam.models.wan22.helpers.io import ModelConfig
        from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

        t0 = time.time()
        print(f"[fastwam] loading UMT5 text encoder for prompt: {prompt!r}", flush=True)
        text_cfg = ModelConfig(
            model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors",
        )
        tok_cfg = ModelConfig(
            model_id=self._tokenizer_model_id,
            origin_file_pattern="google/umt5-xxl/",
        )
        text_cfg.download_if_necessary()
        tok_cfg.download_if_necessary()

        # UMT5-xxl is ~5.5B params (11GB bf16) — does not fit on GPU alongside
        # the 13.45GB Wan22 DiT on a 24GB card. Run it on CPU bf16: prompt
        # encoding is ~1s on CPU and only the (1, L, 4096) emb tensor crosses
        # the bus (~1MB).
        text_encoder = _load_registered_model(
            text_cfg.path, "wan_video_text_encoder",
            torch_dtype=self._torch_dtype, device="cpu",
        )
        text_encoder.eval()
        tokenizer = HuggingfaceTokenizer(
            name=tok_cfg.path, seq_len=self._tokenizer_max_len, clean="whitespace",
        )

        ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to("cpu")
        mask_cpu = mask.to("cpu", dtype=torch.bool)
        prompt_emb = text_encoder(ids, mask_cpu)  # (1, L, D), CPU bf16
        # FastWAM zero-pads beyond seq_lens (per original WanTextEncoder logic).
        seq_lens = mask_cpu.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        context = prompt_emb.to(device=self.device, dtype=self._torch_dtype)
        context_mask = torch.ones_like(mask_cpu).to(self.device)

        self._cached_prompt = prompt
        self._cached_context = context.detach()
        self._cached_context_mask = context_mask.detach()

        del text_encoder, tokenizer, ids, mask_cpu, prompt_emb
        torch.cuda.empty_cache()
        print(f"[fastwam] prompt encoded in {time.time()-t0:.1f}s, UMT5 evicted; "
              f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    @torch.no_grad()
    def predict_action(
        self,
        front_img: np.ndarray,
        wrist_img: np.ndarray,
        state6: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Return (T, 6) absolute joint-position action chunk for SO-101."""
        front = _resize_center_crop(front_img, self.img_size)
        wrist = _resize_center_crop(wrist_img, self.img_size)
        rgb = np.concatenate([front, wrist], axis=1)  # (H, 2W, 3)

        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, 2W)
        x = x.to(device=self.device, dtype=self.model.torch_dtype)
        x = x * (2.0 / 255.0) - 1.0

        proprio = torch.from_numpy(_state6_to_proprio8(state6)).unsqueeze(0)
        proprio = proprio.to(device=self.device, dtype=self.model.torch_dtype)

        # Prompt is cached at init; if a client sends a different one, re-encode.
        if prompt != self._cached_prompt or self._cached_context is None:
            self._precompute_prompt_context(prompt)
        context = self._cached_context
        context_mask = self._cached_context_mask

        out = self.model.infer_action(
            prompt=None,
            input_image=x,
            action_horizon=self.action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=self.num_inference_steps,
            text_cfg_scale=1.0,
            seed=0,
            rand_device="cpu",
        )
        act_chunk = out["action"].cpu().numpy().astype(np.float32)  # (T, 7)
        if act_chunk.ndim != 2 or act_chunk.shape[1] != 7:
            raise ValueError(f"unexpected action chunk shape {act_chunk.shape}")

        # cosmetic remap (T,7 EEF Δ) -> (T,6 SO-101 joint absolute)
        arm_delta = act_chunk[:, :5] * self.arm_delta_scale
        arm_abs = state6[:5].astype(np.float32)[None, :] + np.cumsum(arm_delta, axis=0)
        grip = act_chunk[:, 6:7]
        return np.concatenate([arm_abs, grip], axis=1).astype(np.float32)

    def get_action(self, obs: dict) -> dict:
        front = obs.get("video.front")
        wrist = obs.get("video.wrist")
        if front is None or wrist is None:
            raise ValueError(
                f"fastwam server needs both 'video.front' and 'video.wrist'. Got: {sorted(obs)}"
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
def serve(server: FastWAMDemoServer, host: str, port: int) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[fastwam] ready, listening on tcp://{host}:{port}", flush=True)

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
                if step % 5 == 0:
                    a = action["action.single_arm"][0].tolist()
                    g = float(action["action.gripper"][0, 0])
                    print(f"[fastwam] step={step} action5={a} grip={g:.3f} "
                          f"latency={infer_ms:.0f}ms", flush=True)
                continue

            sock.send(msgpack.packb({"status": "error", "message": f"Unknown endpoint: {ep}"}))
        except KeyboardInterrupt:
            print("[fastwam] interrupted, shutting down", flush=True)
            break
        except Exception as e:
            traceback.print_exc()
            try:
                sock.send(msgpack.packb({"status": "error", "message": str(e)}))
            except Exception:
                pass


def _resolve_paths(repo_root: str, config_path: str, ckpt_path: str) -> tuple[str, str]:
    cfg = config_path
    if not os.path.isabs(cfg):
        cfg = str(Path(repo_root) / cfg)
    ckpt = ckpt_path
    if not os.path.isabs(ckpt):
        candidate = Path(
            os.path.expanduser("~/.cache/huggingface/hub/models--yuanty--fastwam")
        )
        snaps = list((candidate / "snapshots").glob("*")) if candidate.exists() else []
        if snaps:
            for s in snaps:
                hit = s / ckpt
                if hit.exists():
                    ckpt = str(hit)
                    break
    return cfg, ckpt


def main() -> None:
    repo_root = os.environ.get("FASTWAM_REPO_ROOT", os.path.expanduser("~/work/fastwam-repo"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5559)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="path under fastwam-repo or absolute")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="ckpt filename in yuanty/fastwam, or absolute path")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--arm-delta-scale", type=float, default=DEFAULT_ARM_DELTA_SCALE)
    ap.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    ap.add_argument("--num-inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    config_path, ckpt_path = _resolve_paths(repo_root, args.config, args.ckpt)
    # CWD must be fastwam repo so that ActionDiT pretrained path resolves.
    os.chdir(repo_root)

    server = FastWAMDemoServer(
        config_path=config_path,
        ckpt_path=ckpt_path,
        default_prompt=args.prompt,
        arm_delta_scale=args.arm_delta_scale,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        img_size=args.img_size,
        dtype=args.dtype,
    )
    serve(server, args.host, args.port)


if __name__ == "__main__":
    main()
