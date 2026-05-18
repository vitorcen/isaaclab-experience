#!/usr/bin/env python3
"""FastWAM QLoRA inference server (post-finetune).

Same ZMQ wire as `server.py` (bf16 demo) — drop-in for eval_pi05.sh.

Difference vs bf16 demo:
  - proprio_dim=6, action_dim=6 (SO-101 native, not LIBERO 7-D EEF)
  - Apply manual LoRA + NF4 quant before loading checkpoint
  - Load LoRA-only state from training output dir, filtering bnb buffer keys
    (bnb Params4bit.quant_state corrupts in safetensors round-trip — see
    fastwam-qlora-finetune memory)
  - No EEF→joint cosmetic remap (model was trained on joint deltas directly)
"""

from __future__ import annotations

# Eager torchgen import — same race-mitigation as the trainer.
import torchgen.model as _torchgen_model  # noqa: F401

import argparse
import io
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import zmq
from omegaconf import OmegaConf


# Make `fastwam_qlora` importable (same path as the training scripts).
_FT_DIR = "/home/david/work/isaaclab-experience/LeIsaac/scripts/finetune"
if _FT_DIR not in sys.path:
    sys.path.insert(0, _FT_DIR)


DEFAULT_CONFIG = "configs/model/fastwam.yaml"
DEFAULT_CKPT_DIR = "/home/david/work/fastwam-repo/runs/train/fastwam_qlora_pickorange_5phase/phase2/checkpoints/state/step_004000"
DEFAULT_PROMPT = "Grab orange and place into plate"
DEFAULT_ARM_DELTA_SCALE = 0.05
DEFAULT_ACTION_HORIZON = 32  # MUST match training (num_frames-1=32 divisibility constraint)
DEFAULT_INFERENCE_STEPS = 10
DEFAULT_IMG_SIZE = 224


# --- wire format -------------------------------------------------------------
def _pack_ndarray(arr: np.ndarray) -> dict:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return {"__ndarray__": True, "data": buf.getvalue(),
            "dtype": str(arr.dtype), "shape": arr.shape}


def _unpack_ndarray(obj: Any) -> np.ndarray:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.load(io.BytesIO(obj["data"]), allow_pickle=False)
    return np.array(obj)


def _resize_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    h, w = img.shape[:2]
    scale = max(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil = Image.fromarray(img.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)
    left = max((new_w - size) // 2, 0)
    top = max((new_h - size) // 2, 0)
    return np.asarray(pil.crop((left, top, left + size, top + size)), dtype=np.uint8)


class FastWAMQLoRAServer:
    """QLoRA bf16+NF4 inference, SO-101 6-DoF native action space."""

    def __init__(
        self,
        config_path: str,
        ckpt_dir: str,
        default_prompt: str = DEFAULT_PROMPT,
        arm_delta_scale: float = DEFAULT_ARM_DELTA_SCALE,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
        img_size: int = DEFAULT_IMG_SIZE,
        device: str = "cuda:0",
        stats_path: str | None = None,
    ) -> None:
        from fastwam.runtime import create_fastwam
        from fastwam_qlora.qlora_utils import apply_lora  # NF4 quant intentionally skipped at inference (route C)

        self.default_prompt = default_prompt
        self._cached_prompt: str | None = None
        self._cached_context: torch.Tensor | None = None
        self._cached_context_mask: torch.Tensor | None = None
        self.arm_delta_scale = float(arm_delta_scale)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.img_size = int(img_size)
        self.device = device
        self._torch_dtype = torch.bfloat16
        # Normalizer scale/offset (filled by _load_normalizer_stats); both are
        # 1-D float32 arrays shape (6,).  forward (raw → norm):
        #     x_norm = x_raw * scale + offset, clamp [-5, 5]
        # backward (norm → raw):  x_raw = (x_norm - offset) / scale
        self._state_scale: np.ndarray | None = None
        self._state_offset: np.ndarray | None = None
        self._action_scale: np.ndarray | None = None
        self._action_offset: np.ndarray | None = None
        self._stats_path = stats_path

        print(f"[fastwam-qlora] config={config_path}", flush=True)
        cfg = OmegaConf.load(config_path)

        # SO-101 6-DoF (not LIBERO 7/8)
        cfg.proprio_dim = 6
        cfg.video_dit_config.action_dim = 6
        cfg.action_dit_config.action_dim = 6
        cfg.action_dit_config.text_dim = int(cfg.video_dit_config.text_dim)
        cfg.action_dit_config.freq_dim = int(cfg.video_dit_config.freq_dim)
        cfg.action_dit_config.num_heads = int(cfg.video_dit_config.num_heads)
        cfg.action_dit_config.attn_head_dim = int(cfg.video_dit_config.attn_head_dim)
        cfg.action_dit_config.num_layers = int(cfg.video_dit_config.num_layers)
        cfg.video_dit_config.use_gradient_checkpointing = False
        cfg.action_dit_config.use_gradient_checkpointing = False

        self._tokenizer_model_id = str(cfg.tokenizer_model_id)
        self._tokenizer_max_len = int(cfg.tokenizer_max_len)

        t0 = time.time()
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
            model_dtype=self._torch_dtype,
            device=device,
        )
        print(f"[fastwam-qlora] model built in {time.time()-t0:.1f}s, "
              f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

        # Route C inference: LoRA wrap ONLY, no NF4 quant — base stays bf16.
        # Saved ckpt's 4bit packed weights are manually dequantized below
        # before load_state_dict, restoring trained bf16 weights cleanly.
        t0 = time.time()
        self.model.dit = apply_lora(self.model.dit, r=16, alpha=16, dropout=0.0,
                                    target_modules=("q", "k", "v", "o"))
        torch.cuda.empty_cache()
        print(f"[fastwam-qlora] LoRA wrap (bf16, no quant) in {time.time()-t0:.1f}s, "
              f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

        # Load trained LoRA weights from checkpoint dir.
        ckpt_dir_p = Path(ckpt_dir)
        if ckpt_dir_p.is_file():
            ckpt_path = ckpt_dir_p
        else:
            ckpt_path = ckpt_dir_p / "model.safetensors"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no model.safetensors at {ckpt_dir}")
        t0 = time.time()
        self._load_lora_weights(ckpt_path)
        print(f"[fastwam-qlora] LoRA ckpt loaded in {time.time()-t0:.1f}s", flush=True)

        self.model.eval()
        self._load_normalizer_stats(ckpt_dir)
        self._precompute_prompt_context(default_prompt)

    def _load_normalizer_stats(self, ckpt_dir: str) -> None:
        """Load `dataset_stats.json` produced by training and pre-compute
        per-dim min/max → [-1, 1] scale/offset for state and action.

        Without this the server was feeding RAW degrees into a model that
        expects [-1, 1] normalized input — model sees OOD 50×, outputs in
        [-1, 1] get treated as raw, robot stuck micro-vibrating.
        """
        from pathlib import Path
        import json

        if self._stats_path:
            stats_path = Path(self._stats_path)
        else:
            # ckpt_dir = .../phaseN/checkpoints/state/step_XXXXXX
            # walk up to phaseN/ which contains dataset_stats.json
            p = Path(ckpt_dir).resolve()
            stats_path = None
            for parent in [p] + list(p.parents):
                cand = parent / "dataset_stats.json"
                if cand.exists():
                    stats_path = cand
                    break
            if stats_path is None:
                raise FileNotFoundError(
                    f"dataset_stats.json not found near ckpt_dir={ckpt_dir}; "
                    f"pass --stats-path explicitly"
                )

        with open(stats_path, "r") as f:
            ds = json.load(f)

        def _make(stats_field):
            mn = np.asarray(stats_field["global_min"], dtype=np.float32)
            mx = np.asarray(stats_field["global_max"], dtype=np.float32)
            rng = mx - mn
            rng[rng < 1e-4] = 2.0  # ignore_dim → map to constant 0
            scale = 2.0 / rng
            offset = -1.0 - 2.0 * mn / rng
            return scale, offset

        self._state_scale, self._state_offset = _make(ds["state"]["default"])
        self._action_scale, self._action_offset = _make(ds["action"]["default"])
        print(
            f"[fastwam-qlora] normalizer loaded from {stats_path}\n"
            f"  state  min={ds['state']['default']['global_min']}\n"
            f"  state  max={ds['state']['default']['global_max']}\n"
            f"  action min={ds['action']['default']['global_min']}\n"
            f"  action max={ds['action']['default']['global_max']}",
            flush=True,
        )

    def _normalize_state(self, state6: np.ndarray) -> np.ndarray:
        x = state6.astype(np.float32) * self._state_scale + self._state_offset
        return np.clip(x, -5.0, 5.0)

    def _unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        # (T, 6) — same broadcasting as state
        return (action_norm.astype(np.float32) - self._action_offset) / self._action_scale

    def _load_lora_weights(self, ckpt_path: Path) -> None:
        """Route C: manually dequant bnb 4bit packed weights back to bf16,
        then load into the bf16 model.  Avoids bnb's flaky load-state hook
        and the QuantState reconstruction races we hit during training.
        Validated by smoke test producing sane action range [-1.6, 1.6].
        """
        from safetensors.torch import load_file
        from bitsandbytes.functional import QuantState, dequantize_4bit

        BNB_META = (".absmax", ".nested_absmax", ".nested_quant_map",
                    ".quant_map", ".quant_state.bitsandbytes__nf4")

        t0 = time.time()
        sd_raw = load_file(str(ckpt_path), device="cpu")
        print(f"[fastwam-qlora] raw ckpt loaded ({len(sd_raw)} keys) in {time.time()-t0:.1f}s",
              flush=True)

        # Find every Params4bit base (keys ending in .quant_state.bitsandbytes__nf4)
        base_keys = set()
        for k in sd_raw:
            if k.endswith(".quant_state.bitsandbytes__nf4"):
                base_keys.add(k[: -len(".quant_state.bitsandbytes__nf4")])

        sd_clean = {}
        n_dequant = 0
        t0 = time.time()
        for k, v in sd_raw.items():
            if any(k.endswith(suf) for suf in BNB_META):
                continue  # consumed by base lookup
            if k in base_keys:
                qs_dict = {}
                for suf in BNB_META:
                    if k + suf in sd_raw:
                        qs_dict[k + suf] = sd_raw[k + suf]
                qs = QuantState.from_dict(qs_dict, device="cuda")
                packed = v.to("cuda")
                bf16 = dequantize_4bit(packed, qs).to("cpu", dtype=torch.bfloat16)
                sd_clean[k] = bf16
                n_dequant += 1
                del packed, bf16, qs
            else:
                sd_clean[k] = v
        torch.cuda.empty_cache()
        print(f"[fastwam-qlora] dequantized {n_dequant} bnb base weights in {time.time()-t0:.1f}s",
              flush=True)

        missing, unexpected = self.model.load_state_dict(sd_clean, strict=False)
        lora_loaded = sum(1 for k in sd_clean if ".lora_A" in k or ".lora_B" in k)
        print(f"[fastwam-qlora] state_dict load: dequant_keys={len(sd_clean)} "
              f"lora={lora_loaded} missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)

    def _prompt_cache_path(self, prompt: str) -> "Path":
        from pathlib import Path
        import hashlib
        key = hashlib.sha1(
            f"{prompt}|{self._tokenizer_model_id}|{self._tokenizer_max_len}|{self._torch_dtype}".encode()
        ).hexdigest()[:16]
        cache_dir = Path.home() / ".cache" / "fastwam_leisaac" / "prompt_ctx"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key}.pt"

    @torch.no_grad()
    def _precompute_prompt_context(self, prompt: str) -> None:
        from fastwam.models.wan22.helpers.loader import _load_registered_model
        from fastwam.models.wan22.helpers.io import ModelConfig
        from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

        t0 = time.time()
        cache_path = self._prompt_cache_path(prompt)
        if cache_path.exists():
            blob = torch.load(cache_path, map_location="cpu", weights_only=True)
            self._cached_prompt = prompt
            self._cached_context = blob["context"].to(self.device, dtype=self._torch_dtype)
            self._cached_context_mask = blob["context_mask"].to(self.device)
            print(f"[fastwam-qlora] prompt loaded from cache {cache_path} "
                  f"in {time.time()-t0:.2f}s, gpu={torch.cuda.memory_allocated()/1e9:.2f}GB",
                  flush=True)
            return

        print(f"[fastwam-qlora] encoding prompt (CPU UMT5): {prompt!r}", flush=True)
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
        text_encoder = _load_registered_model(
            text_cfg.path, "wan_video_text_encoder",
            torch_dtype=self._torch_dtype, device="cpu",
        ).eval()
        tokenizer = HuggingfaceTokenizer(
            name=tok_cfg.path, seq_len=self._tokenizer_max_len, clean="whitespace",
        )
        ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
        mask_cpu = mask.to("cpu", dtype=torch.bool)
        prompt_emb = text_encoder(ids.to("cpu"), mask_cpu)
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
        torch.save(
            {"context": context.detach().cpu(),
             "context_mask": context_mask.detach().cpu()},
            cache_path,
        )
        print(f"[fastwam-qlora] prompt encoded in {time.time()-t0:.1f}s, "
              f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB, "
              f"cached at {cache_path}", flush=True)

    @torch.no_grad()
    def predict_action(
        self,
        front_img: np.ndarray,
        wrist_img: np.ndarray,
        state6: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Return (T, 6) absolute joint-position action chunk."""
        front = _resize_center_crop(front_img, self.img_size)
        wrist = _resize_center_crop(wrist_img, self.img_size)
        rgb = np.concatenate([front, wrist], axis=1)

        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        x = x.to(device=self.device, dtype=self.model.torch_dtype)
        x = x * (2.0 / 255.0) - 1.0

        # CRITICAL: training pipeline normalizes state to [-1, 1] via min/max
        # of dataset stats.  Server was feeding raw degrees (OOD 50× for the
        # model) — the cumsum+arm_delta_scale hack only masked symptoms.
        state6_norm = self._normalize_state(state6)
        proprio = torch.from_numpy(state6_norm).unsqueeze(0)
        proprio = proprio.to(device=self.device, dtype=self.model.torch_dtype)

        # Always use the cached prompt — re-encoding UMT5 on CPU takes ~4 min
        # which blocks ZMQ recv and crashes the eval client.  The trained
        # task is fixed; prompt variations from the client are ignored.
        if self._cached_context is None:
            self._precompute_prompt_context(self.default_prompt)
        if prompt != self._cached_prompt:
            print(f"[fastwam-qlora] WARN: ignoring client prompt {prompt!r}; "
                  f"using cached {self._cached_prompt!r}", flush=True)

        out = self.model.infer_action(
            prompt=None,
            input_image=x,
            action_horizon=self.action_horizon,
            proprio=proprio,
            context=self._cached_context,
            context_mask=self._cached_context_mask,
            num_inference_steps=self.num_inference_steps,
            text_cfg_scale=1.0,
            seed=0,
            rand_device="cpu",
        )
        act_chunk_norm = out["action"].cpu().numpy().astype(np.float32)  # (T, 6) in [-1, 1]
        if act_chunk_norm.ndim != 2 or act_chunk_norm.shape[1] != 6:
            raise ValueError(f"unexpected action chunk shape {act_chunk_norm.shape}")

        # Training stores absolute joint positions (action ≈ state[t+1]); the
        # `delta_action_dim_mask` in fastwam_processor.py only handles padding,
        # NOT delta conversion (verified by reading processor lines 251-259).
        # So model output is normalized absolute joint position trajectory →
        # just un-normalize, no cumsum / no arm_delta_scale needed.
        act_chunk = self._unnormalize_action(act_chunk_norm)
        return act_chunk.astype(np.float32)

    def get_action(self, obs: dict) -> dict:
        front = obs.get("video.front")
        wrist = obs.get("video.wrist")
        if front is None or wrist is None:
            raise ValueError(f"need both video.front + video.wrist. got: {sorted(obs)}")
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


def serve(server: FastWAMQLoRAServer, host: str, port: int) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[fastwam-qlora] ready, listening on tcp://{host}:{port}", flush=True)
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
                sock.send(msgpack.packb({"status": "ok", "data": data,
                                          "inference_time_ms": infer_ms}))
                step += 1
                if step % 5 == 0:
                    a = action["action.single_arm"][0].tolist()
                    g = float(action["action.gripper"][0, 0])
                    print(f"[fastwam-qlora] step={step} action5={a} grip={g:.3f} "
                          f"latency={infer_ms:.0f}ms", flush=True)
                continue
            sock.send(msgpack.packb({"status": "error",
                                     "message": f"Unknown endpoint: {ep}"}))
        except KeyboardInterrupt:
            print("[fastwam-qlora] interrupted", flush=True)
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
    ap.add_argument("--port", type=int, default=5559)
    ap.add_argument("--repo-root", default=os.environ.get(
        "FASTWAM_REPO_ROOT", os.path.expanduser("~/work/fastwam-repo")))
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR,
                    help="Path to training ckpt dir (containing model.safetensors)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--arm-delta-scale", type=float, default=DEFAULT_ARM_DELTA_SCALE)
    ap.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    ap.add_argument("--num-inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    args = ap.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(Path(args.repo_root) / config_path)
    os.chdir(args.repo_root)

    server = FastWAMQLoRAServer(
        config_path=config_path,
        ckpt_dir=args.ckpt_dir,
        default_prompt=args.prompt,
        arm_delta_scale=args.arm_delta_scale,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
    )
    serve(server, args.host, args.port)


if __name__ == "__main__":
    main()
