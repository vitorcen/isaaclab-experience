#!/usr/bin/env python3
"""π0.5 PyTorch LoRA fine-tuning on a LeRobot dataset.

Uses the full lerobot pre/post-processor pipeline so train and inference
share one contract:

  1. multi-camera support (3-slot, handled by ``_preprocess_images``)
  2. real state input (``Pi05PrepareStateTokenizerProcessorStep``)
  3. real PaliGemma tokenizer prompt (``TokenizerProcessorStep``)
  4. real batch size (configurable, default 16)
  5. real AdamW + cosine warmup (replaces hand-rolled SGD)
  6. loss masked to the dataset's actual action dim (handled by
     ``PI05Policy.forward`` which truncates to ``output_features[ACTION].shape[0]``)

Usage:
    python -m pi05_leisaac.train \\
        --policy-id lerobot/pi05_base \\
        --dataset-repo-id LightwheelAI/leisaac-pick-orange \\
        --output-dir outputs/pi05-leisaac-pt-v3 \\
        --steps 3000 --batch-size 16 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# lerobot import path: editable fork by convention. Override via $LEROBOT_SRC.
_LEROBOT_SRC = os.environ.get(
    "LEROBOT_SRC", str(Path.home() / "work/lerobot-experience/lerobot/src")
)
if os.path.isdir(_LEROBOT_SRC) and _LEROBOT_SRC not in sys.path:
    sys.path.insert(0, _LEROBOT_SRC)

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.policies.factory import make_policy  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402

from .lora import (  # noqa: E402
    LoRALinear,
    _LAYER_PREFIX_MAP,
    _TOPLEVEL_MAP,
    wrap_pi05_with_lora,
)


# ---------------------------------------------------------------------------
# LoRA helpers (init for training, dump npz)
# ---------------------------------------------------------------------------
def init_lora_for_training(wrapped: dict[str, LoRALinear]) -> int:
    """Switch LoRA params to trainable + standard init (A: kaiming, B: zero)."""
    n_trainable = 0
    for _, wrap in wrapped.items():
        wrap.lora_A.requires_grad_(True)
        wrap.lora_B.requires_grad_(True)
        nn.init.kaiming_uniform_(wrap.lora_A, a=math.sqrt(5))
        nn.init.zeros_(wrap.lora_B)
        n_trainable += wrap.lora_A.numel() + wrap.lora_B.numel()
    return n_trainable


def _pt_path_to_npz_prefix(pt_path: str) -> tuple[str, str | None]:
    """Reverse of lora._resolve_pt_name → npz key prefix.

    Returns (npz_prefix_without_AB, None) on success or ("", None) on no match.
    """
    # toplevel projections
    for short, long in _TOPLEVEL_MAP.items():
        if pt_path == long:
            return short, None
    # per-layer self_attn
    # e.g. pt_path = "model.paligemma_with_expert.paligemma.model.language_model.layers.5.self_attn.q_proj"
    for short, long_prefix in _LAYER_PREFIX_MAP.items():
        if pt_path.startswith(long_prefix + "."):
            tail = pt_path[len(long_prefix) + 1 :]  # "5.self_attn.q_proj"
            return f"{short}.{tail}", None
    return "", None


def dump_lora_npz(wrapped: dict[str, LoRALinear], path: str) -> int:
    """Save trained LoRA A/B as a .npz so server.py can load it."""
    arrays: dict[str, np.ndarray] = {}
    for pt_path, wrap in wrapped.items():
        npz_prefix, _ = _pt_path_to_npz_prefix(pt_path)
        if not npz_prefix:
            raise RuntimeError(f"cannot reverse-map pt path: {pt_path}")
        arrays[f"{npz_prefix}.lora_A"] = wrap.lora_A.detach().to(
            dtype=torch.float32, device="cpu"
        ).numpy()
        arrays[f"{npz_prefix}.lora_B"] = wrap.lora_B.detach().to(
            dtype=torch.float32, device="cpu"
        ).numpy()
    np.savez(path, **arrays)
    return len(arrays)


def dump_lora_safetensors(wrapped: dict[str, LoRALinear], path: str) -> int:
    """Save trained LoRA as a safetensors file (PT-native, keyed by pt path)."""
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    for pt_path, wrap in wrapped.items():
        tensors[f"{pt_path}.lora_A"] = wrap.lora_A.detach().to(
            dtype=torch.float32, device="cpu"
        )
        tensors[f"{pt_path}.lora_B"] = wrap.lora_B.detach().to(
            dtype=torch.float32, device="cpu"
        )
    save_file(tensors, path)
    return len(tensors)


# ---------------------------------------------------------------------------
# Scheduler: linear warmup then cosine decay
# ---------------------------------------------------------------------------
def build_cosine_warmup(
    optimizer: torch.optim.Optimizer, *, warmup: int, total: int, min_lr_ratio: float = 0.1
):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / ".logs"
    log_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        args.dtype
    ]
    print(f"[train] device={device} dtype={dtype}", flush=True)

    # --- dataset meta + delta timestamps ---------------------------------
    ds_root = args.dataset_root or None
    print(
        f"[train] loading dataset metadata {args.dataset_repo_id} "
        f"(root={ds_root or 'HF cache'}) ...",
        flush=True,
    )
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=ds_root)
    print(
        f"[train]   fps={ds_meta.fps} episodes={ds_meta.total_episodes} "
        f"frames={ds_meta.total_frames}",
        flush=True,
    )

    cfg = PI05Config()
    cfg.pretrained_path = args.policy_id
    cfg.device = str(device)
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    # --- dataset ---------------------------------------------------------
    print("[train] loading dataset ...", flush=True)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=ds_root,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
        tolerance_s=args.tolerance_s,
    )

    # --- policy (loads pretrained + injects features from dataset) -------
    print(f"[train] loading PI05Policy from {args.policy_id} ...", flush=True)
    if args.gradient_checkpointing:
        cfg.gradient_checkpointing = True
    policy = make_policy(cfg, ds_meta=dataset.meta).to(device=device, dtype=dtype)
    policy.train()
    if args.gradient_checkpointing:
        # The PI05 model checks two flags; set both so backward picks the
        # checkpointed path. Saves ~50% activation memory at ~25% step-time cost.
        policy.model.gradient_checkpointing = True
        if hasattr(policy.model.paligemma_with_expert.gemma_expert.model, "gradient_checkpointing"):
            policy.model.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        print("[train]   gradient_checkpointing = True", flush=True)

    # --- LoRA injection --------------------------------------------------
    print(f"[train] wrapping LoRA targets (rank={args.lora_r} alpha={args.lora_alpha}) ...", flush=True)
    wrapped = wrap_pi05_with_lora(policy, rank=args.lora_r, alpha=args.lora_alpha)
    # Freeze base, then unfreeze LoRA params.
    for p in policy.parameters():
        p.requires_grad_(False)

    if args.init_lora_npz:
        # Continuation: keep the previous LoRA weights, just flip
        # requires_grad on. Skips kaiming/zeros re-init.
        print(f"[train]   continuing from LoRA weights {args.init_lora_npz}", flush=True)
        from .lora import load_lora_npz  # local import to avoid cycle
        report = load_lora_npz(
            policy, args.init_lora_npz, rank=args.lora_r, alpha=args.lora_alpha
        )
        if report["missing"]:
            raise RuntimeError(f"init LoRA missing layers: {report['missing'][:5]}")
        if report["skipped"]:
            print(f"[train]   skipped keys (kept as-is): {report['skipped'][:3]}", flush=True)
        for wrap in wrapped.values():
            wrap.lora_A.requires_grad_(True)
            wrap.lora_B.requires_grad_(True)
        n_trainable_lora = sum(
            w.lora_A.numel() + w.lora_B.numel() for w in wrapped.values()
        )
    else:
        n_trainable_lora = init_lora_for_training(wrapped)
    # Make sure trainable params are in fp32 for stable AdamW even when the
    # rest of the model is bf16 (PEFT-style "master copy" pattern).
    for wrap in wrapped.values():
        wrap.lora_A.data = wrap.lora_A.data.to(dtype=torch.float32)
        wrap.lora_B.data = wrap.lora_B.data.to(dtype=torch.float32)
    n_params_total = sum(p.numel() for p in policy.parameters())
    n_params_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(
        f"[train]   wrapped={len(wrapped)} trainable_params={n_params_train:,} "
        f"({n_params_train / n_params_total:.2%} of {n_params_total:,})",
        flush=True,
    )

    # --- LoRALinear dtype safety: LoRA path stays fp32 -------------------
    # Override forward so the LoRA delta computes in fp32 (matches lora_A/B)
    # and we only cast x to base.weight.dtype for the base path.
    for wrap in wrapped.values():
        _orig_forward_base = wrap.base  # type: ignore[attr-defined]

    def _patched_forward(self: LoRALinear, x: torch.Tensor) -> torch.Tensor:
        wdtype = self.base.weight.dtype
        x_base = x.to(wdtype) if x.dtype != wdtype else x
        out = self.base(x_base)
        if self.scale != 0.0:
            x32 = x.to(torch.float32)
            down = F.linear(x32, self.lora_A)
            up = F.linear(down, self.lora_B)
            delta = (self.scale * up).to(out.dtype)
            out = out + delta
        return out

    LoRALinear.forward = _patched_forward  # type: ignore[assignment]

    # --- processors ------------------------------------------------------
    print("[train] building processors ...", flush=True)
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        policy.config, dataset_stats=dataset.meta.stats
    )

    # --- optimizer + scheduler ------------------------------------------
    lora_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    scheduler = build_cosine_warmup(optimizer, warmup=args.warmup_steps, total=args.steps)

    # --- dataloader ------------------------------------------------------
    sampler = None
    shuffle = True
    if args.phased_sampler:
        # Anchor-frame weighting: heavily oversample the episode-start chunks
        # so the optimizer doesn't keep flattening rare large reach-over actions.
        # Each dataset[i] is a chunk anchored at frame i; for i in episode prefix
        # the chunk covers the start trajectory we want to recover.
        eps = ds_meta.episodes
        n = ds_meta.total_frames
        weights = np.ones(n, dtype=np.float64)
        n_start = 0
        n_mid = 0
        for from_idx, to_idx in zip(eps["dataset_from_index"], eps["dataset_to_index"]):
            from_idx = int(from_idx)
            to_idx = int(to_idx)
            ep_len = to_idx - from_idx
            head = min(args.phased_head_frames, ep_len)
            mid = min(args.phased_mid_frames, max(0, ep_len - head))
            weights[from_idx : from_idx + head] = args.phased_head_weight
            weights[from_idx + head : from_idx + head + mid] = args.phased_mid_weight
            n_start += head
            n_mid += mid
        sampler = WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=args.steps * args.batch_size * max(1, args.grad_accum),
            replacement=True,
        )
        shuffle = False
        print(
            f"[train]   phased sampler: head={args.phased_head_frames}f×{args.phased_head_weight:.1f} "
            f"({n_start} frames), mid={args.phased_mid_frames}f×{args.phased_mid_weight:.1f} "
            f"({n_mid} frames), rest×1.0 ({n - n_start - n_mid} frames)",
            flush=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    def cycle(loader: DataLoader):
        while True:
            for batch in loader:
                yield batch

    dl_iter = cycle(loader)
    camera_keys = dataset.meta.camera_keys

    print(
        f"[train] starting: steps={args.steps} batch={args.batch_size} "
        f"lr={args.lr} warmup={args.warmup_steps} save_freq={args.save_freq}",
        flush=True,
    )
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 2**20
        print(f"[train]   gpu allocated={mem:.0f}MiB", flush=True)

    train_log = open(log_dir / "train_metrics.jsonl", "w", buffering=1)

    grad_accum = max(1, args.grad_accum)
    losses_window: list[float] = []
    t_start = time.time()

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype != torch.float32
        else nullcontext()
    )

    for step in range(1, args.steps + 1):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        for _ in range(grad_accum):
            batch = next(dl_iter)
            # uint8 image → float in [0,1] (preprocessor expects this).
            for k in camera_keys:
                if k in batch and batch[k].dtype == torch.uint8:
                    batch[k] = batch[k].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)

            with autocast_ctx:
                loss, _info = policy.forward(batch)
            (loss / grad_accum).backward()
            accum_loss += loss.detach().float().item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        avg_loss = accum_loss / grad_accum
        losses_window.append(avg_loss)
        if len(losses_window) > 50:
            losses_window.pop(0)

        step_s = time.time() - t0
        if step % args.log_freq == 0 or step == 1:
            avg = sum(losses_window) / len(losses_window)
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            eta_min = (elapsed / step) * (args.steps - step) / 60
            print(
                f"[train] step={step}/{args.steps} loss={avg_loss:.4f} "
                f"avg50={avg:.4f} lr={lr_now:.2e} step_s={step_s:.2f} "
                f"eta={eta_min:.1f}min",
                flush=True,
            )
            train_log.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": avg_loss,
                        "avg50": avg,
                        "lr": lr_now,
                        "step_s": step_s,
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )

        if step % args.save_freq == 0 or step == args.steps:
            ckpt_npz = output_dir / f"checkpoint-{step}.npz"
            ckpt_st = output_dir / f"checkpoint-{step}.safetensors"
            n_npz = dump_lora_npz(wrapped, str(ckpt_npz))
            n_st = dump_lora_safetensors(wrapped, str(ckpt_st))
            print(
                f"[train]   saved {ckpt_npz.name} ({n_npz} arrays) + "
                f"{ckpt_st.name} ({n_st} tensors)",
                flush=True,
            )

    # final
    final_npz = output_dir / "final_lora.npz"
    final_st = output_dir / "final_lora.safetensors"
    dump_lora_npz(wrapped, str(final_npz))
    dump_lora_safetensors(wrapped, str(final_st))
    train_log.close()
    print(f"[train] done. final → {final_npz}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-id", default="lerobot/pi05_base")
    p.add_argument("--dataset-repo-id", default="LightwheelAI/leisaac-pick-orange")
    p.add_argument("--dataset-root", default=None, help="Local v3.0 dataset root (skips HF fetch)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--log-freq", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tolerance-s", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable activation re-compute to fit larger batches on 24GB GPUs",
    )
    p.add_argument(
        "--init-lora-npz",
        default=None,
        help="Initialize LoRA weights from an existing .npz "
             "(continuation training; skips kaiming/zeros re-init).",
    )
    p.add_argument(
        "--phased-sampler",
        action="store_true",
        help="Weight anchor frames near episode start more heavily to fix "
             "distribution-biased overfit on start-of-trajectory chunks.",
    )
    p.add_argument("--phased-head-frames", type=int, default=100,
                   help="First N anchor frames per episode get head weight.")
    p.add_argument("--phased-head-weight", type=float, default=8.0)
    p.add_argument("--phased-mid-frames", type=int, default=100,
                   help="Next N anchor frames (after head) get mid weight.")
    p.add_argument("--phased-mid-weight", type=float, default=4.0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
