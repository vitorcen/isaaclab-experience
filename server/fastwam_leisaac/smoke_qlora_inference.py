#!/usr/bin/env python3
"""Offline smoke: dequant ckpt → bf16 forward → print action magnitude.

Validates that the trained LoRA + small layers (action_encoder/head/proprio_encoder)
actually produce sane joint deltas (expected: |a| < π/4 ≈ 0.78), not the
1e7-magnitude garbage we saw with the bnb-quantized inference path.

If this gives sensible output → training IS converging; the bug was just the
inference loading path (route A/B vs route C from the debug doc).

Usage:
    conda activate fastwam
    cd ~/work/fastwam-repo
    python -m fastwam_leisaac.smoke_qlora_inference \\
        --ckpt-dir runs/train/fastwam_qlora_pickorange_5phase/phase2/checkpoints/state/step_004000
"""

from __future__ import annotations

import torchgen.model as _torchgen_model  # noqa: F401 — early enum init

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

_FT_DIR = "/home/david/work/isaaclab-experience/LeIsaac/scripts/finetune"
if _FT_DIR not in sys.path:
    sys.path.insert(0, _FT_DIR)


def dequant_bnb_in_state_dict(sd: dict) -> tuple[dict, int]:
    """Walk sd, for each bnb-saved 4bit weight, reconstruct QuantState and
    dequant back to bf16 on CPU.  Returns (clean_sd, num_dequantized).

    bnb 0.49 QuantState.from_dict expects qs_dict with keys WITHOUT the model
    prefix — it does `k.split(".")[-1]` internally.  We pass keys with prefix;
    the split-last-segment still works on names like
    `mot.mixtures.action.blocks.0.self_attn.q.base_layer.weight.absmax`
    because the final segment is `absmax` / `quant_state.bitsandbytes__nf4` etc.

    Wait — actually `quant_state.bitsandbytes__nf4` has a dot in it.  So
    `k.split(".")[-1]` would yield `bitsandbytes__nf4` not the full key.
    Verify by reading bnb source — line in `QuantState.from_dict`:
        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}
    AND earlier:
        qs_key = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, torch.Tensor)]
    So it locates the "quant_state.bitsandbytes__nf4" key by substring match,
    pops it, and unpacks its non-tensor contents.  The remaining keys
    (absmax, nested_absmax, etc.) are then split-last-segment to short keys.
    So passing keys with our model prefix should work — split-last-segment
    on `...absmax` gives `absmax`, on `...nested_absmax` gives `nested_absmax`, etc.
    ✓ Compatible.
    """
    from bitsandbytes.functional import QuantState, dequantize_4bit

    BNB_META = (".absmax", ".nested_absmax", ".nested_quant_map",
                ".quant_map", ".quant_state.bitsandbytes__nf4")

    base_keys = set()
    for k in sd:
        if k.endswith(".quant_state.bitsandbytes__nf4"):
            base_keys.add(k[: -len(".quant_state.bitsandbytes__nf4")])

    out = {}
    n_dequant = 0
    for k, v in sd.items():
        if any(k.endswith(suf) for suf in BNB_META):
            continue  # consumed below
        if k in base_keys:
            qs_dict = {}
            for suf in BNB_META:
                meta_k = k + suf
                if meta_k in sd:
                    qs_dict[meta_k] = sd[meta_k]
            qs = QuantState.from_dict(qs_dict, device="cuda")
            packed = v.to("cuda")
            bf16 = dequantize_4bit(packed, qs).to("cpu", dtype=torch.bfloat16)
            # Convert back to expected shape if needed.  Linear4bit stores
            # weight as a 1-D packed view internally, but dequantize_4bit
            # returns the original 2-D shape (out_features, in_features).
            out[k] = bf16
            n_dequant += 1
            del packed, bf16, qs
        else:
            out[k] = v
    torch.cuda.empty_cache()
    return out, n_dequant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--config", default="configs/model/fastwam.yaml")
    ap.add_argument("--repo-root", default=os.path.expanduser("~/work/fastwam-repo"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.chdir(args.repo_root)
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(Path(args.repo_root) / config_path)

    cfg = OmegaConf.load(config_path)
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

    from fastwam.runtime import create_fastwam
    from fastwam_qlora.qlora_utils import apply_lora

    print(f"[smoke] building bf16 model (no NF4 quantization)...", flush=True)
    t0 = time.time()
    model = create_fastwam(
        model_id=cfg.model_id,
        tokenizer_model_id=cfg.tokenizer_model_id,
        video_dit_config=cfg.video_dit_config,
        tokenizer_max_len=int(cfg.tokenizer_max_len),
        load_text_encoder=False,
        proprio_dim=int(cfg.proprio_dim),
        action_dit_config=cfg.action_dit_config,
        action_dit_pretrained_path=cfg.get("action_dit_pretrained_path"),
        skip_dit_load_from_pretrain=False,
        video_scheduler=cfg.video_scheduler,
        action_scheduler=cfg.action_scheduler,
        loss=cfg.get("loss"),
        mot_checkpoint_mixed_attn=False,
        redirect_common_files=True,
        model_dtype=torch.bfloat16,
        device=args.device,
    )
    print(f"[smoke] model built in {time.time()-t0:.1f}s "
          f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Apply LoRA wrap ONLY (no quantization).
    print("[smoke] applying manual LoRA wrap (no quantize)...", flush=True)
    model.dit = apply_lora(model.dit, r=16, alpha=16, dropout=0.0,
                            target_modules=("q", "k", "v", "o"))
    torch.cuda.empty_cache()
    print(f"[smoke] post-LoRA gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Load ckpt + dequant bnb 4bit → bf16.
    from safetensors.torch import load_file
    ckpt_path = Path(args.ckpt_dir) / "model.safetensors"
    print(f"[smoke] loading + dequantizing ckpt {ckpt_path}...", flush=True)
    t0 = time.time()
    sd_raw = load_file(str(ckpt_path), device="cpu")
    print(f"[smoke]   raw ckpt loaded ({len(sd_raw)} keys) in {time.time()-t0:.1f}s",
          flush=True)
    t0 = time.time()
    sd_clean, n_deq = dequant_bnb_in_state_dict(sd_raw)
    print(f"[smoke]   dequantized {n_deq} bnb weights in {time.time()-t0:.1f}s, "
          f"clean keys={len(sd_clean)}", flush=True)
    t0 = time.time()
    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    print(f"[smoke]   load_state_dict in {time.time()-t0:.1f}s: "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if unexpected:
        print(f"[smoke]   unexpected sample: {unexpected[:3]}", flush=True)
    if missing:
        # Filter informative missing keys (skip aliases)
        non_alias_missing = [k for k in missing
                             if not any(a in k for a in
                                        ["action_expert.", "video_expert.", "dit.", "vae."])]
        print(f"[smoke]   non-alias missing ({len(non_alias_missing)}): "
              f"{non_alias_missing[:5]}", flush=True)

    # Quick sanity: check action_encoder weight has non-trivial values.
    ae = model.action_expert.action_encoder
    print(f"[smoke] action_encoder.weight stats: "
          f"shape={tuple(ae.weight.shape)} "
          f"norm={ae.weight.norm().item():.4f} "
          f"abs_mean={ae.weight.abs().mean().item():.4f}", flush=True)
    # Check a LoRA adapter:
    for name, p in model.named_parameters():
        if name.endswith("blocks.0.self_attn.q.lora_A.weight"):
            print(f"[smoke] LoRA sample {name}: "
                  f"norm={p.norm().item():.4f} abs_mean={p.abs().mean().item():.4f}",
                  flush=True)
            break

    # Encode prompt on CPU.
    print("[smoke] encoding test prompt on CPU UMT5...", flush=True)
    t0 = time.time()
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    from fastwam.models.wan22.helpers.io import ModelConfig
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    text_cfg = ModelConfig(
        model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors",
    )
    tok_cfg = ModelConfig(
        model_id=str(cfg.tokenizer_model_id),
        origin_file_pattern="google/umt5-xxl/",
    )
    text_cfg.download_if_necessary()
    tok_cfg.download_if_necessary()
    text_enc = _load_registered_model(text_cfg.path, "wan_video_text_encoder",
                                       torch_dtype=torch.bfloat16, device="cpu").eval()
    tok = HuggingfaceTokenizer(name=tok_cfg.path,
                                seq_len=int(cfg.tokenizer_max_len), clean="whitespace")
    prompt = "Grab orange and place into plate"
    ids, mask = tok(prompt, return_mask=True, add_special_tokens=True)
    mask_cpu = mask.to("cpu", dtype=torch.bool)
    emb = text_enc(ids.to("cpu"), mask_cpu)
    seq_lens = mask_cpu.gt(0).sum(dim=1).long()
    for i, v in enumerate(seq_lens):
        emb[i, v:] = 0
    context = emb.to(args.device, dtype=torch.bfloat16)
    context_mask = torch.ones_like(mask_cpu).to(args.device)
    del text_enc, tok, ids, mask_cpu, emb
    torch.cuda.empty_cache()
    print(f"[smoke] prompt encoded in {time.time()-t0:.1f}s "
          f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Synthetic forward — fixed-seed image, see action output magnitude.
    model.eval()
    rng = np.random.default_rng(0)
    front = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    wrist = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    # center-crop to 224 then concat horiz to 224x448
    from PIL import Image
    def crop(img):
        pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)
    rgb = np.concatenate([crop(front), crop(wrist)], axis=1)  # (224, 448, 3)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(args.device, dtype=torch.bfloat16)
    x = x * (2.0/255.0) - 1.0
    proprio = torch.zeros(1, 6, device=args.device, dtype=torch.bfloat16)

    with torch.no_grad():
        t0 = time.time()
        out = model.infer_action(
            prompt=None, input_image=x, action_horizon=24,
            proprio=proprio, context=context, context_mask=context_mask,
            num_inference_steps=10, text_cfg_scale=1.0, seed=0, rand_device="cpu",
        )
        dt_ms = 1000 * (time.time() - t0)
    act = out["action"].cpu().numpy().astype(np.float32)
    print(f"[smoke] inference {dt_ms:.0f}ms shape={act.shape}", flush=True)
    print(f"[smoke] action stats: min={act.min():.4f} max={act.max():.4f} "
          f"abs_mean={np.abs(act).mean():.4f} abs_max={np.abs(act).max():.4f}",
          flush=True)
    print(f"[smoke] first step action6: {act[0].tolist()}", flush=True)
    print(f"[smoke] last  step action6: {act[-1].tolist()}", flush=True)

    if np.abs(act).max() < 5.0:
        print("[smoke] ✅ SANE — action magnitudes look reasonable (< 5)", flush=True)
    elif np.abs(act).max() < 100.0:
        print("[smoke] ⚠️ borderline — action magnitudes moderate (5..100)", flush=True)
    else:
        print(f"[smoke] ❌ STILL GARBAGE — action max {np.abs(act).max():.0f}", flush=True)


if __name__ == "__main__":
    main()
