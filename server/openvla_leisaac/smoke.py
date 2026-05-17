#!/usr/bin/env python3
"""Offline OpenVLA-7B 4-bit smoke test.

Loads the base model in NF4 4-bit and probes 4 prompts on one input image.
Use this BEFORE wiring to Isaac Sim to verify:
  - model loads (~32s, ~4.4 GB GPU)
  - inference produces plausible 7-DoF action numbers
  - language sensitivity is at least weakly present (move up/left tilts xyz)
  - latency budget (~250 ms/action warm, no flash_attn)

Run:
    python -m openvla_leisaac.smoke --image /tmp/leisaac_frame0.png
    python -m openvla_leisaac.smoke --image <png>  --prompt "stack the blocks"
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


DEFAULT_MODEL = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"

DEFAULT_PROBES = [
    ("target",       "pick up the orange and place it on the plate"),
    ("decoy_still",  "do nothing"),
    ("decoy_left",   "move the gripper left"),
    ("decoy_up",     "move the gripper up"),
]


def load_vla(model_name: str = DEFAULT_MODEL):
    print(f"[smoke] loading {model_name} (NF4 + bf16 compute)...")
    t0 = time.time()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"[smoke] loaded in {time.time()-t0:.1f}s, "
          f"gpu={torch.cuda.memory_allocated()/1e9:.2f} GB")
    return vla, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True,
                    help="path to an RGB image (PNG/JPG). For LeIsaac frames, "
                         "extract one with torchvision since videos are av1.")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    ap.add_argument("--prompt", default=None,
                    help="single prompt; if omitted, runs 4 default probes")
    args = ap.parse_args()

    vla, processor = load_vla(args.model_name)
    img = Image.open(args.image).convert("RGB")
    print(f"[smoke] image: {img.size}, mode={img.mode}")

    probes = [(tag, p) for tag, p in DEFAULT_PROBES] if args.prompt is None \
        else [("user", args.prompt)]

    for tag, instr in probes:
        prompt = f"In: What action should the robot take to {instr}?\nOut:"
        inputs = processor(prompt, img).to("cuda:0", dtype=torch.float16)
        t0 = time.time()
        action = vla.predict_action(
            **inputs, unnorm_key=args.unnorm_key, do_sample=False
        )
        dt_ms = 1000 * (time.time() - t0)
        print(f"[{tag}] '{instr}'")
        print(f"  action: {np.array2string(action, precision=4, suppress_small=True)}")
        print(f"  latency: {dt_ms:.0f} ms")


if __name__ == "__main__":
    main()
