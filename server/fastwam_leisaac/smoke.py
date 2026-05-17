#!/usr/bin/env python3
"""Offline smoke test for FastWAM: load model + LIBERO ckpt, run one
synthetic 2-cam frame, print shape + latency.

Usage:
    conda activate fastwam
    cd ~/work/isaaclab-experience
    python -m fastwam_leisaac.smoke
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .server import FastWAMDemoServer, DEFAULT_CONFIG, DEFAULT_CKPT, _resolve_paths


def main() -> None:
    repo_root = os.environ.get("FASTWAM_REPO_ROOT", os.path.expanduser("~/work/fastwam-repo"))
    config_path, ckpt_path = _resolve_paths(repo_root, DEFAULT_CONFIG, DEFAULT_CKPT)
    os.chdir(repo_root)

    server = FastWAMDemoServer(
        config_path=config_path,
        ckpt_path=ckpt_path,
        action_horizon=24,
        num_inference_steps=10,
    )

    rng = np.random.default_rng(0)
    front = (rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8))
    wrist = (rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8))
    state6 = np.zeros(6, dtype=np.float32)

    # warmup
    print("[smoke] warmup call...", flush=True)
    _ = server.predict_action(front, wrist, state6, "pick up the orange")

    print("[smoke] timing 3 inferences:", flush=True)
    for i in range(3):
        t0 = time.time()
        out = server.predict_action(front, wrist, state6, "pick up the orange")
        dt_ms = 1000 * (time.time() - t0)
        print(f"  iter {i}: shape={out.shape} dtype={out.dtype} latency={dt_ms:.0f}ms")

    print("[smoke] sample first-step action6:", out[0].tolist())
    print("[smoke] OK")


if __name__ == "__main__":
    main()
