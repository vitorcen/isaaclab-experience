#!/usr/bin/env python3
"""Look up the right inference action_horizon for a given policy.

Two-stage lookup:

  1. Read ``scripts/benchmark/baselines.tsv`` — authoritative manifest
     (col 3 = horizon, col 4 = ckpt). Matches on ckpt column.
  2. Fallback: download ``config.json`` from the HF model id and read its
     ``action_horizon`` field (the value the model was *trained* with).

Print the integer to stdout so shells can do ``ACTION_HORIZON=$(... $MODEL)``.

Usage::

    python3 scripts/benchmark/get_action_horizon.py hi-space/GR00T-N1.7-3B-Pick-Orange
    # → 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "scripts" / "benchmark" / "baselines.tsv"


def lookup_tsv(model_id: str) -> int | None:
    if not TSV_PATH.exists():
        return None
    for line in TSV_PATH.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        # baselines.tsv columns: slug | policy_type | horizon | ckpt | server_kind | label
        if len(parts) < 4:
            continue
        if parts[3].strip() == model_id:
            try:
                return int(parts[2].strip())
            except ValueError:
                return None
    return None


def lookup_hf_config(model_id: str) -> int | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        cfg_path = hf_hub_download(model_id, "config.json")
    except Exception:
        return None
    cfg = json.loads(Path(cfg_path).read_text())
    return cfg.get("action_horizon")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_id", help="HF model id, e.g. hi-space/GR00T-N1.7-3B-Pick-Orange")
    p.add_argument("--default", type=int, default=16, help="Fallback if not found (default 16)")
    args = p.parse_args()

    h = lookup_tsv(args.model_id)
    if h is None:
        h = lookup_hf_config(args.model_id)
    if h is None:
        print(f"[warn] no action_horizon for {args.model_id}, using default={args.default}", file=sys.stderr)
        h = args.default
    print(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
