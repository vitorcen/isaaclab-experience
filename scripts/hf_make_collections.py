#!/usr/bin/env python3
"""Group the wsagi HF profile into Collections (idempotent, re-runnable).

HF has no auto-grouping by name — Collections are the only native grouping. This
builds one collection per project family. Safe to re-run: create_collection and
add_collection_item both use exists_ok=True.

Auth: needs a WRITE token. Either `huggingface-cli login` first, or:
    HF_TOKEN=hf_xxx python scripts/hf_make_collections.py
Never paste the token on the command line in a shared shell; use the env/keyring.
"""
import os
from huggingface_hub import create_collection, add_collection_item

NS = "wsagi"
TOKEN = os.environ.get("HF_TOKEN")  # falls back to cached login if None

# title | description | [(item_id, item_type), ...]
GROUPS = [
    ("LeIsaac PickOrange",
     "SO-101 single-arm pick-orange-and-place benchmark — same task, many policy families (strict 20-round eval).",
     [(f"{NS}/DiffusionPolicy-PickOrange", "model"),
      (f"{NS}/ACT-PickOrange", "model"),
      (f"{NS}/SmolVLA-PickOrange", "model"),
      (f"{NS}/X-VLA-PickOrange", "model"),
      (f"{NS}/OpenVLA-PickOrange", "model"),
      (f"{NS}/GR00T-N1.6-PickOrange", "model"),
      (f"{NS}/GR00T-N1.7-PickOrange", "model"),
      (f"{NS}/Pi0.5-PickOrange", "model"),
      (f"{NS}/StarVLA-PickOrange", "model"),
      (f"{NS}/StarVLA-Qwen3-VL-8B-PickOrange", "model"),
      (f"{NS}/StarVLA-Qwen3-VL-8B-PI_v3-PickOrange", "model"),
      (f"{NS}/StarVLA-Qwen3.5-2B-PI_v3-PickOrange", "model"),
      (f"{NS}/StarVLA-Qwen3.5-4B-PI_v3-PickOrange", "model"),
      (f"{NS}/StarVLA-Qwen3.5-9B-PI_v3-PickOrange", "model")]),

    ("RoboCasa",
     "GR00T policies on RoboCasa kitchen manipulation tasks.",
     [(f"{NS}/GR00T-N1.7-RoboCasa-OpenCabinet", "model")]),

    ("MimicKit",
     "MimicKit motion-tracking on Unitree G1 (LAFAN clips).",
     [(f"{NS}/MimicKit-G1-LAFAN", "model")]),

    ("SONIC",
     "GR00T-in-loop driving the SONIC whole-body controller on G1, plus the VLA dataset.",
     [(f"{NS}/GR00T-N1.7-G1-SONIC", "model"),
      (f"{NS}/SONIC-VLA-LeRobot", "dataset")]),

    # Optional 5th group for the two HumanoidBench RL baselines — comment out if unwanted.
    ("HumanoidBench",
     "HumanoidBench RL baselines (DrQ-v2, TD-MPC2).",
     [(f"{NS}/HumanoidBench-DrQ", "model"),
      (f"{NS}/HumanoidBench-TD-MPC2", "model")]),
]


def main():
    for title, desc, items in GROUPS:
        col = create_collection(title=title, namespace=NS, description=desc,
                                private=False, exists_ok=True, token=TOKEN)
        print(f"[collection] {title} -> {col.slug}")
        for item_id, item_type in items:
            try:
                add_collection_item(col.slug, item_id=item_id, item_type=item_type,
                                    exists_ok=True, token=TOKEN)
                print(f"    + {item_type:7} {item_id}")
            except Exception as e:
                print(f"    ! FAILED {item_id}: {e}")
    print("\nDone. Reorder collections by dragging on https://huggingface.co/" + NS)


if __name__ == "__main__":
    main()
