#!/usr/bin/env python3
"""Convert raw SONIC token episodes (.npz) into a UNITREE_G1_SONIC LeRobot dataset.

Input : datasets/sonic_vla_raw/<motion_key>/episode_*.npz  (from scripts/gear_sonic_record.sh)
        each npz: motion_token(T,64) joint_pos(T,J) root_quat(T,4) ego_rgb(T,H,W,3) fps prompt motion_key
Output: a LeRobot v2.1 dataset with meta/modality.json, ready for Isaac-GR00T finetune
        (--embodiment-tag UNITREE_G1_SONIC).

Uses the AUTHORITATIVE writer gear_sonic.data.exporter.Gr00tDataExporter + the exact
feature/modality schema from gear_sonic.data.features_sonic_vla. Fields we captured are
filled with real values; the remaining teleop-side fields (smpl_*, planner_*, vr_*, eef_*)
are zero-filled so validate_frame passes — they are unused for the blind prompt->token task.

Run in the data-collection venv (has the OLD lerobot.common layout that the exporter needs):
    cd dependencies/GR00T-WholeBodyControl
    .venv_data_collection/bin/python ../../scripts/sonic_vla_npz_to_lerobot.py \
        --raw-dir ../../datasets/sonic_vla_raw \
        --out ../../datasets/sonic_vla_lerobot
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

_DTYPE = {
    "float64": np.float64,
    "float32": np.float32,
    "int64": np.int64,
    "int32": np.int32,
    "bool": np.bool_,
}


def _zero(spec: dict) -> np.ndarray:
    shape = tuple(spec["shape"])
    return np.zeros(shape, dtype=_DTYPE.get(spec["dtype"], np.float32))


def _fit(vec: np.ndarray, n: int) -> np.ndarray:
    """Fit a 1-D vector to length n (truncate or zero-pad)."""
    vec = np.asarray(vec).reshape(-1)
    if vec.shape[0] == n:
        return vec
    out = np.zeros(n, dtype=vec.dtype)
    k = min(n, vec.shape[0])
    out[:k] = vec[:k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()

    from gear_sonic.data.exporter import Gr00tDataExporter
    from gear_sonic.data.features_sonic_vla import (
        get_features_sonic_vla,
        get_modality_config_sonic_vla,
    )
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.data_collection.transforms import compute_projected_gravity

    robot_model = instantiate_g1_robot_model()
    features = get_features_sonic_vla(robot_model)
    modality = get_modality_config_sonic_vla(robot_model)

    state_dim = int(np.prod(features["observation.state"]["shape"]))
    video_key = "observation.images.ego_view"
    print(f"[convert] num_joints(state_dim)={state_dim}  features={len(features)} keys")

    episodes = sorted(glob.glob(os.path.join(args.raw_dir, "*", "episode_*.npz")))
    if not episodes:
        print(f"[convert] no episodes under {args.raw_dir}")
        return 1
    print(f"[convert] {len(episodes)} episodes -> {args.out}")

    first_prompt = str(np.load(episodes[0], allow_pickle=True)["prompt"])
    exporter = Gr00tDataExporter.create(
        save_root=args.out,
        fps=args.fps,
        features=features,
        modality_config=modality,
        task=first_prompt,
        overwrite_existing=args.overwrite,
    )

    total = 0
    for ep_path in episodes:
        d = np.load(ep_path, allow_pickle=True)
        prompt = str(d["prompt"])
        token = d["motion_token"].astype(np.float64)     # (T,64)
        joints = d["joint_pos"].astype(np.float64)        # (T,J)
        root_quat = d["root_quat"].astype(np.float64)     # (T,4)
        rgb = d["ego_rgb"] if "ego_rgb" in d else None    # (T,H,W,3) u8
        T = token.shape[0]

        for t in range(T):
            frame: dict = {}
            for key, spec in features.items():
                if key == video_key:
                    frame[key] = (
                        rgb[t] if rgb is not None
                        else np.zeros(tuple(spec["shape"]), dtype=np.uint8)
                    )
                elif key == "action.motion_token":
                    frame[key] = token[t].astype(np.float64)
                elif key == "observation.state":
                    frame[key] = _fit(joints[t], state_dim)
                elif key == "action.wbc":
                    frame[key] = _fit(joints[t], state_dim)
                elif key == "observation.root_orientation":
                    frame[key] = root_quat[t]
                elif key == "observation.projected_gravity":
                    # real gravity-in-body-frame from recorded root quat (codex fix:
                    # was zero-filled -> train/deploy mismatch; modality.state uses it)
                    frame[key] = compute_projected_gravity(root_quat[t]).astype(np.float64)
                else:
                    frame[key] = _zero(spec)
            frame["task"] = prompt
            exporter.add_frame(frame)

        exporter.save_episode()
        total += T
        print(f"[convert] {os.path.basename(os.path.dirname(ep_path))}: {T} frames  (prompt='{prompt}')")

    print(f"[convert] DONE: {len(episodes)} episodes, {total} frames -> {args.out}")
    print(f"[convert] modality.json + meta written; ready for Isaac-GR00T (UNITREE_G1_SONIC).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
