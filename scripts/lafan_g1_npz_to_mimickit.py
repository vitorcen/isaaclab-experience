#!/usr/bin/env python3
"""
Convert ember-lab-berkeley/LAFAN-G1 .npz motions → MimicKit native .pkl.

npz schema (Isaac Lab AMP motion loader):
  fps:           (1,) float32
  dof_names:     (29,) <U
  body_names:    (30,) <U   — index 0 = pelvis (root)
  dof_positions: (N, 29) float32
  body_rotations: (N, 30, 4) float32 — quaternion in WXYZ order (Isaac Lab convention)
  body_positions: (N, 30, 3) float32 — body[0] is the pelvis world position

MimicKit native pkl schema (per gmr_to_mimickit.py):
  Motion(loop_mode, fps, frames=(N, 6+29)) where each row is
    [root_pos(3), root_rot_expmap(3), dof_pos(29)]

DoF order is verified identical between ember-lab dof_names and MimicKit g1.xml joint order
(left_hip_pitch/roll/yaw → ... → right_wrist_yaw), so no reordering needed.

Usage:
  python scripts/lafan_g1_npz_to_mimickit.py \
      --input  dependencies/MimicKit/data/motions/g1_extra/ember_lab/LAFAN_fight1_subject2_0_-1.npz \
      --output dependencies/MimicKit/data/motions/g1/lafan_fight1.pkl
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dependencies", "MimicKit"))
from mimickit.anim.motion import LoopMode, Motion
from mimickit.util.torch_util import quat_to_exp_map


def npz_to_mimickit(
    in_path: str,
    out_path: str,
    loop: str = "wrap",
    start_frame: int = 0,
    end_frame: int = -1,
) -> None:
    arr = np.load(in_path, allow_pickle=True)
    fps = int(arr["fps"].item() if arr["fps"].ndim else arr["fps"])
    dof_pos = arr["dof_positions"].astype(np.float32)             # (N, 29)
    body_rot_wxyz = arr["body_rotations"].astype(np.float32)      # (N, 30, 4) WXYZ
    body_pos = arr["body_positions"].astype(np.float32)           # (N, 30, 3)

    if end_frame == -1:
        end_frame = dof_pos.shape[0]
    assert 0 <= start_frame < end_frame <= dof_pos.shape[0], \
        f"bad slice [{start_frame},{end_frame}] vs N={dof_pos.shape[0]}"
    dof_pos = dof_pos[start_frame:end_frame]
    body_rot_wxyz = body_rot_wxyz[start_frame:end_frame]
    body_pos = body_pos[start_frame:end_frame]

    n = dof_pos.shape[0]
    assert dof_pos.shape[1] == 29, f"expected 29 DoF, got {dof_pos.shape[1]}"
    assert body_rot_wxyz.shape == (n, 30, 4)
    assert body_pos.shape == (n, 30, 3)

    root_pos = body_pos[:, 0, :]                                  # (N, 3)
    root_quat_wxyz = body_rot_wxyz[:, 0, :]                       # (N, 4) WXYZ
    # MimicKit quat_to_exp_map expects XYZW order
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    root_expmap = quat_to_exp_map(torch.from_numpy(root_quat_xyzw)).numpy()  # (N, 3)

    frames = np.concatenate([root_pos, root_expmap, dof_pos], axis=1)
    assert frames.shape == (n, 35), f"got frames {frames.shape}, expected (N, 35)"

    loop_mode = LoopMode.WRAP if loop == "wrap" else LoopMode.CLAMP
    motion = Motion(loop_mode=loop_mode, fps=fps, frames=frames)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    motion.save(out_path)

    print(f"✓ {in_path}")
    print(f"  → {out_path}")
    print(f"  frames={n}  fps={fps}  duration={n/fps:.2f}s  loop={loop}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--loop", default="wrap", choices=["wrap", "clamp"])
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    a = p.parse_args()
    npz_to_mimickit(a.input, a.output, a.loop, a.start_frame, a.end_frame)


if __name__ == "__main__":
    main()
