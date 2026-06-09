"""Convert MimicKit rollout episodes (npz produced by mimickit/rollout_record.py)
into a LeRobot dataset + GR00T-style meta/modality.json.

Run in the lerobot env (NOT isaaclab — lerobot is not installed there):
  conda run -n lerobot-v040 python scripts/mimickit_episodes_to_lerobot.py \
    --rollout_dirs dependencies/MimicKit/output/vla_rollouts/fight,dependencies/MimicKit/output/vla_rollouts/dance \
    --repo_id  vitorcen/g1-lafan-vla-sanity \
    --root     datasets/g1-lafan-vla-sanity

Each rollout dir holds episode_*.npz (action, state, rgb, phase, task, fps) and a
shared rollout_meta.json (state/action index layout). We emit one LeRobot episode per
npz, then write meta/modality.json so GR00T can map state/action/video/annotation keys.
The first student (ACT/DP) trains straight from this dataset in lerobot-v040.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np

import lerobot.datasets.lerobot_dataset as _lrd
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _ffmpeg_bin():
    # prefer the active env's ffmpeg (PATH), fall back to system ffmpeg
    return shutil.which("ffmpeg") or "/usr/bin/ffmpeg"


def _patch_encode_cli():
    """Replace lerobot's in-process PyAV video encoder with an ffmpeg-CLI subprocess.

    Root cause of the intermittent SIGSEGV / pandas __finalize__ / sre_compile global
    stomp / datasets schema crash (all moving around, all only at 20+ episodes): PyAV
    (encode) and torchcodec (decode-for-stats) link two different ffmpeg builds into one
    process. Repeated in-process encode calls trip the symbol/heap conflict. Moving the
    encoder into a clean subprocess leaves torchcodec as the only in-process ffmpeg user.
    Output: libx264 / yuv420p / faststart — stable and widely decodable."""
    if not hasattr(_lrd, "encode_video_frames"):
        return  # newer lerobot (0.5.x) doesn't expose it here / image datasets skip encode
    ffmpeg = _ffmpeg_bin()
    assert ffmpeg, "no usable ffmpeg binary with libx264 found"

    def _cli_encode(imgs_dir, video_path, fps, vcodec="h264", pix_fmt="yuv420p",
                    g=2, crf=30, overwrite=False, **kwargs):
        imgs_dir, video_path = str(imgs_dir), str(video_path)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        files = sorted(glob.glob(os.path.join(imgs_dir, "frame-" + "[0-9]" * 6 + ".png")))
        if not files:
            raise FileNotFoundError(f"No images found in {imgs_dir}")
        start = int(os.path.basename(files[0]).split("-")[-1].split(".")[0])
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-start_number", str(start),
               "-i", os.path.join(imgs_dir, "frame-%06d.png"),
               "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", str(crf),
               "-g", str(g), "-movflags", "+faststart", "-loglevel", "error", video_path]
        subprocess.run(cmd, check=True)

    _lrd.encode_video_frames = _cli_encode


def _patch_task_index():
    """lerobot v0.4 get_task_index uses `self.tasks.loc[task].task_index`
    (lerobot_dataset.py:301), which on pandas 2.3.x intermittently raises
    `IndexError: tuple index out of range` from `__finalize__` (the `.loc[label]` →
    Series → `.col` path goes through `xs`). Use the scalar `.at` accessor instead —
    it returns the value directly and never touches that code path."""
    Meta = getattr(_lrd, "LeRobotDatasetMetadata", None)
    if Meta is None or not hasattr(Meta, "get_task_index"):
        return  # different internals (e.g. lerobot 0.5.x) — nothing to patch

    def get_task_index(self, task):
        if self.tasks is None or task not in self.tasks.index:
            return None
        return int(self.tasks.at[task, "task_index"])

    Meta.get_task_index = get_task_index


def state_feature_names(layout):
    """Human-readable per-dim names from the [name, start, end] layout."""
    names = []
    for name, start, end in layout:
        n = end - start
        if name == "root_ang_vel":
            names += ["root_ang_vel.x", "root_ang_vel.y", "root_ang_vel.z"]
        elif name == "projected_gravity":
            names += ["proj_grav.x", "proj_grav.y", "proj_grav.z"]
        elif name == "phase":
            names += ["phase.sin", "phase.cos"]
        else:
            names += ["{}.{}".format(name, i) for i in range(n)]
    return names


def build_modality_json(meta):
    """GR00T meta/modality.json: name->{start,end} groups for state/action, plus
    video key map and the language annotation pointer."""
    state = {name: {"start": start, "end": end} for name, start, end in meta["state_layout"]}
    action = {name: {"start": start, "end": end} for name, start, end in meta["action_layout"]}
    return {
        "state": state,
        "action": action,
        "video": {"front": {"original_key": meta["image_key"]}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_dirs", required=True,
                    help="comma-separated rollout dirs (one per motion/task)")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", required=True, help="local output dataset dir")
    ap.add_argument("--robot_type", default="unitree_g1")
    ap.add_argument("--fps", type=int, default=None, help="override fps (else from rollout_meta)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_video", action="store_true",
                    help="store frames as images (dtype=image, use_videos=False) instead "
                         "of mp4. Escapes the ffmpeg decode instability that crashes ACT "
                         "training in this env; use for the ACT/DP sanity. Keep video off "
                         "for the GR00T path (it wants mp4).")
    args = ap.parse_args()

    _patch_encode_cli()
    _patch_task_index()

    dirs = [d.strip() for d in args.rollout_dirs.split(",") if d.strip()]
    assert dirs, "no rollout dirs"

    with open(os.path.join(dirs[0], "rollout_meta.json")) as f:
        meta = json.load(f)
    fps = args.fps or meta["fps"]
    ndof = meta["ndof"]
    state_dim = meta["state_dim"]
    H, W = meta["image_size"]
    img_key = meta["image_key"]

    if args.overwrite and os.path.exists(args.root):
        shutil.rmtree(args.root)

    # shapes MUST be tuples: lerobot v0.4.0 validate_frame compares value.shape (tuple)
    # against feature["shape"] with !=, and a list never equals a tuple.
    img_dtype = "image" if args.no_video else "video"
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,),
                              "names": state_feature_names(meta["state_layout"])},
        "action": {"dtype": "float32", "shape": (ndof,),
                   "names": ["joint_target.{}".format(i) for i in range(ndof)]},
        img_key: {"dtype": img_dtype, "shape": (H, W, 3),
                  "names": ["height", "width", "channels"]},
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=args.root,
        robot_type=args.robot_type,
        use_videos=not args.no_video,
    )

    total_eps = 0
    total_frames = 0
    for d in dirs:
        eps = sorted(glob.glob(os.path.join(d, "episode_*.npz")))
        print(f"[convert] {d}: {len(eps)} episodes")
        for ep_path in eps:
            z = np.load(ep_path, allow_pickle=True)
            action = z["action"].astype(np.float32)
            state = z["state"].astype(np.float32)
            rgb = z["rgb"]  # [T,H,W,3] uint8
            task = str(z["task"])
            T = action.shape[0]
            for t in range(T):
                dataset.add_frame({
                    "observation.state": state[t],
                    "action": action[t],
                    img_key: rgb[t],
                    "task": task,
                })
            dataset.save_episode()
            total_eps += 1
            total_frames += T
        print(f"[convert]   running total: {total_eps} eps / {total_frames} frames")

    # GR00T modality.json (harmless for ACT/DP; required for GR00T phase-2)
    modality = build_modality_json(meta)
    meta_dir = os.path.join(args.root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "modality.json"), "w") as f:
        json.dump(modality, f, indent=2)

    print(f"[convert] DONE: {total_eps} episodes, {total_frames} frames -> {args.root}")
    print(f"[convert] wrote meta/modality.json  (state_dim={state_dim}, ndof={ndof}, fps={fps})")


if __name__ == "__main__":
    main()
