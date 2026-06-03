#!/usr/bin/env python3
"""Dump GR00T's PREDICTED 64-dim SONIC motion_token trajectory per motion.

Open-loop: feed each motion's (ego_view, prompt, state) from the LeRobot dataset to the
finetuned GR00T policy, collect the predicted motion_token per frame, save to npz. These
get injected into the SONIC WBC decode in the Isaac viewer (scripts/gear_sonic_inject.sh)
so you can SEE what the current GR00T checkpoint actually drives.

Run in the Isaac-GR00T venv:
    cd dependencies/Isaac-GR00T
    COMPILE_ACTION_HEAD_DISABLE=1 .venv/bin/python \
        ../../scripts/gr00t_dump_pred_tokens.py \
        --model_path ../../outputs/gr00t_sonic_derisk/checkpoint-2000 \
        --dataset_path ../../datasets/sonic_vla_lerobot \
        --out ../../datasets/sonic_vla_pred
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
import os

import numpy as np

# traj_id -> SONIC robot_filtered motion key (dataset episode order = convert order)
TRAJ_TO_KEY = [
    "dance_in_da_party_001__A464",
    "forward_lunge_R_001__A359_M",
    "macarena_001__A545",
    "neutral_kick_R_001__A543",
    "squat_001__A359",
    "tired_one_leg_jumping_R_001__A359",
    "walking_quip_360_R_002__A428",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--action_horizon", type=int, default=40)
    ap.add_argument("--traj_ids", type=int, nargs="+", default=list(range(7)))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    emb = EmbodimentTag.resolve("unitree_g1_sonic")
    policy = Gr00tPolicy(
        embodiment_tag=emb,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    os.makedirs(args.out, exist_ok=True)
    AH = args.action_horizon

    for traj_id in args.traj_ids:
        traj = loader[traj_id]
        T = len(traj)
        mod_no_action = deepcopy(loader.modality_configs)
        mod_no_action.pop("action")
        preds = []
        for step in range(0, T, AH):
            dp = extract_step_data(traj, step, mod_no_action, emb)
            obs = {}
            for k, v in dp.states.items():
                obs[f"state.{k}"] = v
            for k, v in dp.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lk in loader.modality_configs["language"].modality_keys:
                obs[lk] = dp.text
            parsed = parse_observation_gr00t(obs, loader.modality_configs)
            chunk, _ = policy.get_action(parsed)
            chunk = parse_action_gr00t(chunk)
            mt = np.asarray(chunk["action.motion_token"])  # (AH, 64)
            for j in range(AH):
                preds.append(mt[j])
        preds = np.asarray(preds)[:T].astype(np.float32)  # (T, 64)
        key = TRAJ_TO_KEY[traj_id] if traj_id < len(TRAJ_TO_KEY) else f"traj{traj_id}"
        out_path = os.path.join(args.out, f"{key}.npz")
        np.savez_compressed(out_path, motion_token=preds, motion_key=key, prompt=str(dp.text))
        logging.info(f"[dump] traj {traj_id} ({key}): pred_token {preds.shape} -> {out_path}")

    print(f"[dump] done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
