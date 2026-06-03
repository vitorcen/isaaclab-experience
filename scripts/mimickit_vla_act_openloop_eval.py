"""Open-loop go/pivot check for the distilled ACT student (runs in the lerobot env,
no Isaac, no env-split). Cheap first signal BEFORE building the closed-loop bridge.

Answers two questions:
  1. Did ACT learn the teacher's state+vision -> action mapping?  (action MSE vs ground truth)
  2. Does the language/vision conditioning actually separate motions?  (does feeding a
     fight frame vs a dance frame at a *matched proprio phase* yield different actions?)

If open-loop already fails (high MSE, or fight≈dance outputs), pivot now — don't build
the closed-loop eval. If it passes, the closed-loop bridge is worth the env-split work.

  conda run -n lerobot python scripts/mimickit_vla_act_openloop_eval.py \
    --ckpt outputs/act_g1_lafan_sanity/checkpoints/last/pretrained_model \
    --dataset_root datasets/g1-lafan-vla-sanity-img \
    --repo_id vitorcen/g1-lafan-vla-sanity-img
"""

import argparse

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="…/pretrained_model dir")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=200, help="frames to score per task")
    args = ap.parse_args()

    device = args.device
    policy = ACTPolicy.from_pretrained(args.ckpt).to(device).eval()
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)

    # group frame indices by task
    by_task = {}
    for i in range(ds.num_frames):
        t = ds[i]["task"]
        by_task.setdefault(t, []).append(i)
    tasks = sorted(by_task)
    print(f"tasks ({len(tasks)}):")
    for t in tasks:
        print(f"  [{len(by_task[t])} frames] {t}")

    img_key = "observation.images.front"

    def predict(frame):
        """single-frame action (first of the predicted chunk), reset queues each call."""
        policy.reset()
        batch = {
            "observation.state": frame["observation.state"].unsqueeze(0).to(device),
            img_key: frame[img_key].unsqueeze(0).to(device),
            "task": [frame["task"]],
        }
        with torch.no_grad():
            a = policy.select_action(batch)
        return a.squeeze(0).cpu().numpy()

    # 1) action MSE vs teacher ground truth, per task
    print("\n=== 1) action MSE vs teacher (lower = ACT fit the mapping) ===")
    rng = np.random.default_rng(0)
    task_pred = {}
    for t in tasks:
        idxs = by_task[t]
        sample = rng.choice(idxs, size=min(args.n, len(idxs)), replace=False)
        errs, preds = [], []
        for i in sample:
            fr = ds[int(i)]
            pred = predict(fr)
            gt = fr["action"].numpy()
            errs.append(np.mean((pred - gt) ** 2))
            preds.append(pred)
        task_pred[t] = np.stack(preds)
        gt_std = np.std([ds[int(i)]["action"].numpy() for i in sample])
        print(f"  {t[:40]:40s}  MSE={np.mean(errs):.4f}  (action std≈{gt_std:.3f})")

    # 2) cross-task separation: are the action distributions per task actually different?
    print("\n=== 2) task separation (do prompts produce different actions?) ===")
    means = {t: task_pred[t].mean(0) for t in tasks}
    if len(tasks) >= 2:
        a, b = tasks[0], tasks[1]
        between = np.linalg.norm(means[a] - means[b])
        within = 0.5 * (task_pred[a].std(0).mean() + task_pred[b].std(0).mean())
        print(f"  mean-action L2 distance  {a[:20]} vs {b[:20]}: {between:.3f}")
        print(f"  within-task action std (avg):                  {within:.3f}")
        print(f"  separation ratio (between/within): {between / (within + 1e-6):.2f}  "
              f"(>1 ⇒ prompts drive distinct motions)")

    print("\n[verdict hint] low MSE + separation ratio >1 ⇒ GO (build closed-loop). "
          "High MSE or ratio≈0 ⇒ PIVOT (vision/phase/arch).")


if __name__ == "__main__":
    main()
