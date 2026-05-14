"""Realtime MuJoCo viewer for GR00T server inference on robocasa GR1 tabletop.

Reuses rollout_policy.py's create_eval_env + run_rollout_gymnasium_policy but
slips a ViewerWrapper underneath MultiStepWrapper, so every inner sim step is
mirrored to a live mujoco.viewer window (no mp4 recording).

Run via scripts/realtime_gr00t_viewer.sh (sets MUJOCO_GL=glfw and the right
venv python).
"""
import argparse
import os

import gymnasium as gym
import mujoco
import mujoco.viewer


class ViewerWrapper(gym.Wrapper):
    """Sync a passive mujoco viewer on every step (and reset)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # robocasa wraps robosuite; reach native mujoco objects
        robosuite_env = env.unwrapped.env
        sim = robosuite_env.sim
        self._model = sim.model._model
        self._data = sim.data._data
        print("[viewer] launching mujoco viewer ...", flush=True)
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        print(f"[viewer] launched, is_running={self._viewer.is_running()}", flush=True)

    def step(self, action):
        out = super().step(action)
        if self._viewer.is_running():
            self._viewer.sync()
        return out

    def reset(self, **kw):
        out = super().reset(**kw)
        # robosuite recreates sim on reset → rebind viewer's model/data
        sim = self.env.unwrapped.env.sim
        new_model, new_data = sim.model._model, sim.data._data
        if new_model is not self._model or new_data is not self._data:
            self._viewer.close()
            self._model, self._data = new_model, new_data
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        else:
            self._viewer.sync()
        return out

    def close(self):
        try:
            self._viewer.close()
        except Exception:
            pass
        return super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env")
    parser.add_argument("--policy_client_host", default="127.0.0.1")
    parser.add_argument("--policy_client_port", type=int, default=5555)
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=8)
    args = parser.parse_args()

    # Keep robosuite's offscreen path (EGL); the GLFW viewer window
    # we open below uses its own context, independent of MUJOCO_GL.
    os.environ.setdefault("MUJOCO_GL", "egl")

    from gr00t.eval.rollout_policy import (
        WrapperConfigs,
        MultiStepConfig,
        VideoConfig,
        get_robocasa_env_fn,
        create_gr00t_sim_policy,
        run_rollout_gymnasium_policy,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

    embodiment_tag = get_embodiment_tag_from_env_name(args.env_name)
    assert embodiment_tag == EmbodimentTag.GR1, f"viewer only validated for GR1 (got {embodiment_tag})"

    # Build env with viewer wrapper INSIDE MultiStepWrapper so every inner
    # sim step (one per env.step call) updates the live viewer.
    def make_env_fn():
        inner = get_robocasa_env_fn(args.env_name)
        def fn():
            env = inner()
            env = ViewerWrapper(env)
            return MultiStepWrapper(
                env,
                video_delta_indices=__import__("numpy").array([0]),
                state_delta_indices=__import__("numpy").array([0]),
                n_action_steps=args.n_action_steps,
                max_episode_steps=args.max_episode_steps,
                terminate_on_success=True,
            )
        return fn

    policy = create_gr00t_sim_policy(
        model_path="",
        embodiment_tag=embodiment_tag,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
    )

    # Reuse rollout loop but with our env_fn (no video wrapper at all)
    import gymnasium as gym
    from collections import defaultdict
    from tqdm import tqdm
    import numpy as np

    env = gym.vector.SyncVectorEnv([make_env_fn()])

    completed = 0
    successes = []
    obs, _ = env.reset()
    policy.reset()
    pbar = tqdm(total=args.n_episodes, desc="Episodes")
    while completed < args.n_episodes:
        actions, _ = policy.get_action(obs)
        obs, _, terms, truncs, infos = env.step(actions)
        if terms[0] or truncs[0]:
            succ = False
            if "success" in infos:
                v = infos["success"][0]
                succ = bool(np.any(v)) if hasattr(v, "__iter__") else bool(v)
            successes.append(succ)
            completed += 1
            pbar.update(1)
    pbar.close()

    print(f"success rate: {np.mean(successes) if successes else 0.0}")
    print("[viewer] holding window open — close it (or Ctrl-C) to exit")
    # keep viewer alive until user closes the window
    inner_env = env.envs[0]
    while hasattr(inner_env, "env"):
        if hasattr(inner_env, "_viewer"):
            v = inner_env._viewer
            try:
                while v.is_running():
                    v.sync()
                    import time; time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            break
        inner_env = inner_env.env
    env.close()


if __name__ == "__main__":
    main()
