"""Spawn a single OmniGibson robot in an empty scene and run random actions.

Usage:
    python scripts/launch_robot.py --robot franka
    python scripts/launch_robot.py --robot a1 --steps 5000
"""

import argparse

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.robots import REGISTERED_ROBOTS, Robot
from omnigibson.utils.ui_utils import KeyboardEventHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True, help=f"Robot model name. One of: {sorted(REGISTERED_ROBOTS)}")
    parser.add_argument("--steps", type=int, default=-1, help="Sim steps (-1 = run forever)")
    parser.add_argument("--scale", type=float, default=0.05, help="Action sample scale")
    args = parser.parse_args()

    if args.robot not in REGISTERED_ROBOTS:
        raise SystemExit(f"Unknown robot '{args.robot}'. Available: {sorted(REGISTERED_ROBOTS)}")

    env = og.Environment(configs={"scene": {"type": "Scene"}})

    og.sim.stop()
    robot = Robot(name=args.robot, model=args.robot, obs_modalities=[])
    env.scene.add_object(robot)
    og.sim.play()
    og.sim.step()
    robot.reset()
    robot.keep_still()

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([2.69918369, -3.63686664, 4.57894564]),
        orientation=th.tensor([0.39592411, 0.1348514, 0.29286304, 0.85982]),
    )
    og.sim.enable_viewer_camera_teleoperation()
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: og.shutdown(),
    )

    print(f"Loaded robot: {args.robot}.  Press ESC in viewer to quit.")

    step = 0
    while args.steps < 0 or step < args.steps:
        if step % 30 == 0:
            action = robot.action_space.sample() * args.scale
        env.step(action)
        step += 1

    og.shutdown()


if __name__ == "__main__":
    main()
