"""Non-interactive BEHAVIOR-1K scene launcher.

Loads a single scene (default: Rs_int, Quick load = structure only) and runs
the simulator with the viewer until interrupted.
"""

import argparse

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.constants import STRUCTURE_CATEGORIES
from omnigibson.utils.ui_utils import KeyboardEventHandler

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Rs_int")
    parser.add_argument("--full", action="store_true", help="Load all interactive objects (slower)")
    args = parser.parse_args()

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene,
        }
    }
    if not args.full:
        cfg["scene"]["load_object_categories"] = list(STRUCTURE_CATEGORIES)

    env = og.Environment(configs=cfg)

    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: og.shutdown(),
    )

    print(f"Loaded scene: {args.scene} (mode: {'Full' if args.full else 'Quick'})")
    print("Press ESC in the viewer to quit.")

    # OmniGibson loads the scene with simulator stopped; reset to enter playing state
    # (simulator.step asserts self.is_playing()).
    env.reset()

    while True:
        env.step([])


if __name__ == "__main__":
    main()
