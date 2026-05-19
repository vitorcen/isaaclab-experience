"""Generic OmniGibson example launcher.

Usage:
    python scripts/launch_demo.py <module_path> [--input-sleep N] [--random]

Where <module_path> is the dotted module path of the demo, e.g.
    omnigibson.examples.object_states.slicing_demo
    omnigibson.examples.robots.grasping_mode_example

Behavior:
- Replaces builtins.input(...) with a `time.sleep(N)` so demos that pause for
  ENTER auto-advance and you can watch the result.
- If --random is given, calls the demo's main() with random_selection=True so
  that any choose_from_options menus pick a default automatically.
"""

import argparse
import builtins
import importlib
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="Dotted module path, e.g. omnigibson.examples.object_states.slicing_demo")
    parser.add_argument("--input-sleep", type=int, default=15, help="Seconds to wait at each input() prompt")
    parser.add_argument("--random", action="store_true", help="Pass random_selection=True to main()")
    args = parser.parse_args()

    sleep_s = args.input_sleep

    def patched_input(prompt=""):
        print(f"[auto] {prompt}  (sleeping {sleep_s}s)")
        sys.stdout.flush()
        time.sleep(sleep_s)
        return ""

    builtins.input = patched_input

    mod = importlib.import_module(args.module)

    if args.random:
        mod.main(random_selection=True, headless=False, short_exec=False)
    else:
        mod.main()


if __name__ == "__main__":
    main()
