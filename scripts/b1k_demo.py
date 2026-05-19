"""BEHAVIOR-1K demo launcher CLI.

Usage:
    python scripts/b1k_demo.py list
    python scripts/b1k_demo.py launch <name>
    python scripts/b1k_demo.py status
    python scripts/b1k_demo.py kill

The notebook BEHAVIOR-1K.ipynb calls this via `!python scripts/b1k_demo.py ...`.
Demos run as detached subprocesses; the PID is tracked in /tmp/b1k_demo.pid.
Only one demo runs at a time.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONDA_PY = "/home/david/miniconda3/envs/behavior/bin/python"
DISPLAY = ":1"
PID_FILE = Path("/tmp/b1k_demo.pid")
LOG_FILE = Path("/tmp/b1k_demo.log")

# Registry: name -> dict with keys
#   kind: "scene" | "module"
#   value: scene model name (for kind=scene) OR dotted module path (for kind=module)
#   full: optional bool, for scenes; if True, load full scene
#   random: optional bool, for modules; pass random_selection=True
DEMOS = {
    # Scenes (use scripts/launch_b1k_scene.py)
    "scene:Rs_int":             {"kind": "scene", "value": "Rs_int"},
    "scene:Beechwood_0_int":    {"kind": "scene", "value": "Beechwood_0_int"},
    "scene:house_single_floor": {"kind": "scene", "value": "house_single_floor"},
    "scene:grocery_store_cafe": {"kind": "scene", "value": "grocery_store_cafe"},
    "scene:Rs_int_full":        {"kind": "scene", "value": "Rs_int", "full": True},

    # Robots & manipulation (high-level demos)
    "robot:grasping":           {"kind": "module", "value": "omnigibson.examples.robots.grasping_mode_example", "random": True},
    "robot:control":            {"kind": "module", "value": "omnigibson.examples.robots.robot_control_example",  "random": True},
    "robot:visualizer":         {"kind": "module", "value": "omnigibson.examples.robots.all_robots_visualizer",  "random": True},

    # Robot zoo — single robot in empty scene, random actions (uses scripts/launch_robot.py)
    # Arms
    "robot:franka":             {"kind": "robot", "value": "franka"},      # Franka Panda 7-DOF
    "robot:ur5e":               {"kind": "robot", "value": "ur5e"},        # Universal Robots UR5e
    "robot:kinova":             {"kind": "robot", "value": "kinova"},      # Kinova Gen3
    "robot:vx300s":             {"kind": "robot", "value": "vx300s"},      # Interbotix ViperX 300s
    "robot:a1":                 {"kind": "robot", "value": "a1"},          # AgileX A1 / PiPER 6-DOF (Inspire-hand capable)
    # Humanoid / bimanual
    "robot:r1":                 {"kind": "robot", "value": "r1"},          # UCSD R1
    "robot:r1pro":              {"kind": "robot", "value": "r1pro"},       # UCSD R1 Pro
    # Mobile manipulators
    "robot:tiago":              {"kind": "robot", "value": "tiago"},       # PAL Tiago
    "robot:stretch":            {"kind": "robot", "value": "stretch"},     # Hello Robot Stretch
    "robot:fetch":              {"kind": "robot", "value": "fetch"},       # Fetch
    # Mobile bases
    "robot:husky":              {"kind": "robot", "value": "husky"},       # Clearpath Husky
    "robot:turtlebot":          {"kind": "robot", "value": "turtlebot"},   # TurtleBot
    "robot:locobot":            {"kind": "robot", "value": "locobot"},     # LoCoBot
    "robot:freight":            {"kind": "robot", "value": "freight"},     # Fetch Freight

    # Object states / transition rules
    "state:slicing":            {"kind": "module", "value": "omnigibson.examples.object_states.slicing_demo"},
    "state:dicing":             {"kind": "module", "value": "omnigibson.examples.object_states.dicing_demo"},
    "state:onfire":             {"kind": "module", "value": "omnigibson.examples.object_states.onfire_demo"},
    "state:heated":             {"kind": "module", "value": "omnigibson.examples.object_states.heated_state_demo"},
    "state:water_particles":    {"kind": "module", "value": "omnigibson.examples.object_states.particle_source_sink_demo"},
    "state:particle_applier":   {"kind": "module", "value": "omnigibson.examples.object_states.particle_applier_remover_demo"},
    "state:cloth_fold":         {"kind": "module", "value": "omnigibson.examples.object_states.folded_unfolded_state_demo"},
    "state:attachment":         {"kind": "module", "value": "omnigibson.examples.object_states.attachment_demo"},
    "state:temperature":        {"kind": "module", "value": "omnigibson.examples.object_states.temperature_demo"},
    "state:textures":           {"kind": "module", "value": "omnigibson.examples.object_states.object_state_texture_demo"},

    # Object viewing
    "obj:visualize":            {"kind": "module", "value": "omnigibson.examples.objects.visualize_object",        "random": True},
    "obj:cloth_configs":        {"kind": "module", "value": "omnigibson.examples.objects.view_cloth_configurations","random": True},
    "obj:bbox":                 {"kind": "module", "value": "omnigibson.examples.objects.draw_bounding_box",       "random": True},

    # Environments / tasks
    "env:navigation":           {"kind": "module", "value": "omnigibson.examples.environments.navigation_env_demo", "random": True},
    "env:behavior":             {"kind": "module", "value": "omnigibson.examples.environments.behavior_env_demo",   "random": True},
}


def _running_pid():
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def cmd_list(_args):
    width = max(len(k) for k in DEMOS)
    for name, spec in DEMOS.items():
        if spec["kind"] == "scene":
            extra = " (full)" if spec.get("full") else ""
            desc = f"scene {spec['value']}{extra}"
        elif spec["kind"] == "robot":
            desc = f"robot model {spec['value']} (empty scene, random actions)"
        else:
            extra = " [random]" if spec.get("random") else ""
            desc = f"{spec['value']}{extra}"
        print(f"  {name:<{width}}  {desc}")


def cmd_status(_args):
    pid = _running_pid()
    if pid is None:
        print("idle")
        return
    name = "?"
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            name = f.read().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        pass
    print(f"running pid={pid}  cmd={name}")


def cmd_kill(_args):
    pid = _running_pid()
    if pid is None:
        print("nothing to kill")
        PID_FILE.unlink(missing_ok=True)
        return
    print(f"killing pid={pid} (process group)...")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(20):
        if _running_pid() is None:
            break
        time.sleep(0.5)
    if _running_pid() is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    print("stopped")


def cmd_launch(args):
    name = args.name
    if name not in DEMOS:
        print(f"unknown demo: {name}", file=sys.stderr)
        print("run `python scripts/b1k_demo.py list` to see available demos.", file=sys.stderr)
        sys.exit(2)
    if _running_pid() is not None:
        print(f"another demo is already running (pid {_running_pid()}). run `kill` first.", file=sys.stderr)
        sys.exit(1)

    spec = DEMOS[name]
    if spec["kind"] == "scene":
        cmd = [CONDA_PY, "-u", str(REPO / "scripts" / "launch_b1k_scene.py"), "--scene", spec["value"]]
        if spec.get("full"):
            cmd.append("--full")
    elif spec["kind"] == "robot":
        cmd = [CONDA_PY, "-u", str(REPO / "scripts" / "launch_robot.py"), "--robot", spec["value"]]
    else:
        cmd = [CONDA_PY, "-u", str(REPO / "scripts" / "launch_demo.py"), spec["value"],
               "--input-sleep", str(args.input_sleep)]
        if spec.get("random"):
            cmd.append("--random")

    env = {**os.environ, "DISPLAY": DISPLAY}
    log = open(LOG_FILE, "wb")
    print(f"launching {name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {LOG_FILE}")
    p = subprocess.Popen(cmd, env=env, cwd=str(REPO), start_new_session=True,
                         stdout=log, stderr=subprocess.STDOUT)
    PID_FILE.write_text(str(p.pid))
    print(f"  pid: {p.pid}")
    print(f"Isaac Sim takes 10-30s to launch; viewer will appear on DISPLAY={DISPLAY}.")
    print("use `python scripts/b1k_demo.py kill` to stop, or `... status` to check.")


def cmd_setup(_args):
    """Download / verify BEHAVIOR-1K + OmniGibson robot-assets datasets.

    Both downloads are idempotent — the helpers print "already installed" and
    skip if the target directory already exists.
    """
    cmd = [
        CONDA_PY, "-u", "-c",
        "from omnigibson.utils.asset_utils import "
        "download_omnigibson_robot_assets, download_behavior_1k_assets; "
        "download_omnigibson_robot_assets(); "
        "download_behavior_1k_assets(accept_license=True)",
    ]
    print("running dataset setup (downloads ~30 GB if missing)")
    print(f"  cmd: {' '.join(cmd[:3])} ...")
    subprocess.run(cmd, cwd=str(REPO), check=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("kill").set_defaults(func=cmd_kill)
    sub.add_parser("setup").set_defaults(func=cmd_setup)
    launch_p = sub.add_parser("launch")
    launch_p.add_argument("name")
    launch_p.add_argument("--input-sleep", type=int, default=15)
    launch_p.set_defaults(func=cmd_launch)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
