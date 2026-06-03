#!/usr/bin/env bash
# Two-process MuJoCo sim2sim preview of GEAR-SONIC on the Unitree G1, keyboard-driven.
# This is the deployment-faithful path (same binary you'd run on the real robot), but
# it has HEAVY prerequisites that gear_sonic_setup.sh does NOT cover:
#
#   1. The C++ deployment binary must be BUILT  (TensorRT + cmake; see the repo's
#      Installation Guide / install_scripts — `gear_sonic_deploy/` ships only source).
#   2. `just` task runner installed (deploy.sh ends in `just run g1_deploy_onnx_ref ...`).
#   3. DDS on loopback needs multicast:  sudo ip link set lo multicast on
#      (otherwise run_sim_loop.py dies with CycloneDDS "create domain error").
#
# If you just want to WATCH the policy, use gear_sonic_preview.sh (single-process Isaac
# Sim viewer — no DDS, no C++ build).
#
#   keyboard:  ] start · 9 drop robot · T play motion · N/P next/prev · R restart · O e-stop
#   docs:      https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html  (Sim2Sim in MuJoCo)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
export DISPLAY="${DISPLAY:-:0}"

if [[ ! -d "$WBC_DIR" ]]; then
  echo "[preview-sim2sim] $WBC_DIR missing — run scripts/gear_sonic_setup.sh first."; exit 1
fi
cd "$WBC_DIR"

# DDS on loopback needs multicast, else the sim loop crashes on domain init.
if ! ip link show lo 2>/dev/null | grep -q MULTICAST; then
  echo "[preview-sim2sim] WARNING: 'lo' has no MULTICAST flag — CycloneDDS will fail."
  echo "  fix:  sudo ip link set lo multicast on"
fi

echo "[preview-sim2sim] Terminal-1 (sim loop, background) + Terminal-2 (deploy, foreground)…"
echo "  keyboard in the MuJoCo window:  ] start · 9 drop · T play · N/P next/prev · R restart · O stop"

# Terminal-1 equivalent: the MuJoCo simulator loop in its own venv.
if [[ -f .venv_sim/bin/activate ]]; then
  ( source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py ) \
    > /tmp/gear_sonic_simloop.log 2>&1 &
  SIM_PID=$!
  trap 'kill -9 $SIM_PID 2>/dev/null' EXIT
  sleep 8
else
  echo "[preview-sim2sim] .venv_sim not found — run gear_sonic_setup.sh / install_mujoco_sim.sh."; exit 1
fi

# Terminal-2 equivalent: the deployment binary. deploy.sh lives in gear_sonic_deploy/.
if [[ -f gear_sonic_deploy/deploy.sh ]]; then
  ( cd gear_sonic_deploy && bash deploy.sh sim )
else
  echo "[preview-sim2sim] gear_sonic_deploy/deploy.sh not found — inspect repo layout."; exit 1
fi

echo "[preview-sim2sim] viewer closed."
