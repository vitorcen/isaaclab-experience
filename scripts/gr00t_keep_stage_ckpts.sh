#!/usr/bin/env bash
# Preserve stage checkpoints (multiples of STAGE) permanently via hardlink into keep/,
# so HF's rolling save_total_limit prune can't lose them (zero extra disk: same inodes).
# Override OUT / STAGE via env. Run alongside training as a background watcher.
#   OUT=outputs/gr00t_sonic_8k STAGE=2000 bash scripts/gr00t_keep_stage_ckpts.sh
cd /home/david/work/isaaclab-experience
OUT=${OUT:-outputs/gr00t_sonic_8k}
STAGE=${STAGE:-2000}
KEEP=$OUT/keep
mkdir -p "$KEEP"
for _ in $(seq 1 600); do   # ~10h watcher
  for d in "$OUT"/checkpoint-*; do
    [ -d "$d" ] || continue
    step=$(basename "$d" | sed 's/checkpoint-//')
    if [ $((step % STAGE)) -eq 0 ] && [ ! -d "$KEEP/checkpoint-$step" ]; then
      # only preserve intact checkpoints
      if dependencies/Isaac-GR00T/.venv/bin/python scripts/gr00t_ckpt_intact.py "$d" >/dev/null 2>&1; then
        cp -al "$d" "$KEEP/checkpoint-$step" 2>/dev/null && \
          echo "[keep] hardlinked stage checkpoint-$step -> $KEEP/ @ $(date +%T)"
      fi
    fi
  done
  sleep 60
done
