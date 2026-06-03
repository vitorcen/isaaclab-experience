#!/usr/bin/env bash
# Self-healing resume loop for a GR00T finetune run: prune corrupt (reboot/crash-
# truncated) checkpoints, then resume from the latest INTACT one. Survives mid-save
# crashes. Override OUT / MAX_STEPS / SAVE_STEPS via env.
#   OUT=outputs/gr00t_sonic_8k MAX_STEPS=8000 SAVE_STEPS=500 bash scripts/gr00t_resume_heal.sh
cd /home/david/work/isaaclab-experience
OUT=${OUT:-outputs/gr00t_sonic_8k}
MAX_STEPS=${MAX_STEPS:-8000}
SAVE_STEPS=${SAVE_STEPS:-500}
GVENV=dependencies/Isaac-GR00T/.venv/bin/python
for attempt in $(seq 1 30); do
  # prune corrupt checkpoints (newest first)
  for ck in $(ls -d $OUT/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -rn); do
    if ! "$GVENV" scripts/gr00t_ckpt_intact.py "$OUT/checkpoint-$ck" >/dev/null 2>&1; then
      echo "[resume] checkpoint-$ck CORRUPT -> deleting"
      rm -rf "$OUT/checkpoint-$ck"
    fi
  done
  latest=$(ls -d $OUT/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  latest=${latest:-0}
  if [ "$latest" -ge "$MAX_STEPS" ]; then echo "[resume] reached $latest >= $MAX_STEPS, done"; break; fi
  echo "[resume] attempt $attempt: resume from intact ckpt=$latest @ $(date +%T)"
  MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS OUT_DIR=/home/david/work/isaaclab-experience/$OUT \
    bash scripts/gr00t_sonic_finetune.sh
  echo "[resume] run exited $? @ $(date +%T)"
  sleep 5
done
echo "[resume] LOOP DONE final=$(ls -d $OUT/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | tail -1)"
