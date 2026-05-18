#!/usr/bin/env bash
# Background nvidia-smi sampler. Writes one CSV line per sample to $OUT.
# Stops itself when its pidfile is removed.
#
# Usage:
#   bash gpu_sampler.sh <out.csv> <pidfile> [interval_s=1]

set -euo pipefail
OUT="${1:?out path required}"
PIDFILE="${2:?pidfile required}"
INTERVAL="${3:-1}"

echo $$ > "$PIDFILE"
echo "ts,mem_used_mib,mem_total_mib,util_gpu_pct" > "$OUT"

while [[ -f "$PIDFILE" ]]; do
    line=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    ts=$(date +%s)
    echo "${ts},${line}" >> "$OUT"
    sleep "$INTERVAL"
done
