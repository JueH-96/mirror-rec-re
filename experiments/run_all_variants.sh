#!/bin/bash
# Usage: run_all_variants.sh <model_key>
# Runs all 4 prompt variants sequentially, then per-variant metrics.
set -u
M="$1"
LOG=${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}/experiments/logs/run_${M}.log
cd ${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}
for v in plain structured explain_then_rank direction_constraint; do
  echo "=== $(date -Is) $M / $v ===" >> "$LOG"
  conda run -n mirror-rec python experiments/run_pilot.py --model "$M" --variant "$v" >> "$LOG" 2>&1
  conda run -n mirror-rec python src/metrics/relational_metrics.py --model "$M" --variant "$v" >> "$LOG" 2>&1
done
echo "=== $(date -Is) $M ALL_VARIANTS_DONE ===" >> "$LOG"
