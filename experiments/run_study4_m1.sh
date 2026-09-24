#!/bin/bash
# Study 4 M1: 4c retrain queue (S2c/S7c/S8c, levels {1,2,4}, seed 1042).
# Session-independent: launch with
#   nohup bash experiments/run_study4_m1.sh > experiments/logs/study4_m1.log 2>&1 &
# Resumable: train.py resumes from last.pt; completed runs (final.pt) are
# skipped. Per-run auto commit+push.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
LOG=experiments/logs/study4_m1_stages.log

mark() { echo "$(date -Is) $1" >> "$LOG"; }

for SYS in S2c S7c S8c; do
  RUN="${SYS}_llama_seed1042"
  if [ -f "experiments/study4/${RUN}/final.pt" ]; then
    mark "SKIP ${RUN} (final.pt exists)"
    continue
  fi
  mark "TRAIN_START ${RUN}"
  conda run --no-capture-output -n mirror-rec python src/training/train.py \
    --config "configs/study4/${RUN}.yaml" \
    > "experiments/logs/train_${RUN}.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    mark "TRAIN_FAIL ${RUN} rc=${RC} (log kept, not retried - failure retention)"
    continue
  fi
  mark "TRAIN_DONE ${RUN}"
  git add "experiments/study4/${RUN}/config.yaml" "experiments/study4/${RUN}/train_log.jsonl" 2>/dev/null
  git commit -m "study4 M1: ${RUN} training complete (levels {1,2,4})" >/dev/null 2>&1 \
    && git push https://anonymous.invalid/anon-repo.git master >/dev/null 2>&1
  mark "COMMITTED ${RUN}"
done
mark "M1_QUEUE_DONE"
