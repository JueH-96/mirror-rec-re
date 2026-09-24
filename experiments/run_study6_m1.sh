#!/bin/bash
# Study 6 M1: 4c 3-seed backfill retrain queue (S2c/S7c/S8c x seeds 1043,1044,
# levels {1,2,4}; seed 1042 already trained in Study 4).
# Session-independent: launch with
#   nohup bash experiments/run_study6_m1.sh > experiments/logs/study6_m1.log 2>&1 &
# Resumable: train.py resumes from last.pt; completed runs (final.pt) are
# skipped. Per-run auto commit+push.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
LOG=experiments/logs/study6_m1_stages.log

mark() { echo "$(date -Is) $1" >> "$LOG"; }

for SEED in 1043 1044; do
  for SYS in S2c S7c S8c; do
    RUN="${SYS}_llama_seed${SEED}"
    if [ -f "experiments/study6/${RUN}/final.pt" ]; then
      mark "SKIP ${RUN} (final.pt exists)"
      continue
    fi
    mark "TRAIN_START ${RUN}"
    # timestamped per-attempt log: reruns must never overwrite a failure scene
    # (user directive 2026-08-07 after two rc=1 logs were lost to >)
    TLOG="experiments/logs/train_${RUN}_$(date +%Y%m%dT%H%M%S).log"
    conda run --no-capture-output -n mirror-rec python src/training/train.py \
      --config "configs/study6/${RUN}.yaml" \
      > "$TLOG" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      mark "TRAIN_FAIL ${RUN} rc=${RC} (log kept, not retried - failure retention)"
      continue
    fi
    mark "TRAIN_DONE ${RUN}"
    git add "experiments/study6/${RUN}/config.yaml" "experiments/study6/${RUN}/train_log.jsonl" 2>/dev/null
    git commit -m "study6 M1: ${RUN} training complete (levels {1,2,4} backfill)" >/dev/null 2>&1 \
      && git push https://anonymous.invalid/anon-repo.git master >/dev/null 2>&1
    mark "COMMITTED ${RUN}"
  done
done
mark "M1_QUEUE_DONE"
