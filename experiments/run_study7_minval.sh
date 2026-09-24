#!/bin/bash
# Study 7 permutation-augmentation MINIMAL VALIDATION (exploratory; user-
# authorized 2026-08-11; results do NOT enter conclusions — screening only).
# Trains S7perm/S8perm_llama_seed1042 (order_aug), evaluates test_main +
# 5 order splits, writes the minval memo. Idempotent: completed runs/evals
# are skipped. Per stage-gates discipline: final.pt + train_log max-steps
# entry are the ONLY completion evidence; per-attempt timestamped logs.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
STAGE_LOG=experiments/logs/study7_minval_stages.log
mark() { echo "$(date '+%F %T') $*" >> "$STAGE_LOG"; }

RUNS="S7perm_llama_seed1042 S8perm_llama_seed1042"
SPLITS="test_main test_order_p1 test_order_p2 test_order_p3 test_order_p4 test_order_p5"

for RUN in $RUNS; do
  CFG=configs/study7/${RUN}.yaml
  DIR=experiments/study7/${RUN}
  if [ -f "$DIR/final.pt" ] && grep -q '"step": 1500' "$DIR/train_log.jsonl" 2>/dev/null; then
    mark "SKIP_TRAIN $RUN (final.pt + step-1500 log verified)"
  else
    TS=$(date +%Y%m%d_%H%M%S)
    mark "TRAIN_START $RUN attempt log train_stdout.$TS.log"
    conda run --no-capture-output -n mirror-rec python src/training/train.py \
      --config "$CFG" > "experiments/logs/study7_${RUN}_train.$TS.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$DIR/final.pt" ]; then
      mark "TRAIN_FAIL $RUN rc=$rc"
      echo "MINVAL_FAILED" >> "$STAGE_LOG"
      exit 1
    fi
    mark "TRAIN_DONE $RUN"
    git add "$DIR/config.yaml" "$DIR/train_log.jsonl" 2>/dev/null
    git commit -m "study7 minval: ${RUN} training complete (order_aug, exploratory)" >/dev/null 2>&1 \
      && git push "https://anonymous.invalid/anon-repo.git" master >/dev/null 2>&1
  fi
done
mark "MINVAL_TRAIN_DONE (all runs verified)"

for RUN in $RUNS; do
  for SPLIT in $SPLITS; do
    OUT=results/study7/raw/${RUN}/${SPLIT}.jsonl
    N=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )
    if [ "$N" -ge 800 ]; then
      mark "SKIP_EVAL $RUN $SPLIT ($N rows)"
      continue
    fi
    mark "EVAL_START $RUN $SPLIT"
    conda run --no-capture-output -n mirror-rec python src/training/evaluate.py \
      --config configs/study7/${RUN}.yaml --ckpt experiments/study7/${RUN}/final.pt \
      --split "$SPLIT" --out "$OUT" \
      >> experiments/logs/study7_minval_eval.log 2>&1
    rc=$?
    N=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )
    if [ $rc -ne 0 ] || [ "$N" -lt 800 ]; then
      mark "EVAL_FAIL $RUN $SPLIT rc=$rc rows=$N"
      echo "MINVAL_FAILED" >> "$STAGE_LOG"
      exit 1
    fi
    mark "EVAL_DONE $RUN $SPLIT rows=$N"
  done
done

conda run --no-capture-output -n mirror-rec python src/statistics/study7_minval_memo.py \
  >> experiments/logs/study7_minval_eval.log 2>&1 || { mark "MEMO_FAIL"; echo "MINVAL_FAILED" >> "$STAGE_LOG"; exit 1; }
git add results/study7 reports/study7_minval_memo.md 2>/dev/null
git commit -m "study7 minval: S7perm/S8perm eval + exploratory memo (order_aug screening)" >/dev/null 2>&1 \
  && git push "https://anonymous.invalid/anon-repo.git" master >/dev/null 2>&1
mark "MINVAL_DONE"
