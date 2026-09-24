#!/bin/bash
# Study 7 main chain (plan APPROVED 2026-08-12; Stage A + Stage B released).
#   Stage A : 6 ablation runs (seed 1042) -> test_main eval + metrics
#   SELECT  : mechanical Stage B entry rule (study7_select.py, plan Sec 5)
#   Stage Bа: selected ablations x seeds 1043/1044 -> test_main
#   Stage Bp: perm runs (S2perm x3, S7perm/S8perm x 1043/1044; seed-1042
#             S7perm/S8perm reused from minval) -> test_main + order p1..p5
#             + test_t4_word, + metrics backfill for minval raws
#   Stage C : S7cperm/S8cperm x 3 seeds (plan Sec 5b descriptive leg)
#   M3      : H7 stats + descriptive trade-off + report render
# Idempotent; hard per-run verification (final.pt + step-1500 train log,
# row-count gates); per-stage commit+push; markers in
# experiments/logs/study7_main_stages.log. GPU 1 only.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
STAGE_LOG=experiments/logs/study7_main_stages.log
EVAL_LOG=experiments/logs/study7_main_eval.log
PY="conda run --no-capture-output -n mirror-rec python"
PUSH_URL="https://x-access-token:${GITHUB_TOKEN:-}@anonymous.invalid/anon-repo.git"

mark() { echo "$(date '+%F %T') $*" >> "$STAGE_LOG"; }
fail() { mark "$*"; echo "$(date '+%F %T') MAIN_FAILED" >> "$STAGE_LOG"; exit 1; }
ci() {  # ci <message>  (targeted adds; *.pt never staged)
  git add results/study7 results/study7_statistical_tests.json \
    configs/study7 reports/study7_report.md experiments/study7_plan.md \
    experiment_manifest.yaml 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -m "$1" >/dev/null 2>&1 || mark "COMMIT_FAIL $1"
    git push "$PUSH_URL" master >/dev/null 2>&1 || mark "PUSH_FAIL $1"
    mark "COMMIT $1"
  fi
}

train_run() {  # train_run <run>
  local RUN=$1 CFG=configs/study7/$1.yaml DIR=experiments/study7/$1
  if [ -f "$DIR/final.pt" ] && grep -q '"step": 1500' "$DIR/train_log.jsonl" 2>/dev/null; then
    mark "SKIP_TRAIN $RUN (final.pt + step-1500 verified)"
    return 0
  fi
  local TS
  TS=$(date +%Y%m%d_%H%M%S)
  mark "TRAIN_START $RUN attempt log experiments/logs/study7_${RUN}_train.$TS.log"
  $PY src/training/train.py --config "$CFG" \
    > "experiments/logs/study7_${RUN}_train.$TS.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$DIR/final.pt" ] \
     || ! grep -q '"step": 1500' "$DIR/train_log.jsonl" 2>/dev/null; then
    fail "TRAIN_FAIL $RUN rc=$rc"
  fi
  mark "TRAIN_DONE $RUN"
  git add "$DIR/config.yaml" "$DIR/train_log.jsonl" 2>/dev/null
  git commit -m "study7: ${RUN} training complete" >/dev/null 2>&1 \
    && git push "$PUSH_URL" master >/dev/null 2>&1
}

rows_gate() { case "$1" in test_main) echo 3040 ;; test_t4_word) echo 1520 ;; *) echo 800 ;; esac; }
sets_for() { case "$1" in test_order_p*) echo "data_processed/study6/candidate_sets_$1.jsonl" ;; *) echo data_processed/study3/candidate_sets_test_main.jsonl ;; esac; }
elig_for() { case "$1" in test_main) echo data_processed/study3/eligible_sets_test_main.json ;; test_t4_word) echo data_processed/study4/eligible_sets_test_t4_word.json ;; *) echo "data_processed/study6/eligible_sets_$1.json" ;; esac; }

eval_run() {  # eval_run <run> <split>  (eval if raw short of gate; then metrics)
  local RUN=$1 SPLIT=$2
  local OUT=results/study7/raw/$1/$2.jsonl GATE N rc
  GATE=$(rows_gate "$SPLIT")
  N=$([ -f "$OUT" ] && wc -l < "$OUT" || echo 0)
  if [ "$N" -ge "$GATE" ]; then
    mark "SKIP_EVAL $RUN $SPLIT ($N rows)"
  else
    mark "EVAL_START $RUN $SPLIT"
    $PY src/training/evaluate.py --config "configs/study7/${RUN}.yaml" \
      --ckpt "experiments/study7/${RUN}/final.pt" --split "$SPLIT" \
      --out "$OUT" >> "$EVAL_LOG" 2>&1
    rc=$?
    N=$([ -f "$OUT" ] && wc -l < "$OUT" || echo 0)
    if [ $rc -ne 0 ] || [ "$N" -lt "$GATE" ]; then
      fail "EVAL_FAIL $RUN $SPLIT rc=$rc rows=$N"
    fi
    mark "EVAL_DONE $RUN $SPLIT rows=$N"
  fi
  local MP=results/study7/metrics_${RUN}_${SPLIT}.parquet
  if [ ! -f "$MP" ]; then
    $PY src/metrics/relational_metrics.py --config configs/pilot.yaml \
      --model "$RUN" --variant "$SPLIT" --raw "$OUT" \
      --sets "$(sets_for "$SPLIT")" --eligible "$(elig_for "$SPLIT")" \
      --out "$MP" >> "$EVAL_LOG" 2>&1 || fail "METRICS_FAIL $RUN $SPLIT"
    mark "METRICS_DONE $RUN $SPLIT"
  fi
}

mkdir -p experiments/logs results/study7/raw
mark "CHAIN_START pid=$$"

# ---- Stage A: ablation screening (seed 1042) --------------------------------
ABLS="S8c_ablg S8c_ablga S8c_ablicr S8c_ablchain S8c_ablresp S8c_ablot"
for a in $ABLS; do
  train_run "${a}_llama_seed1042"
  eval_run "${a}_llama_seed1042" test_main
  ci "study7 Stage A: ${a} seed1042 eval + metrics"
done
mark "STAGEA_DONE"

# ---- Mechanical Stage B selection (plan Sec 5 rule) -------------------------
$PY src/statistics/study7_select.py >> "$EVAL_LOG" 2>&1 || fail "SELECT_FAIL"
ci "study7: mechanical Stage B ablation selection (plan Sec 5 rule)"
mark "SELECTION_DONE selected=[$(tr '\n' ' ' < configs/study7/queue_stageB_ablation.txt)]"

# ---- Stage B ablations: selected x seeds 1043/1044 --------------------------
if [ -s configs/study7/queue_stageB_ablation.txt ]; then
  while IFS= read -r RUN; do
    [ -n "$RUN" ] || continue
    train_run "$RUN"
    eval_run "$RUN" test_main
    ci "study7 Stage B: ${RUN} eval + metrics"
  done < configs/study7/queue_stageB_ablation.txt
fi
mark "STAGEB_ABL_DONE"

# ---- Stage B permutation axis ----------------------------------------------
PERM_TRAIN="S2perm_llama_seed1042 S2perm_llama_seed1043 S2perm_llama_seed1044 \
S7perm_llama_seed1043 S7perm_llama_seed1044 \
S8perm_llama_seed1043 S8perm_llama_seed1044"
PERM_ALL="$PERM_TRAIN S7perm_llama_seed1042 S8perm_llama_seed1042"
PERM_SPLITS="test_main test_order_p1 test_order_p2 test_order_p3 test_order_p4 test_order_p5 test_t4_word"
for RUN in $PERM_TRAIN; do
  train_run "$RUN"
done
for RUN in $PERM_ALL; do
  for SPLIT in $PERM_SPLITS; do
    eval_run "$RUN" "$SPLIT"
  done
  ci "study7 Stage B perm: ${RUN} evals + metrics"
done
mark "STAGEB_PERM_DONE"

# ---- Stage C: Sec 5b descriptive c-perm runs --------------------------------
for sysn in S7cperm S8cperm; do
  for s in 1042 1043 1044; do
    RUN="${sysn}_llama_seed${s}"
    train_run "$RUN"
    eval_run "$RUN" test_main
    ci "study7 Stage C (Sec 5b): ${RUN} eval + metrics"
  done
done
mark "STAGEC_DONE"

# ---- M3: preregistered stats + descriptive + report -------------------------
$PY src/statistics/study7_h7.py >> "$EVAL_LOG" 2>&1 || fail "STATS_FAIL study7_h7"
$PY src/statistics/study7_descriptive.py >> "$EVAL_LOG" 2>&1 || fail "STATS_FAIL descriptive"
$PY src/statistics/render_study7_report.py >> "$EVAL_LOG" 2>&1 || fail "STATS_FAIL render"
ci "study7 M3: H7 stats + Sec 5b trade-off + report render"
mark "STATS_DONE"
mark "MAIN_DONE"
