#!/bin/bash
# Study 6 M2 evaluation chain (session-independent; run under nohup).
#
# Waits for the M1 backfill queue (M1_QUEUE_DONE marker), then:
#   1. trained llama evals: S2/S7/S8 x seeds 1042-1044 on the 12 robustness
#      splits (order p1-5, distract d1/d2/d4, mix12/123/1234, neartied,
#      conflict)                                                    [GPU]
#   2. 6b backfill evals: S2c/S7c/S8c x seeds 1043/1044 on test_main [GPU]
#      (seed-1042 evals reused from study4)
#   3. trained qwen evals (confirmation, seed 1042): test_t4_word +
#      order p1-5 + neartied                                        [GPU]
#   4. serve llama -> S0_llama sweep (12 splits)
#   5. serve qwen  -> S0_qwen sweep (order p1-5 + neartied) + S0_qwen
#      test_t4_word via run_study4_s0.py (raw kept with study4 S0 sweeps)
#   6. stop vLLM; metrics for every run x split; commit+push
#
# Idempotent: evaluate.py / S0 sweeps skip completed request ids.
# Log: experiments/logs/study6_m2_chain.log
set -u
cd ${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}
LOG=experiments/logs/study6_m2_chain.log
PY="conda run --no-capture-output -n mirror-rec python"
export CUDA_VISIBLE_DEVICES=1

ROBUST_SPLITS="test_order_p1 test_order_p2 test_order_p3 test_order_p4 test_order_p5 \
test_distract_d1 test_distract_d2 test_distract_d4 \
test_mix12 test_mix123 test_mix1234 test_neartied test_conflict"
QWEN_SPLITS="test_order_p1 test_order_p2 test_order_p3 test_order_p4 test_order_p5 test_neartied"

mark() { echo "=== $(date -Iseconds) $* ===" >> "$LOG"; }
run() { echo "+ $*" >> "$LOG"; $PY "$@" >> "$LOG" 2>&1 || { mark "FAILED: $*"; exit 1; }; }

sets_for() {
  case "$1" in
    test_order_p*)              echo "data_processed/study6/candidate_sets_$1.jsonl" ;;
    test_neartied|test_conflict) echo "data_processed/study6/candidate_sets_$1.jsonl" ;;
    *)                          echo data_processed/study3/candidate_sets_test_main.jsonl ;;
  esac
}
elig_for() {
  case "$1" in
    test_main)    echo data_processed/study3/eligible_sets_test_main.json ;;
    test_t4_word) echo data_processed/study4/eligible_sets_test_t4_word.json ;;
    *)            echo data_processed/study6/eligible_sets_$1.json ;;
  esac
}
metrics() {  # metrics <run_name> <split...>
  local m=$1; shift
  for split in "$@"; do
    run src/metrics/relational_metrics.py --config configs/pilot.yaml \
      --model "$m" --variant "$split" \
      --raw "results/study6/raw/$m/$split.jsonl" \
      --sets "$(sets_for "$split")" \
      --eligible "$(elig_for "$split")" \
      --out "results/study6/metrics_${m}_${split}.parquet"
  done
}
wait_vllm() {
  for i in $(seq 1 120); do
    curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && return 0
    sleep 5
  done
  mark "FAILED: vllm not ready after 600s"; exit 1
}

# Hard gate (user directive 2026-08-07): do NOT trust queue-written markers
# (a no-GPU relaunch once wrote a false M1_QUEUE_DONE). Verify per run that
# the final checkpoint exists AND the train log reached max_steps. This
# verification style applies to every stage gate in later studies too.
M1_RUNS="S2c_llama_seed1043 S7c_llama_seed1043 S8c_llama_seed1043 \
S2c_llama_seed1044 S7c_llama_seed1044 S8c_llama_seed1044"
m1_complete() {
  for r in $M1_RUNS; do
    [ -f "experiments/study6/${r}/final.pt" ] || return 1
    grep -q '"step": 1500' "experiments/study6/${r}/train_log.jsonl" 2>/dev/null || return 1
  done
  return 0
}
mark "CHAIN_START (waiting for M1: hard per-run verification, markers not trusted)"
for i in $(seq 1 960); do
  m1_complete && break
  sleep 60
done
m1_complete || { mark "FAILED: M1 hard verification not satisfied after 16h"; exit 1; }
mark "M1_GATE_PASSED (hard verification: 6/6 final.pt + train_log step-1500)"

# --- Stage 1: trained llama evals on robustness splits ---
for run_name in S2_llama_seed1042 S2_llama_seed1043 S2_llama_seed1044 \
                S7_llama_seed1042 S7_llama_seed1043 S7_llama_seed1044 \
                S8_llama_seed1042 S8_llama_seed1043 S8_llama_seed1044; do
  for split in $ROBUST_SPLITS; do
    mark "STAGE eval $run_name $split"
    run src/training/evaluate.py --config "configs/study3/${run_name}.yaml" \
      --ckpt "experiments/study3/${run_name}/final.pt" --split "$split" \
      --out "results/study6/raw/${run_name}/${split}.jsonl"
  done
done

# --- Stage 2: 6b backfill evals on test_main ---
for run_name in S2c_llama_seed1043 S2c_llama_seed1044 \
                S7c_llama_seed1043 S7c_llama_seed1044 \
                S8c_llama_seed1043 S8c_llama_seed1044; do
  if [ ! -f "experiments/study6/${run_name}/final.pt" ]; then
    mark "SKIP eval $run_name (no final.pt - training failed, failure retained)"
    continue
  fi
  mark "STAGE eval $run_name test_main"
  run src/training/evaluate.py --config "configs/study6/${run_name}.yaml" \
    --ckpt "experiments/study6/${run_name}/final.pt" --split test_main \
    --out "results/study6/raw/${run_name}/test_main.jsonl"
done

# --- Stage 3: trained qwen evals (confirmation, seed 1042) ---
for run_name in S2_qwen_seed1042 S7_qwen_seed1042 S8_qwen_seed1042; do
  for split in test_t4_word $QWEN_SPLITS; do
    mark "STAGE eval $run_name $split"
    run src/training/evaluate.py --config "configs/study3/${run_name}.yaml" \
      --ckpt "experiments/study3/${run_name}/final.pt" --split "$split" \
      --out "results/study6/raw/${run_name}/${split}.jsonl"
  done
done

# --- Stage 4: S0_llama sweep ---
mark "STAGE serve llama"
bash experiments/serve_model.sh NousResearch/Meta-Llama-3.1-8B-Instruct llama_s0_study6
wait_vllm
mark "STAGE sweep S0_llama"
run experiments/run_study6_s0.py --model llama-3.1-8b-instruct --family llama

# --- Stage 5: S0_qwen sweeps ---
mark "STAGE serve qwen"
bash experiments/serve_model.sh Qwen/Qwen2.5-14B-Instruct qwen_s0_study6
wait_vllm
mark "STAGE sweep S0_qwen robustness"
run experiments/run_study6_s0.py --model qwen2.5-14b-instruct --family qwen \
  --splits "$(echo $QWEN_SPLITS | tr ' ' ',')"
mark "STAGE sweep S0_qwen t4 (raw kept under study4 S0 location)"
run experiments/run_study4_s0.py --model qwen2.5-14b-instruct --family qwen \
  --splits test_t4_word

# --- Stage 6: stop vLLM; metrics ---
pkill -f "vllm serve" 2>/dev/null
mark "STAGE vllm stopped"
for run_name in S2_llama_seed1042 S2_llama_seed1043 S2_llama_seed1044 \
                S7_llama_seed1042 S7_llama_seed1043 S7_llama_seed1044 \
                S8_llama_seed1042 S8_llama_seed1044 S8_llama_seed1043; do
  mark "STAGE metrics $run_name"
  metrics "$run_name" $ROBUST_SPLITS
done
for run_name in S2c_llama_seed1043 S2c_llama_seed1044 \
                S7c_llama_seed1043 S7c_llama_seed1044 \
                S8c_llama_seed1043 S8c_llama_seed1044; do
  [ -f "results/study6/raw/${run_name}/test_main.jsonl" ] || continue
  mark "STAGE metrics $run_name"
  metrics "$run_name" test_main
done
for run_name in S2_qwen_seed1042 S7_qwen_seed1042 S8_qwen_seed1042; do
  mark "STAGE metrics $run_name"
  metrics "$run_name" test_t4_word $QWEN_SPLITS
done
mark "STAGE metrics S0_llama"
metrics S0_llama $ROBUST_SPLITS
mark "STAGE metrics S0_qwen"
metrics S0_qwen $QWEN_SPLITS
mark "STAGE metrics S0_qwen t4 (raw from study4 location)"
run src/metrics/relational_metrics.py --config configs/pilot.yaml \
  --model S0_qwen --variant test_t4_word \
  --raw results/study4/raw/S0_qwen/test_t4_word.jsonl \
  --sets data_processed/study3/candidate_sets_test_main.jsonl \
  --eligible data_processed/study4/eligible_sets_test_t4_word.json \
  --out results/study6/metrics_S0_qwen_test_t4_word.parquet

# --- Stage 7: commit results ---
git add results/study6/ results/study4/raw/S0_qwen/ results/summary_*.json \
  experiments/logs/study6_m2_chain.log 2>/dev/null
git commit -m "study6 M2: eval sweep complete (llama 3 seeds x 12 robustness splits; 4c backfill seeds 1043/1044 on test_main; qwen seed1042 T4/order/near-tied confirmation; S0 anchors both families; per-run relational metrics)" >/dev/null 2>&1 \
  && git push https://anonymous.invalid/anon-repo.git master >> "$LOG" 2>&1
mark "CHAIN_DONE"
