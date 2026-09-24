#!/bin/bash
# Standalone S0/S1/S9 evaluation chain (session-independent; run under nohup).
#
# Stages (idempotent: sweeps skip cached/done requests, metrics are cheap to redo):
#   1. metrics for S0_llama (raw already swept)
#   2. S1_llama = S0_llama + isotonic projection, + metrics
#   3. serve Qwen -> S0_qwen sweep -> metrics -> S1_qwen projection -> metrics
#   4. stop vLLM
#   5. S9_<family>_seed<k> = S8_<family>_seed<k> + projection (all seeds), + metrics
#
# Log: experiments/logs/s0s1s9_chain.log (stage markers: === <ts> STAGE ... ===)
set -u
cd ${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}
LOG=experiments/logs/s0s1s9_chain.log
PY="conda run -n mirror-rec python"

mark() { echo "=== $(date -Iseconds) $* ===" >> "$LOG"; }
run() { echo "+ $*" >> "$LOG"; "$@" >> "$LOG" 2>&1 || { mark "FAILED: $*"; exit 1; }; }

sets_for() {
  case "$1" in
    val) echo data_processed/study3/candidate_sets_val.jsonl ;;
    test_unseen_attr) echo data_processed/study3/candidate_sets_test_unseen_attr.jsonl ;;
    *) echo data_processed/study3/candidate_sets_test_main.jsonl ;;
  esac
}

metrics4() {  # metrics4 <run_name>
  local m=$1
  for split in val test_main test_unseen_word test_unseen_attr; do
    run $PY src/metrics/relational_metrics.py --config configs/pilot.yaml \
      --model "$m" --variant "$split" \
      --raw "results/study3/raw/$m/$split.jsonl" \
      --sets "$(sets_for "$split")" \
      --eligible "data_processed/study3/eligible_sets_$split.json" \
      --out "results/study3/metrics_${m}_${split}.parquet"
  done
}

wait_vllm() {
  for i in $(seq 1 120); do
    curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && return 0
    sleep 5
  done
  mark "FAILED: vllm not ready after 600s"; exit 1
}

mark "CHAIN_START"

# --- Stage 1: S0_llama metrics (raw sweep already complete) ---
mark "STAGE metrics S0_llama"
metrics4 S0_llama

# --- Stage 2: S1_llama = S0_llama + projection ---
mark "STAGE projection S1_llama"
run $PY src/projection/study3_projection.py --src S0_llama --dst S1_llama
mark "STAGE metrics S1_llama"
metrics4 S1_llama

# --- Stage 3: Qwen S0/S1 ---
mark "STAGE serve qwen"
bash experiments/serve_model.sh Qwen/Qwen2.5-14B-Instruct qwen_s0
wait_vllm
mark "STAGE sweep S0_qwen"
run $PY experiments/run_study3_s0.py --model qwen2.5-14b-instruct --family qwen
mark "STAGE metrics S0_qwen"
metrics4 S0_qwen
mark "STAGE projection S1_qwen"
run $PY src/projection/study3_projection.py --src S0_qwen --dst S1_qwen
mark "STAGE metrics S1_qwen"
metrics4 S1_qwen

# --- Stage 4: stop vLLM ---
pkill -f "vllm serve" 2>/dev/null
mark "STAGE vllm stopped"

# --- Stage 5: S9 = S8 + projection, all seeds ---
for src in S8_llama_seed1042 S8_llama_seed1043 S8_llama_seed1044 S8_qwen_seed1042; do
  dst="S9_${src#S8_}"
  mark "STAGE projection $dst"
  run $PY src/projection/study3_projection.py --src "$src" --dst "$dst"
  mark "STAGE metrics $dst"
  metrics4 "$dst"
done

mark "CHAIN_DONE"
