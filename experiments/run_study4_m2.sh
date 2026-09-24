#!/bin/bash
# Study 4 M2 evaluation chain (session-independent; run under nohup).
#
# Waits for the M1 retrain queue (M1_QUEUE_DONE marker), then:
#   1. trained llama evals: S2/S7/S8 x seeds 1042-1044 on test_unseen_item,
#      test_domain_phone, test_t4_word (T4 exploratory)   [GPU]
#   2. 4c evals: S2c/S7c/S8c_llama_seed1042 on test_main  [GPU]
#   3. trained qwen evals: S2/S7/S8_qwen_seed1042 on 4b/4d [GPU]
#   4. serve llama -> S0_llama sweep (3 splits) -> S1_llama projection
#   5. serve qwen  -> S0_qwen sweep (4b/4d)     -> S1_qwen projection
#   6. stop vLLM; S9 = S8 + projection (llama 3 seeds, 4b/4d)
#   7. metrics for every run x split; commit+push results
#
# Idempotent: evaluate.py / S0 sweeps skip completed request ids; metrics
# are cheap to redo. Log: experiments/logs/study4_m2_chain.log
set -u
cd ${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}
LOG=experiments/logs/study4_m2_chain.log
PY="conda run --no-capture-output -n mirror-rec python"
export CUDA_VISIBLE_DEVICES=1

mark() { echo "=== $(date -Iseconds) $* ===" >> "$LOG"; }
run() { echo "+ $*" >> "$LOG"; $PY "$@" >> "$LOG" 2>&1 || { mark "FAILED: $*"; exit 1; }; }

sets_for() {
  case "$1" in
    test_unseen_item)  echo data_processed/study4/candidate_sets_test_unseen_item.jsonl ;;
    test_domain_phone) echo data_processed/study4/candidate_sets_test_domain_phone.jsonl ;;
    *)                 echo data_processed/study3/candidate_sets_test_main.jsonl ;;
  esac
}
elig_for() {
  case "$1" in
    test_main) echo data_processed/study3/eligible_sets_test_main.json ;;
    *)         echo data_processed/study4/eligible_sets_$1.json ;;
  esac
}
metrics() {  # metrics <run_name> <split...>
  local m=$1; shift
  for split in "$@"; do
    run src/metrics/relational_metrics.py --config configs/pilot.yaml \
      --model "$m" --variant "$split" \
      --raw "results/study4/raw/$m/$split.jsonl" \
      --sets "$(sets_for "$split")" \
      --eligible "$(elig_for "$split")" \
      --out "results/study4/metrics_${m}_${split}.parquet"
  done
}
wait_vllm() {
  for i in $(seq 1 120); do
    curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && return 0
    sleep 5
  done
  mark "FAILED: vllm not ready after 600s"; exit 1
}

mark "CHAIN_START (waiting for M1)"
for i in $(seq 1 720); do
  grep -q M1_QUEUE_DONE experiments/logs/study4_m1_stages.log 2>/dev/null && break
  sleep 60
done
grep -q M1_QUEUE_DONE experiments/logs/study4_m1_stages.log || { mark "FAILED: M1 not done after 12h"; exit 1; }
mark "M1_GATE_PASSED"

# --- Stage 1: trained llama evals on new axes ---
for run_name in S2_llama_seed1042 S2_llama_seed1043 S2_llama_seed1044 \
                S7_llama_seed1042 S7_llama_seed1043 S7_llama_seed1044 \
                S8_llama_seed1042 S8_llama_seed1043 S8_llama_seed1044; do
  for split in test_unseen_item test_domain_phone test_t4_word; do
    mark "STAGE eval $run_name $split"
    run src/training/evaluate.py --config "configs/study3/${run_name}.yaml" \
      --ckpt "experiments/study3/${run_name}/final.pt" --split "$split" \
      --out "results/study4/raw/${run_name}/${split}.jsonl"
  done
done

# --- Stage 2: 4c evals on test_main ---
for run_name in S2c_llama_seed1042 S7c_llama_seed1042 S8c_llama_seed1042; do
  if [ ! -f "experiments/study4/${run_name}/final.pt" ]; then
    mark "SKIP eval $run_name (no final.pt - training failed, failure retained)"
    continue
  fi
  mark "STAGE eval $run_name test_main"
  run src/training/evaluate.py --config "configs/study4/${run_name}.yaml" \
    --ckpt "experiments/study4/${run_name}/final.pt" --split test_main \
    --out "results/study4/raw/${run_name}/test_main.jsonl"
done

# --- Stage 3: trained qwen evals (confirmation, seed 1042) ---
for run_name in S2_qwen_seed1042 S7_qwen_seed1042 S8_qwen_seed1042; do
  for split in test_unseen_item test_domain_phone; do
    mark "STAGE eval $run_name $split"
    run src/training/evaluate.py --config "configs/study3/${run_name}.yaml" \
      --ckpt "experiments/study3/${run_name}/final.pt" --split "$split" \
      --out "results/study4/raw/${run_name}/${split}.jsonl"
  done
done

# --- Stage 4: S0_llama sweep + S1_llama ---
mark "STAGE serve llama"
bash experiments/serve_model.sh NousResearch/Meta-Llama-3.1-8B-Instruct llama_s0_study4
wait_vllm
mark "STAGE sweep S0_llama"
run experiments/run_study4_s0.py --model llama-3.1-8b-instruct --family llama \
  --splits test_unseen_item,test_domain_phone,test_t4_word
mark "STAGE projection S1_llama"
run src/projection/study4_projection.py --src S0_llama --dst S1_llama \
  --splits test_unseen_item,test_domain_phone

# --- Stage 5: S0_qwen sweep + S1_qwen ---
mark "STAGE serve qwen"
bash experiments/serve_model.sh Qwen/Qwen2.5-14B-Instruct qwen_s0_study4
wait_vllm
mark "STAGE sweep S0_qwen"
run experiments/run_study4_s0.py --model qwen2.5-14b-instruct --family qwen \
  --splits test_unseen_item,test_domain_phone
mark "STAGE projection S1_qwen"
run src/projection/study4_projection.py --src S0_qwen --dst S1_qwen \
  --splits test_unseen_item,test_domain_phone

# --- Stage 6: stop vLLM; S9 = S8 + projection ---
pkill -f "vllm serve" 2>/dev/null
mark "STAGE vllm stopped"
for seed in 1042 1043 1044; do
  mark "STAGE projection S9_llama_seed${seed}"
  run src/projection/study4_projection.py --src "S8_llama_seed${seed}" \
    --dst "S9_llama_seed${seed}" --splits test_unseen_item,test_domain_phone
done

# --- Stage 7: metrics ---
for run_name in S2_llama_seed1042 S2_llama_seed1043 S2_llama_seed1044 \
                S7_llama_seed1042 S7_llama_seed1043 S7_llama_seed1044 \
                S8_llama_seed1042 S8_llama_seed1043 S8_llama_seed1044; do
  mark "STAGE metrics $run_name"
  metrics "$run_name" test_unseen_item test_domain_phone test_t4_word
done
for run_name in S2c_llama_seed1042 S7c_llama_seed1042 S8c_llama_seed1042; do
  [ -f "results/study4/raw/${run_name}/test_main.jsonl" ] || continue
  mark "STAGE metrics $run_name"
  metrics "$run_name" test_main
done
for run_name in S2_qwen_seed1042 S7_qwen_seed1042 S8_qwen_seed1042 \
                S1_qwen S9_llama_seed1042 S9_llama_seed1043 S9_llama_seed1044; do
  mark "STAGE metrics $run_name"
  metrics "$run_name" test_unseen_item test_domain_phone
done
mark "STAGE metrics S0_llama"
metrics S0_llama test_unseen_item test_domain_phone test_t4_word
mark "STAGE metrics S1_llama"
metrics S1_llama test_unseen_item test_domain_phone
mark "STAGE metrics S0_qwen"
metrics S0_qwen test_unseen_item test_domain_phone

# --- Stage 8: commit results ---
git add results/study4/ results/summary_*.json experiments/logs/study4_m2_chain.log 2>/dev/null
git commit -m "study4 M2: eval sweep complete (S2/S7/S8 llama 3 seeds + qwen seed1042 on unseen-item/phone-domain axes; 4c lv124 models on test_main; T4 exploratory; S0/S1 anchors both families; S9 projections; per-run relational metrics)" >/dev/null 2>&1 \
  && git push https://anonymous.invalid/anon-repo.git master >> "$LOG" 2>&1
mark "CHAIN_DONE"
