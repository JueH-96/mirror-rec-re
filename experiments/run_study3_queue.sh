#!/bin/bash
# M2 queue: sequential train + eval for all Study 3 runs on GPU 1.
# Resume-safe: completed runs (final.pt + all eval files) are skipped;
# interrupted training resumes from last.pt automatically. Crashed runs keep
# their logs/checkpoints untouched (failure retention) and the queue moves on.
D=${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}
QLOG=$D/experiments/logs/study3_queue.log
cd "$D"
while read -r cfg_name; do
  cfg="configs/study3/$cfg_name"
  run=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['run_name'])")
  rdir="experiments/study3/$run"
  if [ ! -f "$rdir/final.pt" ]; then
    echo "=== $(date -Is) TRAIN $run ===" >> "$QLOG"
    if ! CUDA_VISIBLE_DEVICES=1 conda run -n mirror-rec python src/training/train.py \
        --config "$cfg" >> "experiments/logs/train_$run.log" 2>&1; then
      echo "=== $(date -Is) TRAIN_FAILED $run (logs kept, moving on) ===" >> "$QLOG"
      continue
    fi
  fi
  ok=1
  for split in val test_main test_unseen_word test_unseen_attr; do
    outf="results/study3/raw/$run/$split.jsonl"
    nexp=$(python3 -c "
import json
el=set(json.load(open('data_processed/study3/eligible_sets_$split.json')))
fams={'test_unseen_word':1}.get('$split',2)
print(len(el)*fams*4)")
    ndone=0; [ -f "$outf" ] && ndone=$(wc -l < "$outf")
    if [ "$ndone" -lt "$nexp" ]; then
      echo "=== $(date -Is) EVAL $run $split ===" >> "$QLOG"
      if ! CUDA_VISIBLE_DEVICES=1 conda run -n mirror-rec python src/training/evaluate.py \
          --config "$cfg" --ckpt "$rdir/final.pt" --split "$split" \
          --out "$outf" >> "experiments/logs/train_$run.log" 2>&1; then
        echo "=== $(date -Is) EVAL_FAILED $run $split ===" >> "$QLOG"; ok=0; continue
      fi
    fi
    sets="data_processed/study3/candidate_sets_test_main.jsonl"
    case $split in
      val) sets="data_processed/study3/candidate_sets_val.jsonl";;
      test_unseen_attr) sets="data_processed/study3/candidate_sets_test_unseen_attr.jsonl";;
    esac
    conda run -n mirror-rec python src/metrics/relational_metrics.py \
      --model "$run" --variant "$split" --raw "$outf" --sets "$sets" \
      --eligible "data_processed/study3/eligible_sets_$split.json" \
      --out "results/study3/metrics_${run}_${split}.parquet" \
      >> "experiments/logs/train_$run.log" 2>&1 || ok=0
  done
  [ "$ok" = "1" ] && echo "=== $(date -Is) RUN_COMPLETE $run ===" >> "$QLOG"
  git add -A >> /dev/null 2>&1
  git commit -q -m "study3 M2: $run complete (train+eval+metrics)" >> /dev/null 2>&1
  git push -q "https://anonymous.invalid/anon-repo.git" master:master >> /dev/null 2>&1 || true
done < configs/study3/queue_order.txt
echo "=== $(date -Is) M2_QUEUE_DONE ===" >> "$QLOG"
