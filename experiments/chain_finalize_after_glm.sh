#!/bin/bash
# Final leg: wait for GLM sweep completion, then build all machine-generatable
# pilot deliverables and back them up. Survives orchestrator interruptions.
D=${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}
CLOG=$D/experiments/logs/chain.log
until grep -q "ALL_VARIANTS_DONE" "$D/experiments/logs/run_glm-4-9b-chat.log" 2>/dev/null; do
  sleep 30
done
echo "=== $(date -Is) FINALIZE: building aggregate outputs ===" >> "$CLOG"
cd "$D"
if conda run -n mirror-rec python src/statistics/build_final_outputs.py >> "$CLOG" 2>&1 \
   && conda run -n mirror-rec python src/validation/validate_pipeline.py >> "$CLOG" 2>&1; then
  echo "=== $(date -Is) FINALIZE_OK (metrics_by_instance.parquet, aggregate_results.csv, statistical_tests.json, gonogo_precheck.json) ===" >> "$CLOG"
else
  echo "=== $(date -Is) FINALIZE_FAILED (see above) ===" >> "$CLOG"
fi
# Free GPU 1: pilot inference is over either way.
pkill -f "vllm serve" 2>/dev/null
git add -A >> "$CLOG" 2>&1
git commit -m "pilot: GLM complete + final aggregates (metrics_by_instance, stats, gonogo_precheck); Study 1/2 runs finished" >> "$CLOG" 2>&1
git push "https://anonymous.invalid/anon-repo.git" master:master >> "$CLOG" 2>&1 \
  && echo "=== $(date -Is) PUSHED ===" >> "$CLOG" \
  || echo "=== $(date -Is) PUSH_FAILED (retry at next stage) ===" >> "$CLOG"
echo "=== $(date -Is) PILOT_RUNS_ALL_DONE ===" >> "$CLOG"
