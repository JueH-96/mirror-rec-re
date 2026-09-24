#!/bin/bash
# Self-contained chain: wait for Llama sweep completion, then serve GLM and run
# its 4-variant sweep. Survives orchestrator-session interruptions.
D=${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}
CLOG=$D/experiments/logs/chain.log
echo "=== $(date -Is) chain armed: waiting for llama ALL_VARIANTS_DONE ===" >> "$CLOG"
until grep -q "ALL_VARIANTS_DONE" "$D/experiments/logs/run_llama-3.1-8b-instruct.log" 2>/dev/null; do
  sleep 30
done
echo "=== $(date -Is) llama done; switching server to GLM ===" >> "$CLOG"
"$D/experiments/serve_model.sh" THUDM/glm-4-9b-chat glm --trust-remote-code >> "$CLOG" 2>&1
for i in $(seq 1 90); do
  if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "=== $(date -Is) GLM server READY ===" >> "$CLOG"; break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "=== $(date -Is) GLM server DIED during load (see vllm_glm.log) ===" >> "$CLOG"; exit 1
  fi
  sleep 10
done
if ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "=== $(date -Is) GLM server TIMEOUT after 15min ===" >> "$CLOG"; exit 1
fi
"$D/experiments/run_all_variants.sh" glm-4-9b-chat
echo "=== $(date -Is) CHAIN_DONE (glm sweep finished) ===" >> "$CLOG"
