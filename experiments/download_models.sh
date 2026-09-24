#!/bin/bash
# Sequential model weight download with resume support.
LOG=${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}/experiments/logs/download_models.log
for m in "Qwen/Qwen2.5-14B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "THUDM/glm-4-9b-chat"; do
  echo "=== $(date -Is) downloading $m ===" >> "$LOG"
  if hf download "$m" >> "$LOG" 2>&1; then
    echo "=== $m OK ===" >> "$LOG"
  else
    echo "=== $m FAILED (continuing) ===" >> "$LOG"
  fi
done
echo "ALL_DOWNLOADS_ATTEMPTED $(date -Is)" >> "$LOG"
