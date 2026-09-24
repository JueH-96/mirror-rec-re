#!/bin/bash
# Usage: serve_model.sh <hf_repo> <logname> [extra vllm flags...]
pkill -f "vllm serve" 2>/dev/null; sleep 5
REPO="$1"; NAME="$2"; shift 2
CUDA_VISIBLE_DEVICES=1 nohup conda run -n mirror-rec vllm serve "$REPO" \
  --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 "$@" \
  > "${MIRROR_REC_ROOT:?export_MIRROR_REC_ROOT=package_root}/experiments/logs/vllm_${NAME}.log" 2>&1 &
echo "vllm launch requested for $REPO (log: vllm_${NAME}.log)"
