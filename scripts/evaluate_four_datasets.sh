#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TARGET_MODEL:?Set TARGET_MODEL to a checkpoint path or model ID}"
PORTFOLIO="${PORTFOLIO:-examples/qwen2b_portfolio.json}"
OUT_ROOT="${OUT_ROOT:-outputs/frozen_target}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
for dataset in llmbar_test mtbench_test alpaca_eval_test arena_hard_test; do
  overlap=()
  if [[ "$dataset" == llmbar_test ]]; then
    overlap=(--allow-content-overlap)
  fi
  python -m blindflip evaluate --backend vllm --model "$TARGET_MODEL" \
    --portfolio "$PORTFOLIO" --data "data/$dataset.json" --budgets 2,3,5,10 \
    --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
    --out "$OUT_ROOT/$dataset" "${overlap[@]}"
done
