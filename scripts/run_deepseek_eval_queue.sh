#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/outputs/queue_logs"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_eval() {
  local label=$1 model=$2 dataset=$3 portfolio=$4 out=$5 overlap=$6
  if [[ -f "$out/summary.json" ]]; then
    echo "[$(date -Is)] skip $label/$dataset: summary exists"
    return
  fi
  echo "[$(date -Is)] start $label/$dataset"
  local args=(python -m blindflip evaluate --backend vllm --model "$model"
    --portfolio "$portfolio" --data "$ROOT/data/$dataset.json" --budgets 2,3,5,10
    --batch-size 64 --max-model-len 32768 --gpu-memory-utilization 0.85
    --trust-remote-code --out "$out")
  if [[ "$overlap" == true ]]; then
    args+=(--allow-content-overlap)
  fi
  if "${args[@]}"; then
    echo "[$(date -Is)] done $label/$dataset"
  else
    echo "[$(date -Is)] failed $label/$dataset; continuing queue" >&2
  fi
}

wait_for_qwen4() {
  local config
  while true; do
    config=$(find /root/autodl-tmp -maxdepth 4 -type f -name config.json -path '*Qwen3.5-4B*' -print -quit)
    if [[ -n "$config" ]]; then
      dirname "$config"
      return
    fi
    echo "[$(date -Is)] waiting for Qwen3.5-4B download" >&2
    sleep 300
  done
}

# Finish targets already partially evaluated before moving to the newly queued models.
run_eval glm4_9b /root/autodl-tmp/glm-4-9b-chat arena_hard_test \
  "$ROOT/outputs/deepseek_v4_flash_arena_hard_portfolio_b10.json" \
  "$ROOT/outputs/deepseek_eval_glm4_9b_arena_hard_vllm" true

for dataset in mtbench_test llmbar_test arena_hard_test alpaca_eval_test; do
  case "$dataset" in
    mtbench_test) portfolio=deepseek_v4_flash_mtbench_portfolio_b10.json ;;
    llmbar_test) portfolio=deepseek_v4_flash_llmbar_portfolio_b10.json ;;
    arena_hard_test) portfolio=deepseek_v4_flash_arena_hard_portfolio_b10.json ;;
    alpaca_eval_test) portfolio=deepseek_v4_flash_alpaca_eval_portfolio_b10.json ;;
  esac
  run_eval gemma4b /root/autodl-tmp/gemma-4-E4B-it "$dataset" \
    "$ROOT/outputs/$portfolio" "$ROOT/outputs/deepseek_eval_gemma4b_${dataset%_test}_vllm" true
done

for dataset in mtbench_test llmbar_test arena_hard_test alpaca_eval_test; do
  case "$dataset" in
    mtbench_test) portfolio=deepseek_v4_flash_mtbench_portfolio_b10.json ;;
    llmbar_test) portfolio=deepseek_v4_flash_llmbar_portfolio_b10.json ;;
    arena_hard_test) portfolio=deepseek_v4_flash_arena_hard_portfolio_b10.json ;;
    alpaca_eval_test) portfolio=deepseek_v4_flash_alpaca_eval_portfolio_b10.json ;;
  esac
  run_eval phi4mini /root/autodl-tmp/Phi-4-mini-instruct "$dataset" \
    "$ROOT/outputs/$portfolio" "$ROOT/outputs/deepseek_eval_phi4mini_${dataset%_test}_vllm" true
done

qwen4_model=$(wait_for_qwen4)
for dataset in mtbench_test llmbar_test arena_hard_test alpaca_eval_test; do
  case "$dataset" in
    mtbench_test) portfolio=deepseek_v4_flash_mtbench_portfolio_b10.json ;;
    llmbar_test) portfolio=deepseek_v4_flash_llmbar_portfolio_b10.json ;;
    arena_hard_test) portfolio=deepseek_v4_flash_arena_hard_portfolio_b10.json ;;
    alpaca_eval_test) portfolio=deepseek_v4_flash_alpaca_eval_portfolio_b10.json ;;
  esac
  run_eval qwen35_4b "$qwen4_model" "$dataset" \
    "$ROOT/outputs/$portfolio" "$ROOT/outputs/deepseek_eval_qwen35_4b_${dataset%_test}_vllm" true
done

echo "[$(date -Is)] DeepSeek evaluation queue completed"
