#!/bin/bash
# Scaling experiment: vary retrieval k
# Usage: bash CoMEM-Agent-Inference/scripts/runners/run_scaling_exp.sh
# Or in tmux: tmux new-session -d -s scaling_exp 'bash CoMEM-Agent-Inference/scripts/runners/run_scaling_exp.sh; exec bash'

cd /home/sibo/GUI-Agent-Learn-From-Error

for k in 20; do
  echo "=== Running k=$k ==="
  CUDA_VISIBLE_DEVICES=7 bash CoMEM-Agent-Inference/scripts/runners/run_agent.sh \
    --eval_type webvoyager \
    --domain Coursera \
    --model qwen2.5-vl \
    --tool_model_name qwen2.5-vl \
    --use_continuous_memory \
    --checkpoint_path "/home/sibo/GUI-Agent-Learn-From-Error/checkpoints/lora_qformer_test_V4-700_merged" \
    --use_graph_memory \
    --graph_memory_index_path "graph_index/all_domains" \
    --graph_similar_num $k \
    --graph_expand_hops 1 \
    --use_dynamic_memory_update \
    --max_memory_updates 2 \
    --result_dir "results/scaling_retrieval_k${k}"
done

echo "=== All experiments complete ==="

