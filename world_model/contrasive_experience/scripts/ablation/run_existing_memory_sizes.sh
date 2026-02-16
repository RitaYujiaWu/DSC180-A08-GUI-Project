#!/bin/bash
# ============================================================================
# Run Memory Size Ablation with EXISTING indices
# Uses pre-built indices: 100, 500, 1000, 5k, 8k
# ============================================================================

set -e

CUDA_DEVICE="${CUDA_DEVICE:-2}"
EVAL_TYPE="${EVAL_TYPE:-webvoyager}"
DOMAIN="${DOMAIN:-Amazon}"
MODEL="${MODEL:-qwen2.5-vl}"
CHECKPOINT_PATH="/home/sibo/GUI-Agent-Learn-From-Error/checkpoints/lora_qformer_test_V4-700_merged"
BASE_RESULT_DIR="results/ablation_memory_size"

# Pre-built indices mapping: name -> path
declare -A INDICES=(
    ["100"]="graph_index/ablation_size_100"
    ["500"]="graph_index/ablation_size_500"
    ["1000"]="graph_index/ablation_size_1000"
    ["5000"]="graph_index/all_domains_5k"
    ["6000"]="graph_index/all_domains_8k"
)

# Which sizes to run (can override with env var)
SIZES="${SIZES:-100 500 1000 5000 6000}"

cd /home/sibo/GUI-Agent-Learn-From-Error

echo "============================================"
echo "Memory Size Ablation (Existing Indices)"
echo "============================================"
echo "CUDA Device: ${CUDA_DEVICE}"
echo "Eval Type: ${EVAL_TYPE}"
echo "Domain: ${DOMAIN}"
echo "Sizes to run: ${SIZES}"
echo "============================================"
echo ""

for size in ${SIZES}; do
    graph_path="${INDICES[$size]}"
    
    if [[ -z "$graph_path" ]]; then
        echo "[ERROR] No index found for size ${size}, skipping..."
        continue
    fi
    
    result_dir="${BASE_RESULT_DIR}/${EVAL_TYPE}/${DOMAIN}/size_${size}"
    
    echo ""
    echo "============================================"
    echo "Running size ${size}: ${graph_path}"
    echo "Result dir: ${result_dir}"
    echo "============================================"
    
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} bash CoMEM-Agent-Inference/scripts/runners/run_agent.sh \
        --eval_type ${EVAL_TYPE} \
        --domain ${DOMAIN} \
        --model ${MODEL} \
        --tool_model_name ${MODEL} \
        --use_continuous_memory \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --use_graph_memory \
        --graph_memory_index_path "${graph_path}" \
        --graph_similar_num 10 \
        --graph_expand_hops 1 \
        --memory_data_dir "/home/sibo/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/data/trajectories" \
        --memory_data_dir "/home/wwy/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/training_data/mind2web" \
        --memory_data_dir "/home/wwy/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/training_data/guiact_converted" \
        --result_dir "${result_dir}"
    
    echo "[DONE] Size ${size} completed"
done

echo ""
echo "============================================"
echo "All experiments completed!"
echo "Results saved to: ${BASE_RESULT_DIR}/${EVAL_TYPE}/${DOMAIN}/"
echo "============================================"

