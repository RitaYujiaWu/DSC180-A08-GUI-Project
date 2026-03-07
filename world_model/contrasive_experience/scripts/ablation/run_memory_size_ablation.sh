#!/bin/bash
# ============================================================================
# Memory Size Ablation Script
# Tests agent performance with different numbers of trajectories in graph memory
# ============================================================================

set -e

# Default configuration
CUDA_DEVICE="${CUDA_DEVICE:-7}"
EVAL_TYPE="${EVAL_TYPE:-webvoyager}"
DOMAIN="${DOMAIN:-Amazon}"
MODEL="${MODEL:-qwen2.5-vl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/sibo/GUI-Agent-Learn-From-Error/checkpoints/lora_qformer_test_V4-700_merged}"
BASE_RESULT_DIR="${BASE_RESULT_DIR:-results/ablation_memory_size}"

# Memory sizes to test (number of trajectories)
# Full memory (all_domains_8k) has 5946 trajectories
MEMORY_SIZES="${MEMORY_SIZES:-100 500 1000 2000 3000 4000 5000 5946}"

# Graph memory configuration
GRAPH_SIMILAR_NUM="${GRAPH_SIMILAR_NUM:-10}"
GRAPH_EXPAND_HOPS="${GRAPH_EXPAND_HOPS:-1}"

# Base graph index path (full memory - use 8k as base)
FULL_GRAPH_INDEX="CoMEM-Agent-Inference/graph_index/all_domains_8k"
MAX_SIZE=5946

# Memory data directories
MEMORY_DATA_DIRS=(
    "/home/sibo/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/data/trajectories"
    "/home/wwy/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/training_data/mind2web"
    "/home/wwy/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/training_data/guiact_converted"
)

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

create_subset_index() {
    local size=$1
    local output_dir="CoMEM-Agent-Inference/graph_index/ablation_size_${size}"
    
    if [[ -f "${output_dir}_trajectories.json" ]]; then
        echo "[INFO] Subset index for size ${size} already exists at ${output_dir}"
        return 0
    fi
    
    echo "[INFO] Creating subset index with ${size} trajectories..."
    
    python3 - <<EOF
import json
import random
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

size = ${size}
base_path = "CoMEM-Agent-Inference/graph_index/all_domains_8k"
output_path = "${output_dir}"

# Load full graph
print(f"Loading full graph from {base_path}...")

with open(f"{base_path}_trajectories.json", 'r') as f:
    all_trajectories = json.load(f)

with open(f"{base_path}_tags.json", 'r') as f:
    all_tags = json.load(f)

with open(f"{base_path}_graph.json", 'r') as f:
    full_graph = json.load(f)

embeddings_data = np.load(f"{base_path}_embeddings.npz", allow_pickle=True)
all_ids = list(embeddings_data['ids'])
all_embeddings = embeddings_data['embeddings']

print(f"Full graph has {len(all_trajectories)} trajectories")

# Sample trajectories
traj_ids = list(all_trajectories.keys())
if size >= len(traj_ids):
    selected_ids = set(traj_ids)
    print(f"Using all {len(traj_ids)} trajectories (requested {size})")
else:
    selected_ids = set(random.sample(traj_ids, size))
    print(f"Sampled {size} trajectories")

# Filter trajectories
subset_trajectories = {k: v for k, v in all_trajectories.items() if k in selected_ids}

# Filter tags (only keep tags that have at least one trajectory in subset)
subset_tags = {}
for tag, traj_list in all_tags.items():
    filtered = [t for t in traj_list if t in selected_ids]
    if filtered:
        subset_tags[tag] = filtered

# Filter graph nodes and edges
subset_nodes = [n for n in full_graph['nodes'] if n['id'] in selected_ids]
subset_links = [l for l in full_graph['links'] if l['source'] in selected_ids and l['target'] in selected_ids]

subset_graph = {
    'directed': full_graph.get('directed', False),
    'multigraph': full_graph.get('multigraph', False),
    'graph': full_graph.get('graph', {}),
    'nodes': subset_nodes,
    'links': subset_links
}

# Filter embeddings
id_to_idx = {str(id_): i for i, id_ in enumerate(all_ids)}
subset_indices = [id_to_idx[str(tid)] for tid in selected_ids if str(tid) in id_to_idx]
subset_emb_ids = [all_ids[i] for i in subset_indices]
subset_embeddings = all_embeddings[subset_indices]

# Save subset
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

with open(f"{output_path}_trajectories.json", 'w') as f:
    json.dump(subset_trajectories, f)

with open(f"{output_path}_tags.json", 'w') as f:
    json.dump(subset_tags, f)

with open(f"{output_path}_graph.json", 'w') as f:
    json.dump(subset_graph, f)

np.savez(f"{output_path}_embeddings.npz", ids=np.array(subset_emb_ids), embeddings=subset_embeddings)

print(f"Saved subset index to {output_path}")
print(f"  - Trajectories: {len(subset_trajectories)}")
print(f"  - Tags: {len(subset_tags)}")
print(f"  - Nodes: {len(subset_nodes)}")
print(f"  - Edges: {len(subset_links)}")
EOF
}

run_experiment() {
    local size=$1
    local graph_index_path=$2
    local result_dir="${BASE_RESULT_DIR}/${EVAL_TYPE}/${DOMAIN}/size_${size}"
    
    print_header "Running experiment with ${size} trajectories"
    echo "Graph index: ${graph_index_path}"
    echo "Result dir: ${result_dir}"
    
    # Build memory_data_dir arguments
    local memory_args=""
    for dir in "${MEMORY_DATA_DIRS[@]}"; do
        memory_args="${memory_args} --memory_data_dir ${dir}"
    done
    
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} bash CoMEM-Agent-Inference/scripts/runners/run_agent.sh \
        --eval_type ${EVAL_TYPE} \
        --domain ${DOMAIN} \
        --model ${MODEL} \
        --tool_model_name ${MODEL} \
        --use_continuous_memory \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --use_graph_memory \
        --graph_memory_index_path "${graph_index_path}" \
        --graph_similar_num ${GRAPH_SIMILAR_NUM} \
        --graph_expand_hops ${GRAPH_EXPAND_HOPS} \
        ${memory_args} \
        --result_dir "${result_dir}"
    
    echo "[DONE] Experiment with size ${size} completed"
}

# ============================================================================
# Main
# ============================================================================

print_header "Memory Size Ablation Study"

echo "Configuration:"
echo "  CUDA Device: ${CUDA_DEVICE}"
echo "  Eval Type: ${EVAL_TYPE}"
echo "  Domain: ${DOMAIN}"
echo "  Model: ${MODEL}"
echo "  Memory Sizes: ${MEMORY_SIZES}"
echo "  Max Size (full graph): ${MAX_SIZE}"
echo "  Graph Similar Num: ${GRAPH_SIMILAR_NUM}"
echo "  Graph Expand Hops: ${GRAPH_EXPAND_HOPS}"
echo ""

# Change to project root
cd /home/sibo/GUI-Agent-Learn-From-Error

# Create subset indices for each size
print_header "Step 1: Creating Subset Indices"

for size in ${MEMORY_SIZES}; do
    if [[ ${size} -eq ${MAX_SIZE} ]]; then
        echo "[INFO] Size ${MAX_SIZE} uses the full graph index (no subset needed)"
    else
        create_subset_index ${size}
    fi
done

# Run experiments
print_header "Step 2: Running Experiments"

for size in ${MEMORY_SIZES}; do
    if [[ ${size} -eq ${MAX_SIZE} ]]; then
        # Full graph - run_agent.sh runs from CoMEM-Agent-Inference/, so use relative path
        graph_path="graph_index/all_domains_8k"
    else
        # Subsets are in CoMEM-Agent-Inference/graph_index/, use relative path for run.py
        graph_path="graph_index/ablation_size_${size}"
    fi
    
    run_experiment ${size} ${graph_path}
done

# Summary
print_header "Ablation Study Complete"

echo "Results saved to: ${BASE_RESULT_DIR}/${EVAL_TYPE}/${DOMAIN}/"
echo ""
echo "Memory sizes tested:"
for size in ${MEMORY_SIZES}; do
    echo "  - ${size} trajectories"
done

