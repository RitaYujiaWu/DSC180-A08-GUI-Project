#!/bin/bash
# Example: Run evaluation with multimodal continuous memory enabled

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_agent.sh" \
    --eval_type webvoyager \
    --domain test \
    --model qwen2.5-vl \
    --max_steps 15 \
    --use_continuous_memory
    # --use_discrete_memory  # Add for hybrid memory
    # --collect_training_data
    # --model ui-tars \
    # --checkpoint_path WenyiWU0111/lora_qformer_uitars_test_V1-400_merged
