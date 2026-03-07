#!/bin/bash
# Example: Run evaluation with discrete memory enabled

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_agent.sh" \
    --eval_type mmina \
    --domain shopping \
    --model qwen2.5-vl \
    --max_steps 15 \
    --use_discrete_memory \
    --collect_training_data
