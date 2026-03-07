#!/bin/bash
# Example: Run MMInA Wikipedia domain evaluation with Qwen2.5-VL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_agent.sh" \
    --eval_type mmina \
    --domain wikipedia \
    --model qwen2.5-vl \
    --max_steps 15
