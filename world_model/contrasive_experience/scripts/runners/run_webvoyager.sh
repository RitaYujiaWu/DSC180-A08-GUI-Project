#!/bin/bash
# Example: Run WebVoyager evaluation with GPT-4o

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_agent.sh" \
    --eval_type webvoyager \
    --domain test \
    --model qwen2.5-vl \
    --max_steps 15
