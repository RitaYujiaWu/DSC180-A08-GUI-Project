#!/bin/bash
# Example: Run Mind2Web evaluation with UI-TARS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_agent.sh" \
    --eval_type mind2web \
    --domain test_domain_Info \
    --model ui-tars \
    --max_steps 15
