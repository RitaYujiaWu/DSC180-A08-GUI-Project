#!/bin/bash
#
# Mind2Web Full Pipeline Script
# 
# This script runs baseline tests on all 3 Mind2Web domains,
# then runs the post-processing script to create a curated subset.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/runners/run_mind2web_full_pipeline.sh
#
# Or with custom options:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/runners/run_mind2web_full_pipeline.sh \
#       --model qwen2.5-vl \
#       --result_dir results/mind2web_baseline \
#       --skip_curate   # Skip post-processing if you want to run it manually later
#

set -e  # Exit on error

# Default values
MODEL="qwen2.5-vl"
TOOL_MODEL="qwen2.5-vl"
RESULT_DIR="results/mind2web_baseline"
MAX_STEPS=15
SKIP_CURATE=false
VLM_URL="http://localhost:8000/v1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tool_model_name)
            TOOL_MODEL="$2"
            shift 2
            ;;
        --result_dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --skip_curate)
            SKIP_CURATE=true
            shift
            ;;
        --vlm_url)
            VLM_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Mind2Web Full Pipeline"
echo "============================================================"
echo "Model: $MODEL"
echo "Tool Model: $TOOL_MODEL"
echo "Result Dir: $RESULT_DIR"
echo "Max Steps: $MAX_STEPS"
echo "Skip Curate: $SKIP_CURATE"
echo "============================================================"
echo ""

# Track start time
START_TIME=$(date +%s)

# ============================================================
# Domain 1: test_domain_Info (first 100 tasks)
# ============================================================
echo ""
echo "============================================================"
echo "[1/3] Running test_domain_Info (first 100 tasks)"
echo "============================================================"
echo ""

bash "$SCRIPT_DIR/run_agent.sh" \
    --eval_type mind2web \
    --domain test_domain_Info \
    --model "$MODEL" \
    --tool_model_name "$TOOL_MODEL" \
    --test_start_idx 0 \
    --test_end_idx 100 \
    --max_steps "$MAX_STEPS" \
    --result_dir "$RESULT_DIR"

echo ""
echo "[1/3] test_domain_Info completed!"
echo ""

# ============================================================
# Domain 2: test_domain_Service (first 100 tasks)
# ============================================================
echo ""
echo "============================================================"
echo "[2/3] Running test_domain_Service (first 100 tasks)"
echo "============================================================"
echo ""

bash "$SCRIPT_DIR/run_agent.sh" \
    --eval_type mind2web \
    --domain test_domain_Service \
    --model "$MODEL" \
    --tool_model_name "$TOOL_MODEL" \
    --test_start_idx 0 \
    --test_end_idx 100 \
    --max_steps "$MAX_STEPS" \
    --result_dir "$RESULT_DIR"

echo ""
echo "[2/3] test_domain_Service completed!"
echo ""

# ============================================================
# Domain 3: test_website (all 142 tasks)
# ============================================================
echo ""
echo "============================================================"
echo "[3/3] Running test_website (all tasks)"
echo "============================================================"
echo ""

bash "$SCRIPT_DIR/run_agent.sh" \
    --eval_type mind2web \
    --domain test_website \
    --model "$MODEL" \
    --tool_model_name "$TOOL_MODEL" \
    --max_steps "$MAX_STEPS" \
    --result_dir "$RESULT_DIR"

echo ""
echo "[3/3] test_website completed!"
echo ""

# ============================================================
# Calculate elapsed time
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "All baseline runs completed!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================================"
echo ""

# ============================================================
# Post-processing: Create curated subset
# ============================================================
if [ "$SKIP_CURATE" = false ]; then
    echo ""
    echo "============================================================"
    echo "Running post-processing to create curated subset..."
    echo "============================================================"
    echo ""
    
    python scripts/curate_mind2web_subset.py \
        --process_all \
        --vlm_url "$VLM_URL"
    
    echo ""
    echo "============================================================"
    echo "Curated subset created!"
    echo "Output files:"
    echo "  - curated_subsets/test_domain_Info.json"
    echo "  - curated_subsets/test_domain_Service.json"
    echo "  - curated_subsets/test_website.json"
    echo "  - curated_subsets/mind2web_combined.json"
    echo "============================================================"
else
    echo ""
    echo "Skipping post-processing (--skip_curate flag set)"
    echo "To run manually later:"
    echo "  python scripts/curate_mind2web_subset.py --process_all"
    echo ""
fi

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"

