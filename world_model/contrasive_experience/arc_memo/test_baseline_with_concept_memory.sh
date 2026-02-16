#!/bin/bash

# ==============================================================================
# Test Baseline Performance with Concept Memory
# ==============================================================================
# This script runs baseline evaluation with concept memory enabled.
# Make sure you have built the concept memory first using run_arc_memo.sh
#
# Usage:
#   ./arc_memo/test_baseline_with_concept_memory.sh --eval_type <type> --domain <domain> --model <model>
# ==============================================================================

# Default values
EVAL_TYPE="mmina"
DOMAIN="shopping"
MODEL="qwen2.5-vl"
MAX_STEPS=15

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONCEPT_MEMORY_PATH="$SCRIPT_DIR/output/memory.json"

cd "$PROJECT_ROOT"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval_type|--evaluation_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --concept_memory_path)
            CONCEPT_MEMORY_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --eval_type TYPE          Evaluation type (mmina, mind2web, webvoyager)"
            echo "  --domain DOMAIN           Domain for evaluation"
            echo "  --model MODEL             Model to use"
            echo "  --max_steps N             Maximum steps per task (default: 15)"
            echo "  --concept_memory_path PATH Path to memory.json (default: arc_memo/output/memory.json)"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --eval_type mmina --domain shopping --model qwen2.5-vl"
            echo "  $0 --eval_type mmina --domain shopping --model qwen2.5-vl --concept_memory_path custom/path/memory.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if concept memory file exists
if [ ! -f "$CONCEPT_MEMORY_PATH" ]; then
    echo "============================================"
    echo "Error: Concept memory file not found!"
    echo "============================================"
    echo "Expected location: $CONCEPT_MEMORY_PATH"
    echo ""
    echo "Please run the arc_memo pipeline first:"
    echo "  ./arc_memo/run_arc_memo.sh"
    echo ""
    echo "Or specify a custom path:"
    echo "  $0 --concept_memory_path /path/to/memory.json ..."
    exit 1
fi

# Set result directory based on eval type, domain, and model
DATETIME=$(date +"%Y%m%d_%H%M%S")
FULL_RESULT_DIR="results/${EVAL_TYPE}/${DOMAIN}/${MODEL}_with_concept_memory/${DATETIME}"

# Create result directory
mkdir -p "$FULL_RESULT_DIR"

# Build command
CMD="python run.py \
    --evaluation_type $EVAL_TYPE \
    --domain $DOMAIN \
    --model $MODEL \
    --max_steps $MAX_STEPS \
    --result_dir $FULL_RESULT_DIR \
    --datetime $DATETIME \
    --use_concept_memory True \
    --concept_memory_path $CONCEPT_MEMORY_PATH"

# Print configuration
echo "============================================"
echo "Baseline Test with Concept Memory"
echo "============================================"
echo "Evaluation Type: $EVAL_TYPE"
echo "Domain:          $DOMAIN"
echo "Model:           $MODEL"
echo "Max Steps:       $MAX_STEPS"
echo "Result Dir:      $FULL_RESULT_DIR"
echo "Concept Memory:  $CONCEPT_MEMORY_PATH"
echo "============================================"
echo ""
echo "Running command:"
echo "$CMD"
echo ""
echo "============================================"

# Set LD_LIBRARY_PATH to include conda lib directory for Chromium
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    echo "Set LD_LIBRARY_PATH to include conda lib: $CONDA_PREFIX/lib"
fi

# Run the evaluation
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Evaluation completed successfully!"
    echo "Results saved to: $FULL_RESULT_DIR"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "Evaluation failed with exit code $?"
    echo "============================================"
    exit 1
fi

