#!/bin/bash

# ==============================================================================
# GUI-Agent Runner Script (entry point)
# ==============================================================================
# This script runs evaluations for different benchmarks and models.
# It supports Reasoning Bank (text or multimodal) injection.
#
# Usage:
#   ./run_agent.sh --eval_type <type> --domain <domain> --model <model> [options]
#
# Examples:
#   ./run_agent.sh --eval_type webvoyager --domain Amazon --model qwen2.5-vl \
#     --grounding_model_name ui-ins-7b --grounding_mode force \
#     --use_reasoning_bank --reasoning_bank_multimodal \
#     --reasoning_bank_path memory/reasoning_bank_Amazon.jsonl \
#     --reasoning_index_base memory_index/reasoning_bank_mm --reasoning_top_k 2  
# ==============================================================================

# Default values
EVAL_TYPE="webvoyager"
DOMAIN="Amazon"
MODEL="qwen2.5-vl"
GROUNDING_MODEL="ui-ins-7b"  # Default grounding model
TOOL_MODEL=""                # Optional tool LLM override (defaults handled in code)
GROUNDING_MODE="auto"        # auto|prefer|force|off
MAX_STEPS=15
MAX_OBS_LENGTH=8192
RESULT_DIR="results"
TEST_START_IDX=""
TEST_END_IDX=""
USE_DISCRETE_MEMORY=false
USE_CONTINUOUS_MEMORY=false
USE_HISTORY=false
COLLECT_TRAINING_DATA=false
SAVE_EXAMPLES_MEMORY=false
USE_REASONING_BANK=false
REASONING_BANK_PATH=""
REASONING_TOP_K=""
REASONING_DOMAIN_FILTER=""
REASONING_INDEX_BASE=""
REASONING_BANK_MULTIMODAL=false
OPEN_ROUTER_API_KEY=""
CHECKPOINT_PATH=""
MEMORY_DATA_DIR=""
FAISS_INDEX_PATH=""
USE_TRAJECTORY_SUMMARY_MEMORY=false
TRAJECTORY_SUMMARY_CACHE_PATH=""
TRAJECTORY_SUMMARY_MAX_ACTIONS=""

# Graph Memory options
USE_GRAPH_MEMORY=false
GRAPH_MEMORY_INDEX_PATH=""
GRAPH_TAG_CACHE_PATH=""
GRAPH_EXPAND_HOPS=""
GRAPH_INITIAL_SEEDS=""
GRAPH_DIVERSITY_WEIGHT=""
GRAPH_SIMILAR_NUM=""
USE_DYNAMIC_MEMORY_UPDATE=false
MAX_MEMORY_UPDATES=""

# Self-evolving graph memory options
USE_SELF_EVOLVING_MEMORY=false
GRAPH_PERSIST_INTERVAL=""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------------------------------------------------------------
# Hugging Face cache configuration
# ------------------------------------------------------------------------------
# Use custom cache location since ~/.cache/huggingface is owned by root
if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME="$HOME/hf_hub"
fi
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

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
        --grounding_model_name|--grounding_model)
            GROUNDING_MODEL="$2"
            shift 2
            ;;
        --tool_model_name|--tool_model)
            TOOL_MODEL="$2"
            shift 2
            ;;
        --grounding_mode)
            GROUNDING_MODE="$2"
            shift 2
            ;;
        --use_continuous_memory)
            USE_CONTINUOUS_MEMORY=true
            shift
            ;;
        --use_history)
            # Accept either no value (sets true) or an explicit value
            if [[ -n "$2" && "$2" != --* ]]; then
                USE_HISTORY="$2"
                shift 2
            else
                USE_HISTORY=true
                shift
            fi
            ;;
        --collect_training_data)
            COLLECT_TRAINING_DATA=true
            shift
            ;;
        --save_examples_memory)
            SAVE_EXAMPLES_MEMORY=true
            shift
            ;;
        --use_reasoning_bank)
            USE_REASONING_BANK=true
            shift
            ;;
        --reasoning_bank_path)
            REASONING_BANK_PATH="$2"
            shift 2
            ;;
        --reasoning_top_k)
            REASONING_TOP_K="$2"
            shift 2
            ;;
        --reasoning_domain_filter)
            REASONING_DOMAIN_FILTER="$2"
            shift 2
            ;;
        --reasoning_index_base)
            REASONING_INDEX_BASE="$2"
            shift 2
            ;;
        --reasoning_bank_multimodal)
            REASONING_BANK_MULTIMODAL=true
            shift
            ;;
        --open_router_api_key)
            OPEN_ROUTER_API_KEY="$2"
            shift 2
            ;;
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --memory_data_dir)
            # Accumulate multiple directories (space-separated)
            if [ -z "$MEMORY_DATA_DIRS" ]; then
                MEMORY_DATA_DIRS="$2"
            else
                MEMORY_DATA_DIRS="$MEMORY_DATA_DIRS $2"
            fi
            shift 2
            ;;
        --faiss_index_path)
            FAISS_INDEX_PATH="$2"
            shift 2
            ;;
        --use_discrete_memory)
            USE_DISCRETE_MEMORY=true
            shift
            ;;
        --discrete_memory_cache_path)
            DISCRETE_MEMORY_CACHE_PATH="$2"
            shift 2
            ;;
        --discrete_memory_max_actions)
            DISCRETE_MEMORY_MAX_ACTIONS="$2"
            shift 2
            ;;
        --discrete_memory_use_checkpoint)
            DISCRETE_MEMORY_USE_CHECKPOINT=true
            shift
            ;;
        --use_graph_memory)
            USE_GRAPH_MEMORY=true
            shift
            ;;
        --graph_memory_index_path)
            GRAPH_MEMORY_INDEX_PATH="$2"
            shift 2
            ;;
        --graph_tag_cache_path)
            GRAPH_TAG_CACHE_PATH="$2"
            shift 2
            ;;
        --graph_expand_hops)
            GRAPH_EXPAND_HOPS="$2"
            shift 2
            ;;
        --graph_initial_seeds)
            GRAPH_INITIAL_SEEDS="$2"
            shift 2
            ;;
        --graph_diversity_weight)
            GRAPH_DIVERSITY_WEIGHT="$2"
            shift 2
            ;;
        --graph_similar_num)
            GRAPH_SIMILAR_NUM="$2"
            shift 2
            ;;
        --use_dynamic_memory_update)
            USE_DYNAMIC_MEMORY_UPDATE=true
            shift
            ;;
        --max_memory_updates)
            MAX_MEMORY_UPDATES="$2"
            shift 2
            ;;
        --use_self_evolving_memory)
            USE_SELF_EVOLVING_MEMORY=true
            shift
            ;;
        --graph_persist_interval)
            GRAPH_PERSIST_INTERVAL="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --result_dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        --test_start_idx)
            TEST_START_IDX="$2"
            shift 2
            ;;
        --test_end_idx)
            TEST_END_IDX="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --eval_type TYPE               Evaluation type (mmina, mind2web, mind2web_executable, webvoyager)"
            echo "  --domain DOMAIN                Domain for evaluation"
            echo "  --model MODEL                  Model to use for agent (default: qwen2.5-vl)"
            echo "  --grounding_model_name MODEL   Model to use for grounding (default: ui-ins-7b)"
            echo "  --tool_model_name MODEL        Model to use for tool LLM (default: qwen2.5-vl)"
            echo "  --grounding_mode MODE          Grounding usage: auto|prefer|force|off (default: auto)"
            echo "  --max_steps N                  Maximum steps per task (default: 15)"
            echo "  --result_dir DIR               Results directory (default: results)"
            echo "  --test_start_idx N             Start index for task list (default: 0)"
            echo "  --test_end_idx N               End index for task list (default: all)"
            echo "  --use_discrete_memory          Enable discrete memory (VLM-summarized trajectories)"
            echo "  --use_continuous_memory        Enable continuous memory (trained Q-Former)"
            echo "  --collect_training_data        Collect training data during evaluation"
            echo "  --save_examples_memory         Save trajectory examples for future distillation"
            echo "  --use_history                  Inject step-history reflection each turn"
            echo "  --use_reasoning_bank           Enable reasoning bank retrieval"
            echo "  --reasoning_bank_path PATH     Path to reasoning bank JSONL"
            echo "  --reasoning_top_k N            Number of items to inject (default: 2)"
            echo "  --reasoning_domain_filter BOOL Filter by current domain (default: True)"
            echo "  --reasoning_index_base PATH    Reasoning bank FAISS index base path"
            echo "  --reasoning_bank_multimodal    Use multimodal reasoning bank (text + images)"
            echo "  --open_router_api_key KEY      Override OpenRouter API key for this run"
            echo "  --checkpoint_path PATH         HF repo id or local path for continuous-memory checkpoint"
            echo "  --memory_data_dir PATH         Root directory of trajectory memories (can specify multiple times)"
            echo "  --faiss_index_path PATH        Optional FAISS index base path (without extension) for memory retrieval"
            echo "  --discrete_memory_cache_path PATH  JSON cache for discrete memory summaries"
            echo "  --discrete_memory_max_actions N    Max actions to show the summarizer per trajectory"
            echo "  --discrete_memory_use_checkpoint   Use fine-tuned checkpoint for discrete memory (instead of tool_llm)"
            echo "  --use_graph_memory             Enable graph memory (FAISS + graph expansion for diversity)"
            echo "  --graph_memory_index_path PATH Path to saved graph index"
            echo "  --graph_tag_cache_path PATH    Path to cache extracted tags"
            echo "  --graph_expand_hops N          Number of hops to expand in graph (default: 1)"
            echo "  --graph_initial_seeds N        Number of FAISS seeds in Phase 1 (default: k/2, set to 1 for expansion-heavy)"
            echo "  --graph_diversity_weight N     Weight for diversity vs similarity (default: 0.3)"
            echo "  --graph_similar_num N          Number of trajectories to retrieve (default: 5)"
            echo "  --use_dynamic_memory_update    Enable dynamic memory update (VLM checkpoint after each action)"
            echo "  --max_memory_updates N         Maximum number of memory updates per task (default: 3)"
            echo "  --use_self_evolving_memory     Enable self-evolving graph memory (add successful trajectories online)"
            echo "  --graph_persist_interval N     Persist graph to disk every N successful additions (default: 5)"
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "Supported Evaluation Types & Domains:"
            echo "  mmina:       shopping, wikipedia, normal, multi567, compare, multipro"
            echo "  mind2web:    test_website, test_domain_Info, test_domain_Service"
            echo "  webvoyager:  test, Allrecipes, Amazon, Apple, ArXiv, Booking, GitHub,"
            echo "               Google_Map, Google_Search, Google_Flights, ESPN, Huggingface,"
            echo "               BBC_News, Wolfram_Alpha"
            echo ""
            echo "Supported Models:"
            echo "  qwen3-vl, qwen3-vl-or, qwen2.5-vl, qwen2-vl, qwen2.5-vl-32b, ui-tars,"
            echo "  ui-ins-7b, ui-ins-32b, cogagent, websight, gemini, claude, gpt-4o"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate evaluation type
case $EVAL_TYPE in
    mmina|mind2web|mind2web_executable|webvoyager)
        ;;
    *)
        echo "Error: Invalid evaluation type '$EVAL_TYPE'"
        echo "Supported types: mmina, mind2web, mind2web_executable, webvoyager"
        exit 1
        ;;
esac

# Validate model
case $MODEL in
    qwen3-vl|qwen3-vl-or|qwen2.5-vl|qwen2-vl|qwen2.5-vl-32b|ui-tars|ui-ins-7b|ui-ins-32b|cogagent|gemini|claude|gpt-4o|websight)
        ;;
    *)
        echo "Warning: Model '$MODEL' not in predefined list. Make sure it's configured in llm_config.py"
        ;;
esac

# Set result directory based on eval type, domain, and model
DATETIME=$(date +"%Y%m%d_%H%M%S")
FULL_RESULT_DIR="${RESULT_DIR}/${EVAL_TYPE}/${DOMAIN}/${MODEL}/${DATETIME}"

# Create result directory
mkdir -p "$FULL_RESULT_DIR"

# Build command
CMD="python \"$PROJECT_ROOT/run.py\" \
    --evaluation_type \"$EVAL_TYPE\" \
    --domain \"$DOMAIN\" \
    --model \"$MODEL\" \
    --grounding_model_name \"$GROUNDING_MODEL\" \
    --grounding_mode \"$GROUNDING_MODE\" \
    --max_steps \"$MAX_STEPS\" \
    --result_dir \"$FULL_RESULT_DIR\" \
    --datetime \"$DATETIME\""

# Optional tool model override
if [ -n "$TOOL_MODEL" ]; then
    CMD="$CMD --tool_model_name \"$TOOL_MODEL\""
fi

# Add optional flags
if [ "$USE_CONTINUOUS_MEMORY" = true ]; then
    CMD="$CMD --use_continuous_memory True"
fi

if [ "$USE_HISTORY" = true ] || [ "$USE_HISTORY" = "True" ]; then
    CMD="$CMD --use_history True"
fi

if [ "$COLLECT_TRAINING_DATA" = true ]; then
    CMD="$CMD --collect_training_data"
fi

if [ "$SAVE_EXAMPLES_MEMORY" = true ]; then
    CMD="$CMD --save_examples_memory"
fi

# Reasoning bank flags
if [ "$USE_REASONING_BANK" = true ]; then
    CMD="$CMD --use_reasoning_bank True"
fi
if [ -n "$REASONING_BANK_PATH" ]; then
    CMD="$CMD --reasoning_bank_path \"$REASONING_BANK_PATH\""
fi
if [ -n "$REASONING_TOP_K" ]; then
    CMD="$CMD --reasoning_top_k \"$REASONING_TOP_K\""
fi
if [ -n "$REASONING_DOMAIN_FILTER" ]; then
    CMD="$CMD --reasoning_domain_filter \"$REASONING_DOMAIN_FILTER\""
fi
if [ -n "$REASONING_INDEX_BASE" ]; then
    CMD="$CMD --reasoning_index_base \"$REASONING_INDEX_BASE\""
fi
if [ "$REASONING_BANK_MULTIMODAL" = true ]; then
    CMD="$CMD --reasoning_bank_multimodal True"
    # If user didn't specify an index base, use multimodal default
    if [ -z "$REASONING_INDEX_BASE" ]; then
        CMD="$CMD --reasoning_index_base \"memory_index/reasoning_bank_mm\""
    fi
fi

if [ -n "$OPEN_ROUTER_API_KEY" ]; then
    CMD="$CMD --open_router_api_key \"$OPEN_ROUTER_API_KEY\""
fi

if [ -n "$CHECKPOINT_PATH" ]; then
    CMD="$CMD --checkpoint_path \"$CHECKPOINT_PATH\""
fi

if [ -n "$MEMORY_DATA_DIRS" ]; then
    # Pass each directory as a separate argument
    for dir in $MEMORY_DATA_DIRS; do
        CMD="$CMD --memory_data_dir \"$dir\""
    done
fi

if [ -n "$FAISS_INDEX_PATH" ]; then
    CMD="$CMD --faiss_index_path \"$FAISS_INDEX_PATH\""
fi

if [ "$USE_DISCRETE_MEMORY" = true ]; then
    CMD="$CMD --use_discrete_memory True"
fi

if [ -n "$DISCRETE_MEMORY_CACHE_PATH" ]; then
    CMD="$CMD --discrete_memory_cache_path \"$DISCRETE_MEMORY_CACHE_PATH\""
fi

if [ -n "$DISCRETE_MEMORY_MAX_ACTIONS" ]; then
    CMD="$CMD --discrete_memory_max_actions \"$DISCRETE_MEMORY_MAX_ACTIONS\""
fi

if [ "$DISCRETE_MEMORY_USE_CHECKPOINT" = true ]; then
    CMD="$CMD --discrete_memory_use_checkpoint True"
fi

# Graph memory flags
if [ "$USE_GRAPH_MEMORY" = true ]; then
    CMD="$CMD --use_graph_memory True"
fi
if [ -n "$GRAPH_MEMORY_INDEX_PATH" ]; then
    CMD="$CMD --graph_memory_index_path \"$GRAPH_MEMORY_INDEX_PATH\""
fi
if [ -n "$GRAPH_TAG_CACHE_PATH" ]; then
    CMD="$CMD --graph_tag_cache_path \"$GRAPH_TAG_CACHE_PATH\""
fi
if [ -n "$GRAPH_EXPAND_HOPS" ]; then
    CMD="$CMD --graph_expand_hops \"$GRAPH_EXPAND_HOPS\""
fi
if [ -n "$GRAPH_INITIAL_SEEDS" ]; then
    CMD="$CMD --graph_initial_seeds \"$GRAPH_INITIAL_SEEDS\""
fi
if [ -n "$GRAPH_DIVERSITY_WEIGHT" ]; then
    CMD="$CMD --graph_diversity_weight \"$GRAPH_DIVERSITY_WEIGHT\""
fi
if [ -n "$GRAPH_SIMILAR_NUM" ]; then
    CMD="$CMD --graph_similar_num \"$GRAPH_SIMILAR_NUM\""
fi
if [ "$USE_DYNAMIC_MEMORY_UPDATE" = true ]; then
    CMD="$CMD --use_dynamic_memory_update True"
fi
if [ -n "$MAX_MEMORY_UPDATES" ]; then
    CMD="$CMD --max_memory_updates \"$MAX_MEMORY_UPDATES\""
fi

# Self-evolving graph memory flags
if [ "$USE_SELF_EVOLVING_MEMORY" = true ]; then
    CMD="$CMD --use_self_evolving_memory True"
fi
if [ -n "$GRAPH_PERSIST_INTERVAL" ]; then
    CMD="$CMD --graph_persist_interval \"$GRAPH_PERSIST_INTERVAL\""
fi

# Test start/end index for limiting tasks
if [ -n "$TEST_START_IDX" ]; then
    CMD="$CMD --test_start_idx \"$TEST_START_IDX\""
fi
if [ -n "$TEST_END_IDX" ]; then
    CMD="$CMD --test_end_idx \"$TEST_END_IDX\""
fi

# Print configuration
echo "============================================"
echo "GUI-Agent Runner"
echo "============================================"
echo "Evaluation Type: $EVAL_TYPE"
echo "Domain:          $DOMAIN"
echo "Agent Model:     $MODEL"
echo "Grounding Model: $GROUNDING_MODEL"
echo "Tool Model:      ${TOOL_MODEL:-default}"
echo "Grounding Mode:  $GROUNDING_MODE"
echo "Max Steps:       $MAX_STEPS"
echo "Result Dir:      $FULL_RESULT_DIR"
echo "Use Discrete Memory:   $USE_DISCRETE_MEMORY"
echo "Use Continuous Memory: $USE_CONTINUOUS_MEMORY"
echo "Use History:     $USE_HISTORY"
echo "Use Reasoning Bank:    $USE_REASONING_BANK"
echo "Reasoning Bank MM:     $REASONING_BANK_MULTIMODAL"
echo "Use Graph Memory:      $USE_GRAPH_MEMORY"
echo "Use Dynamic Memory:    $USE_DYNAMIC_MEMORY_UPDATE"
echo "Use Self-Evolving:     $USE_SELF_EVOLVING_MEMORY"
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

echo "HF Cache:        $HF_HOME"
echo "============================================"

# Run the evaluation
eval "$CMD"
EXIT_CODE=$?

# Check exit status
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Run completed successfully!"
    echo "Results saved to: $FULL_RESULT_DIR"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "Run failed with exit code $EXIT_CODE"
    echo "============================================"
    exit $EXIT_CODE
fi


