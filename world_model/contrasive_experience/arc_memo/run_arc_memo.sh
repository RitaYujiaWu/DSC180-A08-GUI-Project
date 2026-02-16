#!/bin/bash

# ==============================================================================
# ArcMemo Pipeline Execution Script
# ==============================================================================
# This script builds INITIAL concept memory from existing trajectories.
# 
# IMPORTANT: This is NOT the self-evolving loop from the ArcMemo paper!
# - This builds STATIC initial memory from existing trajectories
# - Concept memory does NOT automatically update during inference
# - For self-evolution, concept memory updates would need to be implemented
#
# Usage:
#   ./arc_memo/run_arc_memo.sh [options]
#
# Examples:
#   ./arc_memo/run_arc_memo.sh --skip_extract                    # Skip trajectory extraction
#   ./arc_memo/run_arc_memo.sh --skip_pseudocode                 # Skip pseudocode generation
#   ./arc_memo/run_arc_memo.sh --skip_all_steps                  # Only run abstraction (if data exists)
#   ./arc_memo/run_arc_memo.sh --full_pipeline                   # Run all steps (default)
# ==============================================================================

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARC_MEMO_DIR="$SCRIPT_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Default values
SKIP_EXTRACT=false
SKIP_PSEUDOCODE=false
SKIP_ABSTRACT=false
SKIP_COMPRESS=false
SKIP_ALL_STEPS=false
FULL_PIPELINE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --skip_pseudocode)
            SKIP_PSEUDOCODE=true
            shift
            ;;
        --skip_abstract)
            SKIP_ABSTRACT=true
            shift
            ;;
        --skip_compress)
            SKIP_COMPRESS=true
            shift
            ;;
        --skip_all_steps)
            SKIP_ALL_STEPS=true
            shift
            ;;
        --full_pipeline)
            FULL_PIPELINE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip_extract          Skip trajectory extraction step"
            echo "  --skip_pseudocode       Skip pseudocode generation step"
            echo "  --skip_abstract         Skip concept abstraction step"
            echo "  --skip_compress         Skip memory compression step"
            echo "  --skip_all_steps        Skip all steps (only check if memory.json exists)"
            echo "  --full_pipeline         Run full pipeline (default)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Pipeline Stages:"
            echo "  1. Extract trajectories (optional if you have trajectory data)"
            echo "  2. Generate pseudocode from trajectories"
            echo "  3. Abstract concepts from pseudocode"
            echo "  4. Compress memory to text (optional)"
            echo ""
            echo "Note: Make sure you have:"
            echo "  - vLLM server running at http://localhost:8000/v1"
            echo "  - Trajectory data (or use --skip_extract if you already have it)"
            echo "  - Config files in arc_memo/configs/"
            echo ""
            echo "Important:"
            echo "  - This builds INITIAL memory from existing trajectories (not self-evolving)"
            echo "  - Concept memory does NOT automatically update during inference"
            echo "  - This script does NOT require qformer (that's for --use_continuous_memory)"
            echo "  - This script does NOT run benchmarks or show performance scores"
            echo "  - To test performance, run: ./arc_memo/test_baseline_with_concept_memory.sh"
            echo ""
            echo "Self-Evolution Note:"
            echo "  True self-evolution (updating concepts during inference) is NOT implemented."
            echo "  To update concepts, you need to manually re-run this pipeline periodically."
            echo "  See: arc_memo/SELF_EVOLUTION_EXPLANATION.md for details"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "============================================"
echo "ArcMemo Pipeline Execution"
echo "============================================"
echo "Project Root: $PROJECT_ROOT"
echo "ArcMemo Dir: $ARC_MEMO_DIR"
echo "Skip Extract: $SKIP_EXTRACT"
echo "Skip Pseudocode: $SKIP_PSEUDOCODE"
echo "Skip Abstract: $SKIP_ABSTRACT"
echo "Skip Compress: $SKIP_COMPRESS"
echo "============================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3."
    exit 1
fi

# Check if required directories exist
if [ ! -d "$ARC_MEMO_DIR" ]; then
    echo "Error: arc_memo directory not found at $ARC_MEMO_DIR"
    exit 1
fi

# Default Stage 1 output dir (will be overridden by config if set)
OUTPUT_DIR="$ARC_MEMO_DIR/output"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# ==============================================================================
# Stage 0: Extract Trajectories (Optional)
# ==============================================================================
if [ "$SKIP_ALL_STEPS" = false ] && [ "$SKIP_EXTRACT" = false ]; then
    echo "============================================"
    echo "Stage 0: Extracting Trajectories"
    echo "============================================"
    
    EXTRACT_SCRIPT="$ARC_MEMO_DIR/concept_mem/extract_trajectories.py"
    
    if [ ! -f "$EXTRACT_SCRIPT" ]; then
        echo "Warning: extract_trajectories.py not found. Skipping extraction."
        echo "Make sure you have trajectory data in the expected location."
    else
        echo "Running trajectory extraction..."
        python3 "$EXTRACT_SCRIPT" || {
            echo "Warning: Trajectory extraction failed or no trajectories found."
            echo "You can skip this step with --skip_extract if you already have trajectory data."
        }
    fi
    echo ""
else
    echo "Skipping trajectory extraction (--skip_extract or --skip_all_steps)"
    echo ""
fi

# ==============================================================================
# Stage 1: Generate Pseudocode
# ==============================================================================
if [ "$SKIP_ALL_STEPS" = false ] && [ "$SKIP_PSEUDOCODE" = false ]; then
    echo "============================================"
    echo "Stage 1: Generating Pseudocode"
    echo "============================================"
    
    PSEUDOCODE_SCRIPT="$ARC_MEMO_DIR/concept_mem/pseudocode_simple.py"
    PSEUDOCODE_CONFIG="$ARC_MEMO_DIR/configs/config_simple.yaml"
    
    if [ ! -f "$PSEUDOCODE_SCRIPT" ]; then
        echo "Error: pseudocode_simple.py not found at $PSEUDOCODE_SCRIPT"
        exit 1
    fi
    
    if [ ! -f "$PSEUDOCODE_CONFIG" ]; then
        echo "Error: Config file not found at $PSEUDOCODE_CONFIG"
        exit 1
    fi
    
    # Resolve input/output paths from config
    TRAJECTORY_FILE=$(python3 <<PYEOF
import sys
import yaml
from pathlib import Path

cfg_path = "$PSEUDOCODE_CONFIG"
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
traj_path = cfg.get('trajectories', '')
if traj_path:
    # Convert to absolute path if relative
    traj_path = Path(traj_path)
    if not traj_path.is_absolute():
        from arc_memo.concept_mem.constants import REPO_ROOT
        traj_path = REPO_ROOT / traj_path
    print(str(traj_path))
PYEOF
    )

    CONFIG_OUTPUT_DIR=$(python3 <<PYEOF
import yaml

cfg_path = "$PSEUDOCODE_CONFIG"
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir',''))
PYEOF
    )

    # If config specifies an output_dir, use it
    if [ -n "$CONFIG_OUTPUT_DIR" ]; then
        OUTPUT_DIR="$CONFIG_OUTPUT_DIR"
    fi
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/logs"

    # Check if input trajectories exist (based on config)
    if [ -z "$TRAJECTORY_FILE" ] || [ ! -f "$TRAJECTORY_FILE" ]; then
        echo "Warning: Trajectory file not found (from config): ${TRAJECTORY_FILE:-<empty>}"
        echo "You may need to:"
        echo "  1. Run trajectory extraction first (remove --skip_extract)"
        echo "  2. Or provide trajectory data manually and update $PSEUDOCODE_CONFIG"
        echo ""
        echo "Skipping pseudocode generation..."
        SKIP_PSEUDOCODE=true
    else
        echo "Running pseudocode generation..."
        python3 "$PSEUDOCODE_SCRIPT" --config "$PSEUDOCODE_CONFIG" || {
            echo "Error: Pseudocode generation failed"
            exit 1
        }
        echo "Pseudocode saved to: $OUTPUT_DIR/initial_analysis.json"
    fi
    echo ""
else
    echo "Skipping pseudocode generation (--skip_pseudocode or --skip_all_steps)"
    echo ""
fi

# ==============================================================================
# Stage 2: Abstract Concepts
# ==============================================================================
if [ "$SKIP_ALL_STEPS" = false ] && [ "$SKIP_ABSTRACT" = false ]; then
    echo "============================================"
    echo "Stage 2: Abstracting Concepts"
    echo "============================================"
    
    ABSTRACT_SCRIPT="$ARC_MEMO_DIR/concept_mem/abstract_simple.py"
    ABSTRACT_CONFIG="$ARC_MEMO_DIR/configs/config_abstract.yaml"
    
    if [ ! -f "$ABSTRACT_SCRIPT" ]; then
        echo "Error: abstract_simple.py not found at $ABSTRACT_SCRIPT"
        exit 1
    fi
    
    if [ ! -f "$ABSTRACT_CONFIG" ]; then
        echo "Error: Config file not found at $ABSTRACT_CONFIG"
        exit 1
    fi
    
    # Resolve inputs/outputs from abstraction config
    AB_PSEUDOCODE_FILE=$(python3 <<PYEOF
import sys
import yaml
from pathlib import Path

cfg_path = "$ABSTRACT_CONFIG"
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
pseudocode_file = cfg.get('pseudocode_file', '')
if pseudocode_file:
    # Convert to absolute path if relative
    pseudocode_file = Path(pseudocode_file)
    if not pseudocode_file.is_absolute():
        from arc_memo.concept_mem.constants import REPO_ROOT
        pseudocode_file = REPO_ROOT / pseudocode_file
    print(str(pseudocode_file))
PYEOF
    )

    AB_OUTPUT_DIR=$(python3 <<PYEOF
import yaml

cfg_path = "$ABSTRACT_CONFIG"
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir',''))
PYEOF
    )

    # Fallbacks
    if [ -z "$AB_PSEUDOCODE_FILE" ]; then
        AB_PSEUDOCODE_FILE="$OUTPUT_DIR/initial_analysis.json"
    fi
    if [ -z "$AB_OUTPUT_DIR" ]; then
        AB_OUTPUT_DIR="$OUTPUT_DIR"
    fi

    # Check if pseudocode exists (from abstraction config)
    if [ ! -f "$AB_PSEUDOCODE_FILE" ]; then
        echo "Error: Pseudocode file not found: $AB_PSEUDOCODE_FILE"
        echo "Update $ABSTRACT_CONFIG or run pseudocode generation first."
        exit 1
    fi
    mkdir -p "$AB_OUTPUT_DIR"
    
    echo "Running concept abstraction..."
    python3 "$ABSTRACT_SCRIPT" --config "$ABSTRACT_CONFIG" || {
        echo "Error: Concept abstraction failed"
        exit 1
    }
    echo "Concepts saved to: $AB_OUTPUT_DIR/memory.json"
    echo ""
else
    echo "Skipping concept abstraction (--skip_abstract or --skip_all_steps)"
    echo ""
fi

# ==============================================================================
# Stage 3: Compress Memory (Optional)
# ==============================================================================
if [ "$SKIP_ALL_STEPS" = false ] && [ "$SKIP_COMPRESS" = false ]; then
    echo "============================================"
    echo "Stage 3: Compressing Memory (Optional)"
    echo "============================================"
    
    COMPRESS_SCRIPT="$ARC_MEMO_DIR/concept_mem/compress_memory.py"
    MEMORY_FILE="$AB_OUTPUT_DIR/memory.json"
    
    if [ ! -f "$COMPRESS_SCRIPT" ]; then
        echo "Warning: compress_memory.py not found. Skipping compression."
    elif [ ! -f "$MEMORY_FILE" ]; then
        echo "Warning: memory.json not found. Skipping compression."
    else
        echo "Compressing memory to text format..."
        python3 "$COMPRESS_SCRIPT" \
            --memory "$MEMORY_FILE" \
            --out "$OUTPUT_DIR/gui_init_mem.txt" || {
            echo "Warning: Memory compression failed (non-fatal)"
        }
        echo "Compressed memory saved to: $OUTPUT_DIR/gui_init_mem.txt"
    fi
    echo ""
else
    echo "Skipping memory compression (--skip_compress or --skip_all_steps)"
    echo ""
fi

# ==============================================================================
# Final Check
# ==============================================================================
echo "============================================"
echo "Pipeline Summary"
echo "============================================"

MEMORY_FILE="$AB_OUTPUT_DIR/memory.json"
if [ -f "$MEMORY_FILE" ]; then
    echo "✓ Concept memory file exists: $MEMORY_FILE"
    echo ""
    echo "Next Steps:"
    echo "  1. Test baseline performance WITH concept memory:"
    echo "     ./arc_memo/test_baseline_with_concept_memory.sh --eval_type mmina --domain shopping --model qwen2.5-vl"
    echo ""
    echo "  2. Or run manually:"
    echo "     python run.py \\"
    echo "       --use_concept_memory True \\"
    echo "       --concept_memory_path $MEMORY_FILE \\"
    echo "       --evaluation_type mmina \\"
    echo "       --domain shopping \\"
    echo "       --model qwen2.5-vl"
    echo ""
    echo "  3. Compare with baseline WITHOUT concept memory:"
    echo "     bash scripts/runners/run_agent.sh --eval_type mmina --domain shopping --model qwen2.5-vl"
    echo ""
else
    echo "✗ Concept memory file NOT found: $MEMORY_FILE"
    echo ""
    echo "Pipeline may not have completed successfully."
    echo "Check the logs in: $OUTPUT_DIR/logs/"
    exit 1
fi

# Show file sizes
if [ -f "$MEMORY_FILE" ]; then
    FILE_SIZE=$(du -h "$MEMORY_FILE" | cut -f1)
    echo "Memory file size: $FILE_SIZE"
fi

if [ -f "$OUTPUT_DIR/gui_init_mem.txt" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_DIR/gui_init_mem.txt" | cut -f1)
    echo "Compressed memory size: $FILE_SIZE"
fi

echo ""
echo "============================================"
echo "ArcMemo Pipeline Completed Successfully!"
echo "============================================"

