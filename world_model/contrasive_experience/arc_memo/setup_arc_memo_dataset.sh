#!/bin/bash

# ==============================================================================
# ArcMemo Dataset Setup Script
# ==============================================================================
# Downloads and prepares the CoMEM trajectory dataset from Hugging Face
# for use with the arc_memo pipeline.
#
# Usage:
#   ./arc_memo/setup_arc_memo_dataset.sh [options]
#
# Examples:
#   ./arc_memo/setup_arc_memo_dataset.sh                          # Full setup
#   ./arc_memo/setup_arc_memo_dataset.sh --skip_download          # Skip download (if already have dataset)
#   ./arc_memo/setup_arc_memo_dataset.sh --dataset_path /custom/path  # Use custom dataset location
# ==============================================================================

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARC_MEMO_DIR="$SCRIPT_DIR"

# Default values
SKIP_DOWNLOAD=false
DATASET_PATH=""
HF_DATASET_NAME="WenyiWU0111/CoMEM-agent-memory-trajectories"
DOWNLOAD_DIR="$PROJECT_ROOT/data/downloaded_datasets/CoMEM-agent-memory-trajectories"
SKIP_UNZIP=false
DOMAIN_FILTER=""
MEMORY_PART=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --download_dir)
            DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --skip_unzip)
            SKIP_UNZIP=true
            shift
            ;;
        --domain)
            DOMAIN_FILTER="$2"
            shift 2
            ;;
        --part)
            MEMORY_PART="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip_download           Skip downloading dataset (use existing)"
            echo "  --dataset_path PATH       Use existing dataset at this path"
            echo "  --download_dir DIR        Directory to download dataset (default: ./data/downloaded_datasets/)"
            echo "  --skip_unzip              Skip unzipping (if already unzipped)"
            echo "  --domain DOMAIN           Only process specific domain (e.g., shopping, academic)"
            echo "  --part PART               Only process specific memory part (e.g., part2, part1)"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "This script will:"
            echo "  1. Download dataset from Hugging Face (if not skipped)"
            echo "  2. Unzip trajectory ZIP files"
            echo "  3. Extract trajectories into arc_memo/data/webvoyager_memory/"
            echo "  4. Update config files with correct paths"
            echo "  5. Verify setup"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

echo "============================================"
echo "ArcMemo Dataset Setup"
echo "============================================"
echo "Project Root: $PROJECT_ROOT"
echo "ArcMemo Dir: $ARC_MEMO_DIR"
echo "Download Dir: $DOWNLOAD_DIR"
echo "Skip Download: $SKIP_DOWNLOAD"
if [ -n "$MEMORY_PART" ]; then
    echo "Memory Part: $MEMORY_PART"
fi
if [ -n "$DOMAIN_FILTER" ]; then
    echo "Domain Filter: $DOMAIN_FILTER"
fi
echo "============================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3."
    exit 1
fi

# Check if huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub --quiet
fi

# ==============================================================================
# Step 1: Download Dataset
# ==============================================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "============================================"
    echo "Step 1: Downloading Dataset from Hugging Face"
    echo "============================================"
    
    if [ -n "$DATASET_PATH" ] && [ -d "$DATASET_PATH" ]; then
        echo "Using existing dataset at: $DATASET_PATH"
        DOWNLOAD_DIR="$DATASET_PATH"
    else
        echo "Downloading dataset: $HF_DATASET_NAME"
        echo "This may take a while (dataset is ~100K-1M entries)..."
        
        mkdir -p "$DOWNLOAD_DIR"
        
        # Download using huggingface_hub
        python3 <<EOF
from huggingface_hub import snapshot_download
import os

download_dir = "$DOWNLOAD_DIR"
dataset_name = "$HF_DATASET_NAME"

print(f"Downloading {dataset_name} to {download_dir}...")
snapshot_download(
    repo_id=dataset_name,
    repo_type="dataset",
    local_dir=download_dir,
    local_dir_use_symlinks=False
)
print("Download complete!")
EOF
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download dataset"
            echo "You can try:"
            echo "  pip install huggingface_hub"
            echo "  Or download manually and use --dataset_path"
            exit 1
        fi
        
        echo "✓ Dataset downloaded to: $DOWNLOAD_DIR"
    fi
else
    echo "Skipping download (--skip_download)"
    if [ -z "$DATASET_PATH" ]; then
        DATASET_PATH="$DOWNLOAD_DIR"
    fi
    if [ -n "$DATASET_PATH" ] && [ -d "$DATASET_PATH" ]; then
        DOWNLOAD_DIR="$DATASET_PATH"
        echo "Using existing dataset at: $DOWNLOAD_DIR"
    else
        echo "Warning: Dataset path not found. Please specify --dataset_path"
        exit 1
    fi
fi

# ==============================================================================
# Step 2: Unzip Trajectory Files
# ==============================================================================
echo ""
echo "============================================"
echo "Step 2: Unzipping Trajectory Files"
echo "============================================"

EXPAND_MEMORY_DIR="$DOWNLOAD_DIR/expand_memory"
if [ ! -d "$EXPAND_MEMORY_DIR" ]; then
    echo "Error: expand_memory directory not found in dataset"
    echo "Expected: $EXPAND_MEMORY_DIR"
    exit 1
fi

# Find ZIP files
ZIP_FILES=()
if [ -n "$MEMORY_PART" ]; then
    # Use specific memory part
    PART_ZIP="$EXPAND_MEMORY_DIR/expand_memory_${MEMORY_PART}.zip"
    if [ -f "$PART_ZIP" ]; then
        echo "Using memory part: $MEMORY_PART"
        echo "Found ZIP: expand_memory_${MEMORY_PART}.zip"
    else
        echo "Error: Memory part ZIP not found: expand_memory_${MEMORY_PART}.zip"
        echo "Available parts: part1, part2, part3"
        exit 1
    fi
elif [ -n "$DOMAIN_FILTER" ]; then
    # Filter by domain (only works with part1)
    DOMAIN_ZIP="$EXPAND_MEMORY_DIR/expand_memory_part1/${DOMAIN_FILTER}.zip"
    if [ -f "$DOMAIN_ZIP" ]; then
        ZIP_FILES+=("$DOMAIN_ZIP")
        echo "Found domain ZIP: ${DOMAIN_FILTER}.zip"
    else
        echo "Warning: Domain ZIP not found: ${DOMAIN_FILTER}.zip"
        echo "Available domains: shopping, academic, education, tech, travel, etc."
        exit 1
    fi
else
    # Find all ZIP files
    ZIP_COUNT=$(find "$EXPAND_MEMORY_DIR" -name "*.zip" -type f | wc -l)
    echo "Found $ZIP_COUNT ZIP files (will process all)"
fi

if [ "$SKIP_UNZIP" = false ]; then
    echo "Unzipping trajectory files..."
    
    if [ -n "$MEMORY_PART" ]; then
        # Handle specific memory part (e.g., part2.zip) using Python
        PART_ZIP="$EXPAND_MEMORY_DIR/expand_memory_${MEMORY_PART}.zip"
        PART_DIR="$EXPAND_MEMORY_DIR/expand_memory_${MEMORY_PART}"
        
        if [ -d "$PART_DIR" ]; then
            echo "  Already unzipped: expand_memory_${MEMORY_PART}"
        else
            echo "  Unzipping: expand_memory_${MEMORY_PART}.zip (this may take a while...)"
            python3 <<EOF
import zipfile
import os
import sys

zip_path = "$PART_ZIP"
extract_dir = "$EXPAND_MEMORY_DIR"

if not os.path.exists(zip_path):
    print(f"Error: ZIP file not found: {zip_path}")
    sys.exit(1)

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"  Extracting {len(zip_ref.namelist())} files...")
        zip_ref.extractall(extract_dir)
    print(f"✓ Successfully extracted to {extract_dir}")
except Exception as e:
    print(f"Error: Failed to unzip {zip_path}: {e}")
    sys.exit(1)
EOF
            if [ $? -ne 0 ]; then
                echo "  Error: Failed to unzip $PART_ZIP"
                exit 1
            fi
        fi
    else
        # Find and unzip ZIP files (domain filter or all) using Python
        find "$EXPAND_MEMORY_DIR" -name "*.zip" -type f | while read -r zip_file; do
            zip_dir=$(dirname "$zip_file")
            zip_name=$(basename "$zip_file" .zip)
            
            # Skip if domain filter is set and doesn't match
            if [ -n "$DOMAIN_FILTER" ] && [ "$zip_name" != "$DOMAIN_FILTER" ]; then
                continue
            fi
            
            # Skip part ZIPs if we're not processing them
            if [[ "$zip_name" == expand_memory_part* ]] && [ -z "$MEMORY_PART" ]; then
                continue
            fi
            
            # Check if already unzipped
            if [ -d "$zip_dir/$zip_name" ]; then
                echo "  Already unzipped: $zip_name"
                continue
            fi
            
            echo "  Unzipping: $zip_name.zip (this may take a while...)"
            python3 <<EOF
import zipfile
import os

zip_path = "$zip_file"
extract_dir = "$zip_dir"

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
except Exception as e:
    print(f"  Warning: Failed to unzip {zip_path}: {e}")
EOF
        done
    fi
    
    echo "✓ Unzipping complete"
else
    echo "Skipping unzip (--skip_unzip)"
fi

# ==============================================================================
# Step 3: Extract Trajectory Data
# ==============================================================================
echo ""
echo "============================================"
echo "Step 3: Extracting Trajectory Data"
echo "============================================"

# Create output directory
TRAJECTORY_OUTPUT_DIR="$ARC_MEMO_DIR/data/webvoyager_memory"
mkdir -p "$TRAJECTORY_OUTPUT_DIR"

echo "Scanning for trajectory files..."

# Create a Python script to extract trajectories from the dataset format
cat > "$PROJECT_ROOT/tmp_extract_trajectories.py" <<'PYEOF'
import json
import os
from pathlib import Path
import sys

def extract_from_codemem_dataset(dataset_dir, output_dir, domain_filter=None, memory_part=None):
    """Extract trajectories from CoMEM dataset format"""
    expand_memory_dir = Path(dataset_dir) / "expand_memory"
    output_trajectories = Path(output_dir) / "extracted_trajectories.json"
    if domain_filter:
        output_trajectories = Path(output_dir) / f"extracted_trajectories_{domain_filter}.json"
        output_descriptions = Path(output_dir) / f"extracted_task_descriptions_{domain_filter}.json"
    elif memory_part:
        output_trajectories = Path(output_dir) / f"extracted_trajectories_{memory_part}.json"
        output_descriptions = Path(output_dir) / f"extracted_task_descriptions_{memory_part}.json"
    else:
        output_descriptions = Path(output_dir) / "extracted_task_descriptions.json"
    
    trajectories = {}
    task_descriptions = {}
    
    # Walk through all subdirectories looking for trajectory files
    success_folders = []
    search_paths = []
    
    if memory_part:
        # Only search in the specific memory part directory
        part_dir = expand_memory_dir / f"expand_memory_{memory_part}"
        if part_dir.exists():
            search_paths.append(part_dir)
        else:
            print(f"Warning: Memory part directory not found: {part_dir}")
            return 0
    else:
        search_paths.append(expand_memory_dir)
    
    for search_path in search_paths:
        for root, dirs, files in os.walk(search_path):
            # Skip if domain filter is set and path doesn't contain domain
            if domain_filter:
                root_lower = root.lower()
                if domain_filter.lower() not in root_lower:
                    continue
            
            for dir_name in dirs:
                if dir_name == 'success':
                    success_folders.append(os.path.join(root, dir_name))
    
    print(f"Found {len(success_folders)} success folders")
    
    total_files = 0
    processed = 0
    
    for success_folder in success_folders:
        try:
            jsonl_files = [f for f in os.listdir(success_folder) if f.endswith('.jsonl')]
            total_files += len(jsonl_files)
            
            for jsonl_file in jsonl_files:
                file_path = os.path.join(success_folder, jsonl_file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract task description
                    task_description = data.get('task_description', 'Unknown task')
                    conversation_id = data.get('conversation_id', jsonl_file.replace('.jsonl', ''))
                    
                    # Extract actions from rounds
                    actions = []
                    rounds = data.get('rounds', [])
                    for round_data in rounds:
                        # Extract action from response or messages
                        response = round_data.get('response', '')
                        if response:
                            actions.append(response)
                    
                    if len(actions) >= 3:  # Minimum quality filter
                        # Use conversation_id as key
                        trajectories[conversation_id] = actions
                        task_descriptions[conversation_id] = task_description
                        processed += 1
                        
                except Exception as e:
                    print(f"Warning: Failed to process {file_path}: {e}")
                    continue
        except PermissionError:
            print(f"Warning: Permission denied accessing {success_folder}")
            continue
    
    print(f"Processed {processed} trajectories from {total_files} files")
    
    # Save trajectories
    with open(output_trajectories, 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f"Saved trajectories to: {output_trajectories}")
    
    # Save task descriptions
    with open(output_descriptions, 'w') as f:
        json.dump(task_descriptions, f, indent=2)
    print(f"Saved task descriptions to: {output_descriptions}")
    
    return processed

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]
    domain_filter = sys.argv[3] if len(sys.argv) > 3 else None
    memory_part = sys.argv[4] if len(sys.argv) > 4 else None
    extract_from_codemem_dataset(dataset_dir, output_dir, domain_filter, memory_part)
PYEOF

# Run extraction
python3 "$PROJECT_ROOT/tmp_extract_trajectories.py" "$DOWNLOAD_DIR" "$TRAJECTORY_OUTPUT_DIR" "$DOMAIN_FILTER" "$MEMORY_PART"

if [ $? -eq 0 ]; then
    echo "✓ Trajectories extracted successfully"
    if [ -n "$DOMAIN_FILTER" ]; then
        TRAJECTORY_FILE="$TRAJECTORY_OUTPUT_DIR/extracted_trajectories_${DOMAIN_FILTER}.json"
        TASK_DESC_FILE="$TRAJECTORY_OUTPUT_DIR/extracted_task_descriptions_${DOMAIN_FILTER}.json"
    elif [ -n "$MEMORY_PART" ]; then
        TRAJECTORY_FILE="$TRAJECTORY_OUTPUT_DIR/extracted_trajectories_${MEMORY_PART}.json"
        TASK_DESC_FILE="$TRAJECTORY_OUTPUT_DIR/extracted_task_descriptions_${MEMORY_PART}.json"
    else
        TRAJECTORY_FILE="$TRAJECTORY_OUTPUT_DIR/extracted_trajectories.json"
        TASK_DESC_FILE="$TRAJECTORY_OUTPUT_DIR/extracted_task_descriptions.json"
    fi
    
    if [ -f "$TRAJECTORY_FILE" ]; then
        TRAJECTORY_COUNT=$(python3 -c "import json; print(len(json.load(open('$TRAJECTORY_FILE'))))")
        echo "  Found $TRAJECTORY_COUNT trajectories"
    fi
else
    echo "Error: Failed to extract trajectories"
    exit 1
fi

# Cleanup temp script
rm -f "$PROJECT_ROOT/tmp_extract_trajectories.py"

# ==============================================================================
# Step 4: Update Config Files
# ==============================================================================
echo ""
echo "============================================"
echo "Step 4: Updating Config Files"
echo "============================================"

# Update config_simple.yaml
CONFIG_SIMPLE="$ARC_MEMO_DIR/configs/config_simple.yaml"
if [ -f "$CONFIG_SIMPLE" ]; then
    # Update paths using Python (only update trajectory paths)
    python3 <<EOF
import yaml
import os
from pathlib import Path

config_file = "$CONFIG_SIMPLE"
trajectory_file = "$TRAJECTORY_FILE"
task_desc_file = "$TASK_DESC_FILE"
project_root = "$PROJECT_ROOT"

# Convert to relative paths
def to_relative(path, base=project_root):
    try:
        return os.path.relpath(path, base)
    except:
        return path

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Only update trajectory paths (they may change based on domain filtering)
config['trajectories'] = to_relative(trajectory_file)
config['task_descriptions'] = to_relative(task_desc_file)

# Output dir is already relative, no need to update

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✓ Updated {config_file}")
EOF
    echo "  Trajectories: $TRAJECTORY_FILE"
    echo "  Task Descriptions: $TASK_DESC_FILE"
else
    echo "Warning: Config file not found: $CONFIG_SIMPLE"
fi

# Update config_abstract.yaml
CONFIG_ABSTRACT="$ARC_MEMO_DIR/configs/config_abstract.yaml"
if [ -f "$CONFIG_ABSTRACT" ]; then
    # Config already has relative paths, no need to update
    echo "✓ Config file already uses relative paths: $CONFIG_ABSTRACT"
    echo "  Pseudocode Input: arc_memo/output/initial_analysis.json"
    echo "  Output Directory: arc_memo/output"
else
    echo "Warning: Config file not found: $CONFIG_ABSTRACT"
fi

# ==============================================================================
# Step 5: Verify Setup
# ==============================================================================
echo ""
echo "============================================"
echo "Step 5: Verification"
echo "============================================"

# Check trajectory files
if [ -f "$TRAJECTORY_FILE" ]; then
    echo "✓ Trajectory file exists: $TRAJECTORY_FILE"
    FILE_SIZE=$(du -h "$TRAJECTORY_FILE" | cut -f1)
    echo "  Size: $FILE_SIZE"
else
    echo "✗ Trajectory file NOT found: $TRAJECTORY_FILE"
fi

if [ -f "$TASK_DESC_FILE" ]; then
    echo "✓ Task descriptions file exists: $TASK_DESC_FILE"
    FILE_SIZE=$(du -h "$TASK_DESC_FILE" | cut -f1)
    echo "  Size: $FILE_SIZE"
else
    echo "✗ Task descriptions file NOT found: $TASK_DESC_FILE"
fi

# Check config files
if [ -f "$CONFIG_SIMPLE" ]; then
    echo "✓ Config file exists: $CONFIG_SIMPLE"
else
    echo "✗ Config file NOT found: $CONFIG_SIMPLE"
fi

if [ -f "$CONFIG_ABSTRACT" ]; then
    echo "✓ Abstract config file exists: $CONFIG_ABSTRACT"
else
    echo "✗ Abstract config file NOT found: $CONFIG_ABSTRACT"
fi

# ==============================================================================
# Final Summary
# ==============================================================================
echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Make sure vLLM server is running:"
echo "   python -m vllm.entrypoints.openai.api_server \\"
echo "     --model Qwen/Qwen2.5-VL-7B-Instruct \\"
echo "     --port 8000"
echo ""
echo "2. Run the arc_memo pipeline:"
echo "   ./arc_memo/run_arc_memo.sh --skip_extract"
echo ""
echo "3. Or run full pipeline (if you want to re-extract):"
echo "   ./arc_memo/run_arc_memo.sh"
echo ""
echo "Dataset Location: $DOWNLOAD_DIR"
echo "Trajectories: $TRAJECTORY_FILE"
echo "Config Files: Updated with correct paths"
echo ""
echo "============================================"

