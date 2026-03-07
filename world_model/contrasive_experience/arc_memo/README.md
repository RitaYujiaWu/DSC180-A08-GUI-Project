# GUI Agent Concept Memory (ArcMemo for GUI)

A memory framework for GUI agents that learns reusable concepts from successful task trajectories. Based on the [ArcMemo](https://github.com/xu3kev/arc-memory) framework, adapted for web navigation tasks.

## Overview

This system implements a three-stage pipeline to build and utilize concept memory:

1. **Pseudocode Generation**: Convert raw GUI agent trajectories into high-level pseudocode
2. **Concept Abstraction**: Extract reusable concepts from pseudocode with LLM-based intelligent merging
3. **Concept Selection**: Retrieve relevant concepts for new tasks using CLIP embeddings

## Architecture

```
Raw Trajectories → Pseudocode → Concepts → Memory → Retrieval → Prompt Suggestions
```

### Key Features

- **Intelligent Concept Merging**: When multiple trajectories produce concepts with the same name, LLM intelligently merges them into 1-3 unified concepts (avoiding simple concatenation)
- **CLIP-based Retrieval**: Uses semantic similarity between task descriptions and concept cues for accurate retrieval
- **Fine-grained Concept Types**: 8 categories (ui_navigation, data_entry, search_filter, authentication, verification, data_extraction, form_handling, selection)
- **Modular Design**: Each stage can run independently with configurable inputs/outputs

## Directory Structure

```
arc_memo/
├── concept_mem/           # Core scripts
│   ├── extract_trajectories.py    # Extract from WebVoyager logs
│   ├── pseudocode_simple.py       # Stage 1: Trajectory → Pseudocode
│   ├── abstract_simple.py         # Stage 2: Pseudocode → Concepts
│   ├── select_simple.py           # Stage 3: Concept Retrieval
│   ├── compress_memory.py         # Convert memory.json to prompt string
│   ├── concept.py                 # Concept & ConceptMemory classes
│   ├── utils.py                   # Shared utilities
│   └── constants.py               # Path constants
├── configs/               # Configuration files
│   ├── config_abstract.yaml       # Abstraction config
│   └── annotate/gui_pseudocode.yaml  # Pseudocode config
├── data/                  # Instructions & examples
│   ├── abstract_anno/gui/
│   │   ├── pseudocode_instr.txt   # Pseudocode generation prompt
│   │   ├── concept_instr.txt      # Concept abstraction prompt
│   │   ├── example_annotations.yaml    # ICL examples
│   │   └── example_concepts.yaml       # ICL concept examples
│   └── webvoyager_memory/
│       ├── extracted_trajectories.json      # Parsed trajectories
│       └── extracted_task_descriptions.json # Task descriptions
└── output/                # Generated outputs
    ├── initial_analysis.json      # Pseudocode output
    ├── concept_lists.json         # Raw concepts per task
    ├── memory.json                # Merged concept memory
    ├── prompt_info.json           # Selected concepts for retrieval
    └── ps_mem.txt                 # Formatted prompt suggestions
```

## Pipeline Stages

### Stage 0: Data Extraction 

Extract trajectories from WebVoyager success logs:

```bash
python concept_mem/extract_trajectories.py
```

**Input**: WebVoyager logs with action sequences  
**Output**: `data/webvoyager_memory/extracted_trajectories.json`, `extracted_task_descriptions.json`

### Stage 1: Pseudocode Generation

Convert action sequences into high-level pseudocode:

```bash
python concept_mem/pseudocode_simple.py \
  --config_path configs/annotate/gui_pseudocode.yaml \
  --output_dir output
```

**Input**: 
- Raw trajectories (JSON with action lists)
- Task descriptions
- ICL examples (`example_annotations.yaml`)

**Output**: `output/initial_analysis.json`

**Example**:
```json
{
  "Amazon--1": {
    "pseudocode": [
      "Navigate to search interface",
      "Enter search query for women's golf polos",
      "Apply filters: size M, price $50-$75",
      "Sort results by price ascending",
      "Save lowest priced item"
    ],
    "summary": "Search and filter products with specific criteria"
  }
}
```

### Stage 2: Concept Abstraction

Extract reusable concepts from pseudocode with intelligent merging:

```bash
python concept_mem/abstract_simple.py \
  --config_path configs/config_abstract.yaml \
  --output_dir output
```

**Input**:
- Pseudocode (`initial_analysis.json`)
- ICL examples (`example_concepts.yaml`)

**Output**: 
- `concept_lists.json`: Raw concepts per task (before merging)
- `memory.json`: Unified concept memory (after LLM-based merging)

**Key Feature - Intelligent Merging**:
When multiple tasks generate concepts with the same name (e.g., `filter_products`), the LLM:
1. Detects if implementations are similar → merge into ONE refined concept
2. Detects distinct approaches → create 2-3 variants (e.g., `filter_products_simple`, `filter_products_advanced`)
3. Removes redundancy and preserves key patterns

**Example Concept**:
```yaml
- concept: filter_products
  type: search_filter
  cues:
    - filter interface visible (dropdown, sidebar, or menu)
    - search results need refinement
    - multiple criteria available
  implementation:
    - Locate and access the filter interface
    - Select desired filter criteria (price, size, category)
    - Apply the filters using the apply/confirm button
    - Wait for filtered results to load
```

### Stage 3: Concept Selection & Retrieval

Retrieve relevant concepts for a new task using CLIP embeddings:

```bash
python concept_mem/select_simple.py \
  --task_id "NewTask--1" \
  --reasoning_plan "Search for products and apply multiple filters" \
  --memory_path output/memory.json \
  --output_dir output \
  --top_k 5
```

**Input**:
- Task reasoning plan (string describing approach)
- Concept memory (`memory.json`)

**Output**:
- `prompt_info.json`: Selected concept metadata
- `ps_mem.txt`: Formatted prompt with implementation guidance

**Selection Method**:
1. Compute CLIP embeddings for all concept cues
2. Compute embedding for the reasoning plan
3. Calculate cosine similarity between plan and each cue
4. Select top-k cues and their corresponding concepts
5. Format as structured prompt with implementation steps

**Example Output (`ps_mem.txt`)**:
```markdown
# Implementation Guidance

Based on similar tasks, here are relevant implementation patterns:

## Search Filter

### Filter Products
1. Locate and access the filter interface (may be a dropdown, sidebar, or menu).
2. Select desired filter criteria such as price range, size, or category.
3. Apply the filters using the apply/confirm button.
4. Wait for the page to reload with filtered results.

## Data Entry

### Enter Search Query
1. Identify the search input field.
2. Click to focus the field.
3. Type the search terms clearly.
4. Press enter or click the search button.

---
**Note:** Adapt these patterns to your specific task requirements and UI structure.
```

## Configuration

### Model Setup

Edit config files to specify your LLM endpoint:

```yaml
# configs/config_abstract.yaml
model:
  server_url: "http://localhost:8000/v1"  # vLLM server
  name: "Qwen/Qwen2.5-VL-7B-Instruct"     # Model name
  api_key: "EMPTY"                        # API key
```

### Concept Types

8 fine-grained categories defined in `concept_instr.txt`:

- **ui_navigation**: Navigate menus, tabs, pages
- **data_entry**: Fill forms, input text
- **search_filter**: Search and filter results
- **authentication**: Login, logout, session management
- **verification**: Check results, validate states
- **data_extraction**: Extract information from UI
- **form_handling**: Submit forms, handle validations
- **selection**: Select items, options, checkboxes

## Concept Schema

Each concept contains:

```yaml
concept: <name>              # Unique identifier
type: <category>             # One of 8 types above
cues:                        # When to use this concept
  - <identifying pattern 1>
  - <identifying pattern 2>
implementation:              # How to implement
  - <step 1>
  - <step 2>
```

## Usage Example: Full Pipeline

```bash
# 1. Extract trajectories from logs (optional)
python concept_mem/extract_trajectories.py

# 2. Generate pseudocode
python concept_mem/pseudocode_simple.py \
  --config_path configs/annotate/gui_pseudocode.yaml \
  --output_dir output

# 3. Abstract and merge concepts
python concept_mem/abstract_simple.py \
  --config_path configs/config_abstract.yaml \
  --output_dir output

# 4. Compress memory to text (optional)
python concept_mem/compress_memory.py \
  --memory output/memory.json \
  --out output/gui_init_mem.txt

# 5. Retrieve concepts for new task
python concept_mem/select_simple.py \
  --task_id "Amazon--42" \
  --reasoning_plan "Login, search products, add to cart, checkout" \
  --memory_path output/memory.json \
  --top_k 8
```

## Key Improvements Over Basic Concatenation

| Aspect | Before (Concat) | After (LLM Merge) |
|--------|----------------|-------------------|
| Implementation Steps | 8-15 per concept | 3-5 per concept |
| Cues | 6-10 (overlapping) | 2-4 (focused) |
| Redundancy | High | Low |
| Clarity | Confusing | Clear |
| Retrieval Quality | Noisy | Precise |

**Example**: 3 tasks with `filter_products` concept
- **Before**: 13 steps concatenated, 8 overlapping cues
- **After**: 4 refined steps, 3 focused cues covering all cases

## Resumable Execution

If `concept_lists.json` exists, Stage 2 will skip generation and load from file:

```python
if os.path.exists(output_dir/"concept_lists.json"):
    print("Loading concept lists from file...")
    concept_batch = read_json(output_dir/"concept_lists.json")
```

## Logging

Logs are saved to `output/logs/`:
- `pseudocode_generation.log`: Stage 1 logs
- `concept_abstraction.log`: Stage 2 logs (including merge operations)
- `select_memory.log`: Stage 3 logs

