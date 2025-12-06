# World Model for GUI Agent Learning from Errors

A world model system that enables GUI agents to learn from both successful and failed trajectories through contrastive learning and error analysis.

## Overview

This project provides two main components for improving GUI agent performance:

1. **Contrastive Experience Learning** - Learn what distinguishes successful trajectories from failures
2. **Critical Error Detection** - Identify root causes of failures for targeted improvement

The world model generates text guidance that is injected into agent prompts, helping the agent avoid common mistakes and follow successful patterns.

## Architecture

```
world_model/
├── contrasive_experience/
│   ├── world_model/
│   │   ├── world_model.py          # Main orchestrator
│   │   ├── trajectory_store.py     # FAISS-indexed trajectory storage
│   │   ├── trajectory_analyzer.py  # Contrastive pair analysis
│   │   ├── contrastive_summarizer.py # Text summary generation
│   │   ├── state_predictor.py      # Action outcome prediction
│   │   └── prompts/                # Prompt templates
│   └── compute_success_rate.py     # Evaluation utility
│
└── failure_traj_analysis/
    ├── critical_error_detection.py # Root cause identification
    ├── experience_memory.py        # Memory retrieval system
    ├── fine_grained_analysis.py    # Detailed error analysis
    ├── evaluator.py                # Evaluation utilities
    ├── early_stop.py               # Early stopping detection
    └── reasoning_bank.py           # Reasoning patterns
```

## Key Features

### Contrastive Learning

- **Separate Success/Failure Indices**: Maintains FAISS indices for success and failure trajectories separately
- **Contrastive Pair Retrieval**: Retrieves similar successful AND failed trajectories for any new task
- **Pattern Extraction**: Identifies what successful agents do differently from failed ones
- **Multimodal Support**: Uses CLIP embeddings for both text and screenshot-based similarity

### World Model Guidance

The world model provides guidance at two stages:

**Initial Guidance** (before first action):
- Retrieves similar past trajectories
- Analyzes contrastive patterns
- Generates ~200 token summary injected into system prompt

**Step Guidance** (at each action):
- Re-retrieves trajectories based on current screenshot
- Provides context-aware recommendations
- Warns about potential stuck states or repeated actions

### Critical Error Detection

Identifies the earliest critical error that led to task failure:
- Memory errors (forgetting important information)
- Reflection errors (incorrect self-assessment)
- Planning errors (wrong strategy)
- Action errors (wrong action execution)
- System errors (step limits, environment issues)

## Usage

### Basic World Model Integration

```python
from world_model.world_model import WorldModel, create_world_model

# Initialize world model
world_model = create_world_model(args, tool_llm)

# Get initial guidance before first action
initial_guidance = world_model.get_initial_guidance(
    task="Search for flights to Sydney",
    initial_screenshot=base64_screenshot,
    domain="shopping",
    dataset="mmina"
)

# Get step-by-step guidance during execution
step_guidance = world_model.get_step_guidance(
    task="Search for flights to Sydney",
    current_state=current_screenshot,
    action_history=["click", "type", "click"],
    step_num=3
)
```

### Trajectory Store Usage

```python
from world_model.trajectory_store import TrajectoryStore

# Initialize with training data
store = TrajectoryStore(
    training_data_path="training_data",
    multimodal=True
)

# Retrieve contrastive pairs
success_trajs, failure_trajs = store.retrieve_contrastive_pairs(
    query_task="Book a hotel in Tokyo",
    query_image=screenshot,
    domain="booking",
    top_k=3
)
```

### Critical Error Analysis

```python
from failure_traj_analysis.critical_error_detection import CriticalErrorAnalyzer

analyzer = CriticalErrorAnalyzer()
result = analyzer.process_trajectory(
    phase1_file="error_detection_results/task_123_errors.json",
    original_trajectory_file="html/render_123.html",
    output_dir="error_detection_results"
)

print(f"Critical error at step {result['critical_error']['critical_step']}")
print(f"Error type: {result['critical_error']['error_type']}")
print(f"Root cause: {result['critical_error']['root_cause']}")
```

## Data Format

### Trajectory Files (JSONL)

```json
{
  "task_description": "Find flights from LA to Sydney",
  "total_rounds": 8,
  "rounds": [
    {
      "messages": [...],
      "response": "Action: {\"name\": \"click\", \"arguments\": {...}}"
    }
  ]
}
```

### Directory Structure for Training Data

```
training_data/
├── mmina/
│   ├── shopping/
│   │   ├── success/
│   │   │   └── *.jsonl
│   │   └── failure/
│   │       └── *.jsonl
│   └── wikipedia/
│       ├── success/
│       └── failure/
└── webvoyager/
    └── ...
```

## Evaluation

Compute success rates from evaluation results:

```bash
python compute_success_rate.py --result_dir results/mmina/shopping/qwen2.5-vl/2024-01-15

# Or find recent results
python compute_success_rate.py --eval_type mmina --domain shopping --recent 5 --verbose
```

## Configuration

Key arguments for world model:
- `--use_world_model`: Enable world model guidance
- `--world_model_data_path`: Path to trajectory training data
- `--world_model_index_path`: Path to pre-built FAISS index
- `--world_model_top_k`: Number of trajectories to retrieve (default: 3)
- `--world_model_multimodal`: Use multimodal embeddings (default: True)
- `--world_model_step_guidance`: Enable per-step guidance (default: True)

## How It Works

### Contrastive Learning Pipeline

1. **Retrieval**: Given a new task and screenshot, retrieve top-k similar success AND failure trajectories using FAISS
2. **Analysis**: Compare action sequences, identify divergence points where success/failure paths differ
3. **Summarization**: Generate concise text guidance highlighting:
   - Success patterns to follow
   - Common mistakes to avoid
   - Key decision points
4. **Injection**: Insert guidance into agent prompts (system message + user message)

### Error Detection Pipeline

1. **Phase 1**: Fine-grained error detection at each step (memory, reflection, planning, action)
2. **Phase 2**: Critical error identification - find the earliest error that caused failure
3. **Output**: Structured analysis with root cause, evidence, and correction guidance

## Dependencies

- `faiss-cpu` or `faiss-gpu`: Vector similarity search
- `transformers`: CLIP model for embeddings
- `torch`: Deep learning backend
- `beautifulsoup4`: HTML parsing for trajectory files
- `numpy`: Numerical operations

## Related Projects

This world model integrates with the CoMEM-Agent system for GUI automation. See the parent project for full agent implementation and benchmarks.
