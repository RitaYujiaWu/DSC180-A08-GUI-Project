# CoMEM-Agent Inference & World Model

The complete CoMEM-Agent inference system for GUI automation, integrating experience memory, contrastive world models, and function-calling agents to learn from both successful and failed trajectories.

## Architecture Overview

```
contrasive_experience/
├── run.py / run_chunks.py / run_all_domains.sh  # Entry points
├── config/
│   └── argument_parser.py         # All CLI arguments
│
├── agent/                         # FunctionCallAgent + LLM configs + prompts
├── browser_env/                   # Playwright browser automation
├── actions/                       # Action creation and parsing
├── action_scaling/                # Historical state-action matching
│
├── memory/                        # Experience memory + reasoning bank
├── arc_memo/                      # Concept memory (ArcMemo)
├── graph_memory/                  # Graph-based memory
├── hybrid_memory/                 # Hybrid memory system
├── memory_evolution/              # Data flywheel pipeline
│
├── world_model/                   # Contrastive learning world model
│   ├── world_model.py             # Main orchestrator
│   ├── trajectory_store.py        # FAISS-indexed trajectory storage
│   ├── trajectory_analyzer.py     # Contrastive pair analysis
│   ├── contrastive_summarizer.py  # Text summary generation
│   ├── state_predictor.py         # Action outcome prediction
│   └── prompts/                   # Prompt templates
│
├── tools/                         # GUI tools, analysis tools, web search
├── benchmarks/                    # MMInA, Mind2Web, WebVoyager evaluators
├── scripts/                       # Runner scripts, ablation scripts
├── utils/                         # Shared utilities
│
├── data_preparation/              # FAISS index building
├── training_data/                 # Training trajectory data
├── MMInA_evaluation/              # MMInA benchmark data
├── Mind2Web_evaluation/           # Mind2Web benchmark data
├── webvoyager_evaluation/         # WebVoyager benchmark data
│
├── compute_success_rate.py        # Evaluation utility
├── evaluation.ipynb               # Evaluation notebook
└── requirements_web.txt           # Python dependencies
```

## Quick Start

### 1. Start vLLM Servers

See [VLLM_SERVER_SETUP.md](VLLM_SERVER_SETUP.md) for detailed instructions on setting up the model serving backend.

### 2. Run Evaluations

```bash
# Single domain evaluation
python run.py \
  --model qwen2.5-vl \
  --eval_type mmina \
  --domain shopping \
  --use_world_model \
  --use_memory

# Run all domains
bash run_all_domains.sh

# Run in chunks (for parallelization)
python run_chunks.py --chunk_id 0 --num_chunks 4
```

### 3. Compute Success Rates

```bash
python compute_success_rate.py --result_dir results/mmina/shopping/qwen2.5-vl/2024-01-15

# Or find recent results
python compute_success_rate.py --eval_type mmina --domain shopping --recent 5 --verbose
```

See [SUCCESS_RATE_README.md](SUCCESS_RATE_README.md) for more details.

## Key Components

### Agent System (`agent/`)

The `FunctionCallAgent` orchestrates the full agent loop: receiving tasks, observing browser state, selecting actions via function calling, and managing memory retrieval. LLM configurations and prompt templates live here.

### Browser Environment (`browser_env/`)

Playwright-based browser automation that provides the agent with screenshots, accessibility trees, and action execution (click, type, scroll, navigate).

### Memory Systems

- **Experience Memory** (`memory/`) — Retrieves relevant past trajectories and reasoning patterns using FAISS similarity search
- **ArcMemo** (`arc_memo/`) — Concept-level memory for abstract task patterns
- **Graph Memory** (`graph_memory/`) — Graph-structured memory for relational reasoning
- **Hybrid Memory** (`hybrid_memory/`) — Combines multiple memory sources
- **Memory Evolution** (`memory_evolution/`) — Data flywheel for continuously improving memory from new trajectories

### World Model (`world_model/`)

Contrastive learning system that compares successful vs. failed trajectories to generate text guidance:

- **Initial Guidance**: Before the first action, retrieves similar past trajectories and generates a ~200-token summary of success patterns and common pitfalls
- **Step Guidance**: At each action, provides context-aware recommendations based on the current screenshot and action history

### Function Calling Tools (`tools/`)

GUI interaction tools (click, type, scroll), analysis tools, and web search capabilities exposed to the agent via function calling. See [TOOLS.md](TOOLS.md) for the full tool reference.

## Configuration

All CLI arguments are defined in `config/argument_parser.py`. Key flags:

- `--model`: Model to use (e.g., `qwen2.5-vl`)
- `--eval_type`: Benchmark type (`mmina`, `mind2web`, `webvoyager`)
- `--domain`: Task domain (e.g., `shopping`, `wikipedia`)
- `--use_world_model`: Enable contrastive world model guidance
- `--use_memory`: Enable experience memory retrieval
- `--world_model_top_k`: Number of trajectories to retrieve (default: 3)
- `--world_model_multimodal`: Use multimodal embeddings (default: True)
- `--world_model_step_guidance`: Enable per-step guidance (default: True)

## World Model Details

### Contrastive Learning Pipeline

1. **Retrieval**: Given a new task and screenshot, retrieve top-k similar success AND failure trajectories using FAISS
2. **Analysis**: Compare action sequences, identify divergence points where success/failure paths differ
3. **Summarization**: Generate concise text guidance highlighting success patterns, common mistakes, and key decision points
4. **Injection**: Insert guidance into agent prompts (system message + user message)

### Critical Error Detection

Identifies the earliest critical error that led to task failure:
- Memory errors (forgetting important information)
- Reflection errors (incorrect self-assessment)
- Planning errors (wrong strategy)
- Action errors (wrong action execution)
- System errors (step limits, environment issues)

## Data Format

### Trajectory Files (JSONL)

```json
{
  "task_description": "Find flights from LA to Sydney",
  "total_rounds": 8,
  "rounds": [
    {
      "messages": ["..."],
      "response": "Action: {\"name\": \"click\", \"arguments\": {...}}"
    }
  ]
}
```

### Training Data Structure

```
training_data/
├── mmina/
│   ├── shopping/
│   │   ├── success/*.jsonl
│   │   └── failure/*.jsonl
│   └── wikipedia/
│       ├── success/
│       └── failure/
└── webvoyager/
    └── ...
```

## Dependencies

See `requirements_web.txt` for the full list. Key dependencies:

- `faiss-cpu` / `faiss-gpu` — Vector similarity search
- `transformers` — CLIP model for embeddings
- `torch` — Deep learning backend
- `playwright` — Browser automation
- `beautifulsoup4` — HTML parsing
