# DSC180 A08 Capstone Project - GUI Agent
Right now, there’s a lot of excitement around GUI agents, which are AI agents that interact with graphical user interfaces—things like apps, browsers, and operating systems. Therefore, we chose this as our topic and focused on improving the GUI agent performance through inference-time scaling.

Our team is then split into two groups that strengthen the agent from complementary directions: one group trains a Process Reward Model (PRM) as an external signal [Jump to Part A](#a-training-a-process-reward-model-prm-for-gui-agent), while the other develops an internal world model to enhance the agent’s own reasoning [Jump to Part B](#b-training-an-internal-world-model-for-gui-agent).

# A. Training a Process Reward Model (PRM) for GUI Agent
## Part I — Task Generation and Collecting Agent Trajectory (Based on ZeroGUI and OSWorld's framework)
🎯 **Goal:** Use ZeroGUI's prompt for task generation based on OSWorld's tasks and then collect trajectory using Qwen3VL-4b as agent. \
📈 **Next Step:** Passing all the generated trajectories to the Reward labeling part and preparing for the fine-tuning.

### 1. Project Layout
`git clone` ZeroGUI's [repo](https://github.com/OpenGVLab/ZeroGUI) under the root and replace the pointing files with the files we provided under the `/agent` folder:
<pre><code>repo-root/
  zerogui/
    osworld/
      env_api_manager.py
      ...
      task_generation.py          # <-- change to our modified file
      task_generation_meta.py     # <-- change to our modified file
    ...
</code></pre>

`git clone` OSWorld's [repo](https://github.com/xlang-ai/OSWorld) under the root, and replace the pointing files with the files we provided under the `/OSWorld` folder:
<pre><code>repo-root/
  zerogui
  OSWorld/
    mm_agents/
      ...
      qwen3vl_agent.py            # <-- change to our modified file
    ...
    lib_run_single.py             # <-- change to our modified file
    run_multienv_qwen3vl.py       # <-- change to our modified file
  ...
</code></pre>
### 2. Running Task Generation
You can check out detailed steps here: [Task Generation setup](https://github.com/OpenGVLab/ZeroGUI/tree/main/osworld#task-generation) \
Simply, you can run the following steps:
#### 2.1. OSWorld Configuration
Follow the instruction in OSWorld's [repo](https://github.com/xlang-ai/OSWorld) to set up authentic files.
#### 2.2. Generating step_0 screenshots from OSWorld
<pre><code>cd OSWorld
python run_multienv_qwen3vl.py \
  --test_all_meta_path evaluation_examples/test_all.json \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --max_steps 0 \
  --result_dir ./results/all_step0/</code></pre>
#### 2.3. ZeroGUI Configuration:
configure the following parameters in task_generation.py:
- OpenAI API Key
- Result directory path (location of screenshots, e.g., `YOUR_ROOT/OSWorld/results/all_step0`)
- Save directory path (default: `YOUR_ROOT/OSWorld/evaluation_examples/generated_examples/`, **must** match examples_dir in task_generation_meta.py)
- Option to generate infeasible tasks
#### 2.4. Run Task Generation
<pre><code>cd ZeroGUI/osworld
python task_generation.py</code></pre>
#### 2.5. Save Task IDs in Required Format
<pre><code>cd ZeroGUI/osworld
python make_meta_from_folder.py \
  --examples_root YOUR_ROOT/OSWorld/evaluation_examples/generated_examples \
  --output YOUR_ROOT/OSWorld/evaluation_examples/test_generated.json</code></pre>
### 3. Running Inference for Trajectory
#### 3.1. Use VLLM to host Qwen3VL-4b Locally
<pre><code>pip install vllm
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --trust-remote-code 
  --port 8000 
  --gpu_memory_utilization 0.65    # <-- Optional line, set if your GPU memory is not enough</code></pre>
#### 3.2. Running Inference
<pre><code>cd OSWorld
python run_multienv_qwen3vl.py \
  --test_all_meta_path evaluation_examples/test_generated.json \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --max_steps 25 \
  --sleep_after_execution 0.5 \
  --num_envs 8 \
  --examples_dir generated_examples \
  --result_dir ./results/generated_maxstep25_sleep0.5/
</code></pre>

## Part II — Reward Labeling (Using GPT-5-mini)
🎯 **Goal:** Convert raw GUI trajectories (screenshots + actions) into step-wise, scalar rewards using a multimodal LLM (GPT-5-mini), so we can train a Process Reward Model (PRM) that scores each action in context. \
📈 **Next Step:** Aggregate the annotated trajectories into a PRM-friendly format and plug it into the fine-tuning pipeline in Part III.

### TL;DR
- We take trajectories generated in Part I (`traj.jsonl` + screenshots) as input.
- For each step, we load the task instruction, compare BEFORE/AFTER screenshots, and call GPT-5-mini to assign a reward in \[0.0, 1.0\] plus a brief reason.
- We write one JSON line per trajectory with all steps annotated, which becomes the supervision data for training the PRM.

Before running, set the environment variable `EXAMPLES_ROOT` to point to your OSWorld examples directory:
```bash
export EXAMPLES_ROOT=YOUR_PATH/OSWorld/evaluation_examples/examples
```

### 1. Dry Run (no API calls)
<pre><code>cd YOUR_PATH/PRM_baseline/reward
python step_rewards_annotation.py \
  --base-dir YOUR_PATH/OSWorld/results/generated_maxstep25_sleep0.5 \
  --output-file YOUR_PATH/PRM_baseline/reward/annotated_traj_debug.jsonl \
  --model openai/gpt-5-mini \
  --workers 1 \
  --dry-run
</code></pre>
### 2. Full Annotation Run (with API)
<pre><code>cd YOUR_PATH/PRM_baseline/reward
python step_rewards_annotation.py \
  --base-dir YOUR_PATH/OSWorld/results/generated_maxstep25_sleep0.5 \
  --output-file YOUR_PATH/PRM_baseline/reward/annotated_traj.jsonl \
  --model openai/gpt-5-mini \
  --workers 16
</code></pre>

## Part III — PRM Finetuning  (Based on Llama Factory's framework)
🎯 **Goal:** Fine-tune Qwen3VL-4b on the collected trajectory with reward. \
📈 **Next Step:** Keep testing and refining so the data and fine-tuning works best.

To fine-tune, `git clone` Llama Factory's [repo](https://github.com/hiyouga/LLaMA-Factory), follow the setup instructions. \
Add the file `PRM_baseline/agent/convert_to_llamafactory.py` under `YOUR_ROOT/LLaMA-Factory` and run:
<pre><code>python3 YOUR_ROOT/LLaMA-Factory/convert_to_llamafactory.py \
  --annotated-jsonl (jsonl path of the annotated data) \
  --output YOUR_ROOT/LLaMA-Factory/annot_windows.jsonl (targeted output path) \
  --window-size (context window length) \
  --max-images -1 (keep as -1, unless you want the first few images only for the whole case and dispose of the others)
</code></pre>
Add the following to the end of the file `YOUR_ROOT/LLaMA-Factorydata/dataset_info.json`:
<pre><code>"annot_windows": {
    "file_name": "YOUR_ROOT/LLaMA-Factory/annot_windows.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations", "images": "images" },
    "tags": { "role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant" }
  }</code></pre>
Then, run finetuning using the following (with your desired settings and available GPUs):
<pre><code>CUDA_VISIBLE_DEVICES=3,5 python -m llamafactory.cli train \
  --stage sft \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --trust_remote_code True \
  --dataset annot_windows \
  --dataset_dir YOUR_ROOT/LLaMA-Factory/data \
  --output_dir YOUR_ROOT/qwen3vl_sft_annot_ckpts \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --cutoff_len 3072 \
  --fp16 \
  --gradient_checkpointing \
  --freeze_vision_tower \
  --freeze_multi_modal_projector \
  --template qwen2_vl \
  --do_train True \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 150 \
  --logging_steps 1 \
  --report_to wandb \
  --run_name qwen3vl_sft_annot_run1</code></pre>
Finally, save your trained PRM:
<pre><code>python -m llamafactory.cli export \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --adapter_name_or_path YOUR_ROOT/qwen3vl_sft_annot_ckpts/YOUR_CKPT \
  --export_dir YOUR_ROOT/qwen3vl_merged_final</code></pre>

## Part IV — Agent RL Training with PRM (Started with PPO)
🎯 **Goal:** Train the agent using RL with PRM. \
📈 **Next Step:** Keep testing and refining to find the best policy/parameter settings.

To start, locally host your trained PRM using vllm (adjust the settings based on your available GPUs):
<pre><code>CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model YOUR_ROOT/qwen3vl_merged_final \
  --served-model-name qwen3vl-prm \
  --gpu_memory_utilization 0.75 \
  --host 0.0.0.0 \
  --port 8003</code></pre>
Then, change or modify your desired settings in `agent_training_prm/configs/train.yaml` \
Start training by (setting the GPUs to your available ones and change num_processes accordingly):
<pre><code>cd agent_training_prm
CUDA_VISIBLE_DEVICES=0,5 accelerate launch --num_processes 2 -m src.train --config configs/train.yaml</code></pre>
This code will automatically connect to OSWorld and start docker environment. During the rollout, PRM will give a reward of 0/1 for bad/good actions. Finally, the loss will be calculated, and the policy will be updated accordingly. The training checkpoint and trajectory logs can be found in `agent_training_prm/runs`, which is updated during the training.

## Part V — Online Data Evaluation (Based on ZeroGUI's framework)
📝 **Status:** AndroidWorld environment is interactive; adapter wired. OSWorld/AndroidLab are inherently compatible. \
🎯 **Goal:** Integrate AndroidWorld into online evaluation by running it inside Docker while exposing a ZeroGUI-compatible environment. \
📈 **Next Step:** Try evaluation with the env that we've set up.

### TL;DR
-	We run AndroidWorld in a Docker container.
-	We added an adapter android_world_env.py so ZeroGUI can call the env in its usual way.
-	You can smoke-test the interaction in notebooks/docker_exp.ipynb.

### 1. Project Layout
Move the files under `/online_eval` folder to the ZeroGUI repo cloned in Part I. Please move them to the designated place as shown:
<pre><code>repo-root/
  zerogui/
    openrlhf/
      env/
        __init__.py
        ...
        osworld_env.py
        android_lab_env.py
        android_world_env.py     # <-- our adapter (new)
    ...
    docker_exp.ipynb             # <-- end-to-end sanity test
</code></pre>
### 2. Build / Run the AndroidWorld Container
Build and run the Docker container following AndroidWorld's [🔗Docker setup](https://github.com/google-research/android_world?tab=readme-ov-file#docker-support-experimental) \
Once inside the container:
<pre><code>cd /workspace/zerogui
pip install -r requirements.txt (Please use Python >=3.10)</code></pre>
### 3. Smoke Test
Open and run the notebook `docker_exp.ipynb` \
You could play with the action by changing `action_payload`
<pre><code>action_payload = {"action_type": "open_app", "app_name": "Chrome"}
</code></pre>
And you should see the corresponding screenshots/empty reward of your interactions.

[(Jump back to Part A)](#a-training-a-process-reward-model-prm-for-gui-agent)
# B. Training an Internal "World Model" for GUI Agent

🎯 **Goal:** Build a world model that enables GUI agents to learn from both successful and failed trajectories through contrastive learning and error analysis.

📈 **Approach:** The world model generates text guidance injected into agent prompts, helping the agent avoid common mistakes and follow successful patterns.

## Overview

The world model provides two complementary capabilities:

1. **Contrastive Experience Learning** - Learn what distinguishes successful trajectories from failures
2. **Critical Error Detection** - Identify root causes of failures for targeted improvement

### Project Layout

`contrasive_experience/` contains the complete CoMEM-Agent inference system, including the agent, browser environment, memory systems, and evaluation benchmarks.

```
world_model/
├── contrasive_experience/              # Full CoMEM-Agent inference system
│   ├── run.py / run_chunks.py          # Entry points
│   ├── run_all_domains.sh              # Run all evaluation domains
│   ├── config/                         # CLI argument definitions
│   │
│   ├── agent/                          # FunctionCallAgent + LLM configs + prompts
│   ├── browser_env/                    # Playwright browser automation
│   ├── actions/                        # Action creation and parsing
│   ├── action_scaling/                 # Historical state-action matching
│   │
│   ├── memory/                         # Experience memory + reasoning bank
│   ├── arc_memo/                       # Concept memory (ArcMemo)
│   ├── graph_memory/                   # Graph-based memory
│   ├── hybrid_memory/                  # Hybrid memory system
│   ├── memory_evolution/               # Data flywheel pipeline
│   │
│   ├── world_model/                    # Contrastive learning world model
│   │   ├── world_model.py              # Main orchestrator
│   │   ├── trajectory_store.py         # FAISS-indexed trajectory storage
│   │   ├── trajectory_analyzer.py      # Contrastive pair analysis
│   │   ├── contrastive_summarizer.py   # Text summary generation
│   │   ├── state_predictor.py          # Action outcome prediction
│   │   └── prompts/                    # Prompt templates
│   │
│   ├── tools/                          # GUI tools, analysis tools, web search
│   ├── benchmarks/                     # MMInA, Mind2Web, WebVoyager evaluators
│   ├── scripts/                        # Runner and ablation scripts
│   ├── utils/                          # Shared utilities
│   ├── data_preparation/               # FAISS index building
│   │
│   ├── MMInA_evaluation/               # MMInA benchmark data
│   ├── Mind2Web_evaluation/            # Mind2Web benchmark data
│   ├── webvoyager_evaluation/          # WebVoyager benchmark data
│   └── compute_success_rate.py         # Evaluation utility
│
└── failure_traj_analysis/
    ├── world_model_supplement/         # Enhanced internal world model
    │   ├── internal_world_model.py     # Main orchestrator
    │   ├── action_ranker.py            # Success–failure margin reranking
    │   ├── contrastive_memory.py       # Dual success/failure retrieval
    │   ├── embedding_backend.py        # CLIP / latent embedding backends
    │   ├── contrastive_analyzer.py     # Contrastive pattern extraction
    │   ├── guidance_generator.py       # Step & initial guidance formatting
    │   ├── prompt_templates.py         # Prompt definitions
    │   ├── llm_utils.py                # LLM adapters
    │   ├── runtime_factory.py          # vLLM integration helpers
    │   ├── vllm_openai_client.py       # OpenAI-compatible client for vLLM
    │   └── utils.py                    # Shared utilities
    │
    ├── critical_error_detection.py # Root cause identification
    ├── experience_memory.py        # Memory retrieval system
    ├── fine_grained_analysis.py    # Detailed error analysis
    ├── evaluator.py                # Evaluation utilities
    └── early_stop.py               # Early stopping detection
```

## Part I — Contrastive Experience Learning

### Key Features

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

### Usage

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

### Data Format

Trajectory files should be in JSONL format:

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

Directory structure for training data:

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

## Part II — Critical Error Detection

### Goal

Identify the earliest critical error that led to task failure, categorized by module:

- **Memory errors**: Forgetting important information from previous steps
- **Reflection errors**: Incorrect self-assessment of progress
- **Planning errors**: Wrong strategy or approach
- **Action errors**: Wrong action execution
- **System errors**: Step limits, environment issues

### Usage

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

### Error Detection Pipeline

1. **Phase 1**: Fine-grained error detection at each step (memory, reflection, planning, action)
2. **Phase 2**: Critical error identification - find the earliest error that caused failure
3. **Output**: Structured analysis with root cause, evidence, and correction guidance

## Part III — Evaluation

Compute success rates from evaluation results:

```bash
cd world_model/contrasive_experience

# Compute success rate for a specific result directory
python compute_success_rate.py --result_dir results/mmina/shopping/qwen2.5-vl/2024-01-15

# Or find recent results with filters
python compute_success_rate.py --eval_type mmina --domain shopping --recent 5 --verbose
```

## Configuration

Key arguments for world model integration:

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_world_model` | Enable world model guidance | False |
| `--world_model_data_path` | Path to trajectory training data | - |
| `--world_model_index_path` | Path to pre-built FAISS index | - |
| `--world_model_top_k` | Number of trajectories to retrieve | 3 |
| `--world_model_multimodal` | Use multimodal embeddings | True |
| `--world_model_step_guidance` | Enable per-step guidance | True |

## Dependencies

```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install transformers torch numpy beautifulsoup4
```

## How It Works

### Contrastive Learning Pipeline

1. **Retrieval**: Given a new task and screenshot, retrieve top-k similar success AND failure trajectories using FAISS
2. **Analysis**: Compare action sequences, identify divergence points where success/failure paths differ
3. **Summarization**: Generate concise text guidance highlighting:
   - Success patterns to follow
   - Common mistakes to avoid
   - Key decision points
4. **Injection**: Insert guidance into agent prompts (system message + user message)

[(Jump back to Part B)](#b-training-an-internal-world-model-for-gui-agent)
