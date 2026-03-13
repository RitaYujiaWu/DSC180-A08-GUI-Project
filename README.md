<h1 align="center">Inference-Time Scaling for GUI Agents <br> with Process Reward Models and Internal World Models</h1>
<p align="center">
  <strong>DSC180 A08-2 Capstone Project</strong> · 
  <a href="https://ritayujiawu.github.io/DSC180-A08-GUI-Project/"><strong>🖥️ Website</strong></a>· 
  <a href="https://ritayujiawu.github.io/DSC180-A08-GUI-Project/asset/Q2_Project_Report.pdf"><strong>📝 Report</strong></a> · 
  <a href="https://ritayujiawu.github.io/DSC180-A08-GUI-Project/asset/A08_2_Poster.pdf"><strong>🤖 Poster</strong></a>
</p>
Right now, there’s a lot of excitement around GUI agents, which are AI agents that interact with graphical user interfaces—things like apps, browsers, and operating systems. Therefore, we chose this as our topic and focused on improving the GUI agent performance through inference-time scaling.

Our team is then split into two groups that strengthen the agent from complementary directions: one group trains a Process Reward Model (PRM) as an external signal [Jump to Part A](#a-training-a-process-reward-model-prm-for-gui-agent), while the other develops an internal world model to enhance the agent’s own reasoning [Jump to Part B](#b-training-an-internal-world-model-for-gui-agent).

# A. Training a Process Reward Model (PRM) for GUI Agent

## Install the Dependencies
<pre><code>conda create -n prm python=3.10 -y
conda activate prm
pip install -r requirements_prm.txt</code></pre>

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

## Part V — Online Data Evaluation
🎯 **Goal:** Evaluate the base model and LoRA-trained agent on two online benchmarks (Android World, Android Lab) so we can compare before/after training. This part of evaluation includes interacting with virtual environments. \

### 1. Clone Repo/Pull Data from Official Benchmark Providers.
First, `git clone` two benchmarks: [Android World](https://github.com/google-research/android_world) and [Android Lab](https://github.com/THUDM/Android-Lab) and follow all their setups\
Move the `config.yaml` file under `/online_eval` folder to the root of the Android Lab repo cloned. 
### 2. Build / Run the Docker Container
Build and run the Docker container for **Android World** following: [🔗world docker setup](https://github.com/google-research/android_world?tab=readme-ov-file#docker-support-experimental) \
Build and run the Docker container for **Android Lab** following: [🔗lab docker setup](https://github.com/THUDM/Android-Lab/blob/main/docs/prepare_for_linux.md)
### 3. Locally Host Trained Agent for Evaluation
First, build the VLLM Conda environment compatible with your agent model: 
<pre><code>conda env create --prefix YOUR_ROOT/DSC180-A08-GUI-Project/online_eval -f vllm_environment.yml
</code></pre>
Then start the VLLM engine (example below, adjust based on your condition):
<pre><code>CUDA_VISIBLE_DEVICES=7 vllm serve Qwen/Qwen3-VL-4B-Instruct --trust-remote-code --port 8000 --gpu_memory_utilization 0.75 </code></pre>
### 4. Start the Evaluation
Start Android World evaluation by (make sure the port can point to your hosted model): 
<pre><code>cd YOUR_ROOT/android_world
docker run --rm --privileged \
  --network host -it -v "$PWD/runs:/runs" \
  -e OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  -e OPENAI_MODEL=Qwen/Qwen3-VL-4B-Instruct android-world-fixed:latest python run.py \
  --output_path=/runs --suite_family=android_world --agent_name=m3a_gpt4v \
  --openai_base_url http://127.0.0.1:8000/v1 --openai_model Qwen/Qwen3-VL-4B-Instruct --n_task_combinations=1
</code></pre>
Start Android Lab evaluation by:
<pre><code>cd YOUR_ROOT/Android-Lab
conda run -n Android-Lab python eval.py -n qwen3vl4b_run_fix_launch2 -c config.yaml </code></pre>

## Part VI — Offline Benchmark Evaluation (UI-Vision, MMBench-GUI, Mind2Web, AndroidControl)

🎯 **Goal:** Evaluate the base model and LoRA-trained agent on four offline benchmarks (UI-Vision, MMBench-GUI, Mind2Web, AndroidControl) so we can compare before/after training. \
📈 **Deliverable:** Scripts, config, and environment in <code>benchmark_runner/</code>; data from HuggingFace, paths under <code>PATH_TO/data/...</code>.

### 1. Project layout

Eval scripts and config live in <code>benchmark_runner/</code>. Put all benchmark data under one root, e.g. <code>PATH_TO/data/</code>:

<pre><code>repo-root/
  benchmark_runner/
    config.yaml
    environment.yml
    scripts/
      run_all_benchmarks.sh
    README.md

PATH_TO/data/
  ui-vision/                      # annotations/element_grounding/, images/
  MMBench-GUI/                    # L2_annotations.json, MMBench-GUI-OfflineImages/offline_images/
  mind2web/                       # scores_all_data.pkl, test_website/, test_task/, test_domain/
  AndroidControl/                 # androidcontrol_high_test.parquet, androidcontrol_low_test.parquet
</code></pre>

### 2. Data (download from HuggingFace)

| Benchmark | Download | Where to put |
|-----------|----------|----------------|
| **UI-Vision** | [ServiceNow/ui-vision](https://huggingface.co/datasets/ServiceNow/ui-vision) | <code>PATH_TO/data/ui-vision/</code> — <code>annotations/element_grounding/</code>, <code>images/</code> |
| **MMBench-GUI** | [OpenGVLab/MMBench-GUI](https://huggingface.co/datasets/OpenGVLab/MMBench-GUI) — L2_annotations.json, MMBench-GUI-OfflineImages | <code>PATH_TO/data/MMBench-GUI/</code> — <code>L2_annotations.json</code>, <code>MMBench-GUI-OfflineImages/offline_images/</code> |
| **Mind2Web** | [osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) — scores_all_data.pkl, test.zip (password: <code>mind2web</code>) | <code>PATH_TO/data/mind2web/</code> — <code>scores_all_data.pkl</code>, unzipped <code>test_website/</code>, <code>test_task/</code>, <code>test_domain/</code> |
| **AndroidControl** | [smolagents/android-control](https://huggingface.co/datasets/smolagents/android-control) (parquet) | <code>PATH_TO/data/AndroidControl/</code> — <code>androidcontrol_high_test.parquet</code>, <code>androidcontrol_low_test.parquet</code> |

### 3. Environment

<pre><code>cd benchmark_runner
conda env create -f environment.yml
conda activate gui-benchmark
</code></pre>

Or install manually: <code>vllm</code>, <code>transformers</code>, <code>qwen-vl-utils</code>, <code>lxml</code>, <code>datasets</code>, <code>pillow</code>, <code>pyyaml</code>, <code>torch</code>.

### 4. Config

Copy and edit <code>config.yaml</code>; set <code>base_model</code>, data paths, and optional LoRA:

<pre><code>cp config.yaml my_config.yaml
# Edit: base_model, data paths (PATH_TO/data/...), ac_data_high, ac_data_low, lora_path
export CONFIG_PATH=$(pwd)/my_config.yaml
</code></pre>

| Key | Description |
|-----|-------------|
| <code>base_model</code> | HuggingFace id (e.g. <code>Qwen/Qwen3-VL-4B-Instruct</code>) or <code>PATH_TO/models/YourModel</code> |
| <code>data_root</code> | <code>PATH_TO/data</code> (base for ui-vision, MMBench-GUI, mind2web, AndroidControl) |
| <code>ac_data_high</code> / <code>ac_data_low</code> | <code>PATH_TO/data/AndroidControl/androidcontrol_high_test.parquet</code> (and <code>_low</code>) |
| <code>lora_path</code> | Optional: <code>PATH_TO/adapters/ckpt_ep220</code> |

### 5. Run all four benchmarks

<pre><code>cd benchmark_runner/scripts
chmod +x run_all_benchmarks.sh

# Baseline (no LoRA)
./run_all_benchmarks.sh baseline_v1

# With LoRA
./run_all_benchmarks.sh my_lora_v1 PATH_TO/adapters/ckpt_ep220
</code></pre>

Results: under <code>benchmark_runner/results/</code> (or paths set in config).

### 6. Metrics

- **UI-Vision / MMBench-GUI:** scaled accuracy (0–1000 coordinates scaled to image size).
- **Mind2Web:** macro Element Acc, Action F1, Step SR (test_website).
- **AndroidControl:** Action Type Acc, Grounding Rate, Step Success Rate (high / low).

See project root <code>RESULTS_SUMMARY.md</code> for result tables.

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
    ├── internal_world_model/           # Supplement based on the contrasive_experience folder
    │   ├── planner.py                  # Orchestrates planning and action selection
    │   ├── schemas.py                  # Shared data structures for planner
    │   ├── contrastive_retriever.py    # Retrieves contrastive success/failure trajectories
    │   ├── candidate_generator.py      # Generates candidate next actions
    │   ├── candidate_scorer.py         # Scores candidates using contrastive evidence
    │   └── prompt_builder.py           # Builds prompts for planning modules
    │
    ├── critical_error_detection.py     # Root cause identification
    ├── experience_memory.py            # Memory retrieval system
    ├── fine_grained_analysis.py        # Detailed error analysis
    ├── evaluator.py                    # Evaluation utilities
    └── early_stop.py                   # Early stopping detection
```

## 🚀 Quick Start

This guide shows how to set up the environment, start the model servers, and run evaluations on the GUI benchmarks (**MMInA**, **Mind2Web**, and **WebVoyager**).

---

## 1️⃣ Docker Setup (Recommended)

Docker provides an isolated environment with all dependencies pre-installed, including Playwright, FAISS, and system libraries.

```bash
# Build the container
./docker-run.sh build

# Start the container
./docker-run.sh start

# Enter the container
./docker-run.sh shell
```

Once inside the container, all dependencies are already installed and ready to use.

---

## 2️⃣ Local Setup (Optional)

If you prefer running without Docker, install dependencies manually:

```bash
pip install -r requirements_web.txt
playwright install
```

The system depends on:

- **Playwright** — browser automation
- **FAISS** — trajectory retrieval
- **CLIP / Transformers** — embedding models
- **vLLM** — fast LLM inference server

---

## 3️⃣ Start vLLM Servers

The agent requires two model servers:

- **Main Agent Model** (Qwen2.5-VL)
- **Grounding Model** (UI-Ins)

Start them in **two separate terminals**.

```bash
# Terminal 1 — Main Agent Model
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8010 \
    --tensor-parallel-size 2 \
    --max-model-len 81920 \
    --gpu-memory-utilization 0.9

# Terminal 2 — Grounding Model
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
    --model Tongyi-MiA/UI-Ins-7B \
    --port 8011 \
    --tensor-parallel-size 2 \
    --max-model-len 81920 \
    --gpu-memory-utilization 0.9
```

Verify that the servers are running:

```bash
curl http://localhost:8010/v1/models
curl http://localhost:8011/v1/models
```

These servers must remain running while experiments are executed.

---

## 4️⃣ Run Evaluations

Navigate to the inference directory:

```bash
cd world_model/contrasive_experience
```

### Example: Run a Single Domain

```bash
python run.py \
  --model qwen2.5-vl \
  --eval_type mmina \
  --domain shopping \
  --use_world_model \
  --use_memory
```

### Run All Domains

```bash
bash run_all_domains.sh
```

### Parallel Evaluation

```bash
python run_chunks.py --chunk_id 0 --num_chunks 4
```

---

## 5️⃣ Supported Benchmarks

The system evaluates GUI agents on multiple real-world benchmarks.

### MMInA
- **Shopping** (200 tasks): E-commerce interaction
- **Wikipedia** (308 tasks): Information retrieval

### Mind2Web
Cross-website task execution:
- `test_website` — general websites  
- `test_domain_Info` — information domains  
- `test_domain_Service` — service domains  

### WebVoyager
Multi-domain web navigation across **15+ real websites**, including:

- **E-commerce**: Amazon, Apple  
- **Information**: ArXiv, Wikipedia, BBC News  
- **Services**: Booking, GitHub, Google Maps  

Dataset files are available from **[Hugging Face](https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories)**.

---

## 6️⃣ Compute Success Rates

After running evaluations:

```bash
python compute_success_rate.py \
  --result_dir results/mmina/shopping/qwen2.5-vl/<run_id>
```

Or automatically detect recent runs:

```bash
python compute_success_rate.py \
  --eval_type mmina \
  --domain shopping \
  --recent 5 \
  --verbose
```

---

After completing these steps, you are ready to run GUI agent experiments with **experience memory** and **contrastive world models**.


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

---

## BibTeX Citation

If you use this work, please cite:

```bibtex
@article{wang2025inference,
  title     = {Inference-Time Scaling for GUI Agents with Process Reward
               Models and Internal World Models},
  author    = {Wang, Bella and Wu, Rita Yujia and Liu, Shuchang and
               Huang, Ziyu},
  year      = {2025},
  url       = {https://github.com/RitaYujiaWu/DSC180-A08-GUI-Project},
  note      = {DSC180 A08 Capstone. Mentors: Kun Zhou, Zhiting Hu.},
}
```
