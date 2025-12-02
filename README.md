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

## Part III — Finetuning  (Based on Llama Factory's framework)
🎯 **Goal:** Fine-tune Qwen3VL-4b on the collected trajectory with reward. \
📈 **Next Step:** Keep testing and refining so the data and fine-tuning works best.

To fine-tune, `git clone` Llama Factory's [repo](https://github.com/hiyouga/LLaMA-Factory), follow the setup instructions,
and finally run:
<pre><code> python YOUR_PATH/infer_qwen3vl_baseline.py \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --trust_remote_code \
  --input_jsonl YOUR_PATH/o3_15steps_llamafactory_sft.jsonl \
  --output_jsonl YOUR_PATH/o3_15steps_infer_qwen3vl.jsonl \
  --max_images 4 \
  --max_samples 64 \
  --max_new_tokens 256 \
  --temperature 0.2</code></pre>

## Part IV — Online Data Evaluation (Based on ZeroGUI's framework)
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
...
