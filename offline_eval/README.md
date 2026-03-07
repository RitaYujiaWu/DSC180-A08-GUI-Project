# GUI Benchmark Runner

Run **UI-Vision**, **MMBench-GUI**, **Mind2Web**, and **AndroidControl** for a VLM (e.g. Qwen3-VL-4B) with optional LoRA.

**Location in repo**: `offline_eval/` (this folder). Config: `config.yaml`. Scripts: `scripts/run_all_benchmarks.sh`. Env: `environment.yml`.

## Requirements

- GPU with enough VRAM (e.g. 1× A100 80GB or similar for single-GPU)
- Conda

## Setup

1. **Clone dependencies** (same layout as used in evaluation):
   - [InfiGUI-G1](https://github.com/InfiGUI/InfiGUI-G1) (eval code + UI-Vision, MMBench-GUI data)
   - [UI-AGILE](https://github.com/UI-AGILE/UI-AGILE) (AndroidControl inference + data)
   - [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) (for Mind2Web eval `dom_utils`)

2. **Data**:
   - UI-Vision / MMBench-GUI: under `InfiGUI-G1/data/` as per their repo.
   - Mind2Web: download [scores](https://huggingface.co/datasets/osunlp/Mind2Web) and test split; put under `InfiGUI-G1/data/mind2web/` (e.g. `scores_all_data.pkl`, `test_website/`).
   - AndroidControl: parquet files under `UI-AGILE/android_control/`.

3. **Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate gui-benchmark
   ```
   Or use your existing env with: `vllm`, `transformers`, `qwen-vl-utils`, `lxml`, `datasets`, `pillow`, `pyyaml`.

4. **Config**:
   ```bash
   cp config.yaml my_config.yaml
   # Edit my_config.yaml: set base_model, infigui_root, uiagile_root, mind2web_repo
   export CONFIG_PATH=$(pwd)/my_config.yaml
   ```

## Run all four benchmarks for one model

```bash
cd scripts
chmod +x run_all_benchmarks.sh

# Baseline (no LoRA)
./run_all_benchmarks.sh baseline_v1

# With LoRA
./run_all_benchmarks.sh my_lora_v1 /path/to/ckpt_ep220
```

Results:

- **UI-Vision, MMBench-GUI, Mind2Web**: `InfiGUI-G1/output/<model_name>/<benchmark>/`
- **AndroidControl**: `benchmark_runner/results/android_control/<model_name>/high|low/`

## Run a single benchmark

From `InfiGUI-G1` (after activating env):

```bash
# UI-Vision
python eval/eval.py Qwen/Qwen3-VL-4B-Instruct --benchmark ui-vision --prompt strict-grounding --model-name my_model --batch-size 64

# MMBench-GUI
python eval/eval.py Qwen/Qwen3-VL-4B-Instruct --benchmark mmbench-gui --prompt strict-grounding --model-name my_model

# Mind2Web (set PYTHONPATH to Mind2Web/src)
python eval/eval_mind2web.py Qwen/Qwen3-VL-4B-Instruct --split test_website --data-dir ./data/mind2web --score-file ./data/mind2web/scores_all_data.pkl --model-name my_model
```

AndroidControl: run `inference_android_control.py` from `UI-AGILE/eval/android_control/` with `--model_path`, `--data_path`, `--output_path`, and optional `--lora_path`.

## Config options (config.yaml)

| Key | Description |
|-----|-------------|
| `base_model` | HuggingFace model id or local path |
| `lora_path` | Optional LoRA adapter dir |
| `infigui_root` | Path to InfiGUI-G1 repo |
| `uiagile_root` | Path to UI-AGILE repo |
| `mind2web_repo` | Path to Mind2Web repo (for `src/`) |
| `prompt` | `strict-grounding` or `infigui-g1` |
| `batch_size_uivision` | 64 recommended to avoid OOM with LoRA |

## Metrics

- **UI-Vision / MMBench-GUI**: use **scaled** accuracy (coordinates 0–1000 scaled to image size).
- **Mind2Web**: macro Element Acc, Action F1, Step SR (test_website).
- **AndroidControl**: Action Type Acc, Grounding Rate, Step Success Rate (high / low).

See project root `RESULTS_SUMMARY.md` for current result tables.
