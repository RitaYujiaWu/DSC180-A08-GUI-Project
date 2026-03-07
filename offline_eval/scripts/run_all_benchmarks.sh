#!/bin/bash
# Run all four benchmarks for one model (optionally with LoRA).
# Usage: ./run_all_benchmarks.sh <model_name> [lora_path]
# Example: ./run_all_benchmarks.sh my_model_v1 /path/to/ckpt_ep220
# Requires: set env CONFIG_PATH or run from benchmark_runner with config.yaml

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$RUNNER_ROOT/config.yaml}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH"
  echo "Copy config.yaml and set paths. Or set CONFIG_PATH=..."
  exit 1
fi

MODEL_NAME="${1:?Usage: run_all_benchmarks.sh <model_name> [lora_path]}"
LORA_PATH="${2:-}"

# Env overrides (for CI/lab): INFIGUI_ROOT, UIAGILE_ROOT, MIND2WEB_REPO, BASE_MODEL
get_config() {
  grep -E "^\s*${1}:" "$CONFIG_PATH" 2>/dev/null | sed -E 's/^[^:]+:[[:space:]]*["]?([^"]*)["]?.*/\1/' | sed "s/^[[:space:]]*'[^']*'[[:space:]]*//" | head -1
}
BASE_MODEL="${BASE_MODEL:-$(get_config base_model)}"
INFIGUI_ROOT="${INFIGUI_ROOT:-$(get_config infigui_root)}"
UIAGILE_ROOT="${UIAGILE_ROOT:-$(get_config uiagile_root)}"
MIND2WEB_REPO="${MIND2WEB_REPO:-$(get_config mind2web_repo)}"
PROMPT="${PROMPT:-$(get_config prompt)}"
BATCH_UI="${BATCH_UI:-$(get_config batch_size_uivision)}"
BATCH_MM="${BATCH_MM:-$(get_config batch_size_mmbench)}"
TP="${TP:-$(get_config tensor_parallel)}"
TEMP="${TEMP:-$(get_config temperature)}"
THINK="${THINK:-$(get_config think_mode)}"
AC_OUT="${AC_OUT:-$(get_config ac_output_root)}"

[ -z "$INFIGUI_ROOT" ] || [ "$INFIGUI_ROOT" = "/path/to/InfiGUI-G1" ] && { echo "Set INFIGUI_ROOT or infigui_root in config.yaml"; exit 1; }
[ -z "$UIAGILE_ROOT" ] || [ "$UIAGILE_ROOT" = "/path/to/UI-AGILE" ] && { echo "Set UIAGILE_ROOT or uiagile_root in config.yaml"; exit 1; }

AC_DIR="${UIAGILE_ROOT}/eval/android_control"
AC_HIGH="${UIAGILE_ROOT}/android_control/androidcontrol_high_test.parquet"
AC_LOW="${UIAGILE_ROOT}/android_control/androidcontrol_low_test.parquet"
MIND2WEB_DATA="${INFIGUI_ROOT}/data/mind2web"
MIND2WEB_SCORES="${MIND2WEB_DATA}/scores_all_data.pkl"
AC_OUT="${AC_OUT:-$RUNNER_ROOT/results/android_control}"
mkdir -p "$AC_OUT"

LORA_ARG=""
LORA_ARG_AC=""
[ -n "$LORA_PATH" ] && LORA_ARG="--lora-path $LORA_PATH" && LORA_ARG_AC="--lora_path $LORA_PATH"

echo "=========================================="
echo " Model: $MODEL_NAME"
echo " Base:  $BASE_MODEL"
echo " LoRA:  ${LORA_PATH:-none}"
echo "=========================================="

# 1) UI-Vision
echo ""; echo ">>>>>> [${MODEL_NAME}] ui-vision <<<<<<"
cd "$INFIGUI_ROOT"
python eval/eval.py "$BASE_MODEL" \
  --benchmark ui-vision \
  --prompt "${PROMPT:-strict-grounding}" \
  --tensor-parallel "${TP:-1}" \
  --model-name "$MODEL_NAME" \
  --temperature "${TEMP:-0.0}" \
  --batch-size "${BATCH_UI:-64}" \
  --max-num-seqs 8 \
  --think-mode "${THINK:-1}" \
  $LORA_ARG || echo "[WARN] ui-vision failed"

# 2) MMBench-GUI
echo ""; echo ">>>>>> [${MODEL_NAME}] mmbench-gui <<<<<<"
cd "$INFIGUI_ROOT"
python eval/eval.py "$BASE_MODEL" \
  --benchmark mmbench-gui \
  --prompt "${PROMPT:-strict-grounding}" \
  --tensor-parallel "${TP:-1}" \
  --model-name "$MODEL_NAME" \
  --temperature "${TEMP:-0.0}" \
  --batch-size "${BATCH_MM:-128}" \
  --think-mode "${THINK:-1}" \
  $LORA_ARG || echo "[WARN] mmbench-gui failed"

# 3) Mind2Web (needs Mind2Web repo for dom_utils)
echo ""; echo ">>>>>> [${MODEL_NAME}] mind2web <<<<<<"
cd "$INFIGUI_ROOT"
export PYTHONPATH="${MIND2WEB_REPO:-$INFIGUI_ROOT/../Mind2Web}/src:${PYTHONPATH:-}"
python eval/eval_mind2web.py "$BASE_MODEL" \
  --split test_website \
  --data-dir "$MIND2WEB_DATA" \
  --score-file "$MIND2WEB_SCORES" \
  --top-k 10 \
  --tensor-parallel "${TP:-1}" \
  --batch-size 64 \
  --max-tokens 512 \
  --think-mode "${THINK:-1}" \
  --temperature "${TEMP:-0.0}" \
  --model-name "$MODEL_NAME" \
  $LORA_ARG || echo "[WARN] mind2web failed"

# 4) AndroidControl High + Low
echo ""; echo ">>>>>> [${MODEL_NAME}] androidcontrol-high <<<<<<"
cd "$AC_DIR"
mkdir -p "$AC_OUT/$MODEL_NAME"
python inference_android_control.py \
  --model_path "$BASE_MODEL" \
  --prompt_template android_control_detailed \
  --data_path "$AC_HIGH" \
  --output_path "$AC_OUT/$MODEL_NAME/high" \
  $LORA_ARG_AC || echo "[WARN] ac-high failed"

echo ""; echo ">>>>>> [${MODEL_NAME}] androidcontrol-low <<<<<<"
python inference_android_control.py \
  --model_path "$BASE_MODEL" \
  --prompt_template android_control_detailed \
  --data_path "$AC_LOW" \
  --output_path "$AC_OUT/$MODEL_NAME/low" \
  $LORA_ARG_AC || echo "[WARN] ac-low failed"

echo ""; echo "[DONE] $MODEL_NAME at $(date)"
