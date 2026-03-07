#!/usr/bin/env bash
set -e

# Example runner for baseline vs hybrid
# Usage:
#   bash scripts/run_hybrid_eval.sh shopping amazon_index_dir

DOMAIN="${1:-shopping}"
INDEX_DIR="${2:-hybrid_index/${DOMAIN}}"

echo "[INFO] Building (if needed) index at: ${INDEX_DIR}"
# Example build command (uncomment and adjust glob as needed)
# python scripts/build_hybrid_index.py \
#   --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/**/success/*.jsonl" \
#   --output_dir "${INDEX_DIR}" \
#   --domain "${DOMAIN}"

echo "[INFO] Baseline run (no hybrid)"
python run.py --domain "${DOMAIN}" --max_steps 15

echo "[INFO] Hybrid run"
python run.py --domain "${DOMAIN}" --use_hybrid_memory --hybrid_index_dir "${INDEX_DIR}" --hybrid_k 3 --max_steps 15




