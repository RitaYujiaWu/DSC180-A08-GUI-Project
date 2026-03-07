#!/usr/bin/env bash
set -e

# Simple ablation over k (top-K exemplars)
# Usage:
#   bash scripts/run_hybrid_ablate.sh shopping hybrid_index/shopping

DOMAIN="${1:-shopping}"
INDEX_DIR="${2:-hybrid_index/${DOMAIN}}"

for K in 1 2 3 4 5; do
  echo "[INFO] Hybrid ablation: k=${K}"
  python run.py --domain "${DOMAIN}" --use_hybrid_memory --hybrid_index_dir "${INDEX_DIR}" --hybrid_k "${K}" --max_steps 15
done




