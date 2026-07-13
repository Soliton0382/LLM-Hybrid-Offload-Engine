#!/usr/bin/env bash
# =============================================================================
#  run_benchmark.sh -- one command, correct flags, no surprises.
#
#  SSCC Phase A only (fast, numpy, NO model):   ./run_benchmark.sh
#  SSCC Phase A + Phase B (real GGUF answer):    ./run_benchmark.sh --model
#  Hybrid throughput (CPU vs SMT vs GPU offload):./run_benchmark.sh --hybrid
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

case "${1:-}" in
  --hybrid)
    echo "[*] Hybrid throughput benchmark (3-mode automated comparison)..."
    python3 bench/hybrid_offload_benchmark.py --auto
    ;;
  --model)
    echo "[*] SSCC Phase A + Phase B (loading the real GGUF model)..."
    RUN_MODEL_EVAL=1 MODEL_TRIALS="${MODEL_TRIALS:-10}" TRIALS="${TRIALS:-500}" \
      NO_COLOR="${NO_COLOR:-0}" python3 -u bench/sscc_benchmark.py
    ;;
  *)
    echo "[*] SSCC Phase A only (numpy retention, no model)."
    echo "    Use --model for the full SSCC run, or --hybrid for throughput."
    TRIALS="${TRIALS:-500}" python3 bench/sscc_benchmark.py
    ;;
esac
