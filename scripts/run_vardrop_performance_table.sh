#!/usr/bin/env bash
set -euo pipefail

cd /home/yulim/MSH/patch-latent/iTransformer
mkdir -p results logs

PYTHON=${PYTHON:-/home/yulim/anaconda3/envs/patchtst/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-2021}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-10}
PATIENCE=${PATIENCE:-3}
NUM_WORKERS=${NUM_WORKERS:-0}
RESULT_CSV=${RESULT_CSV:-./results/coverage_fallback_test_for_vardrop.csv}
LOG=${LOG:-./logs/vardrop_performance_table.queue.log}

{
  echo "start $(date '+%Y-%m-%d %H:%M:%S %Z') gpu=${GPU} seeds=${SEEDS} result=${RESULT_CSV}"
  "${PYTHON}" - <<'PY'
import sys
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("CUDA is not available in this shell; refusing to run test training on CPU.", file=sys.stderr)
    raise SystemExit(20)
print("device_name", torch.cuda.get_device_name(0))
PY

  "${PYTHON}" -u scripts/run_coverage_fallback_grid.py \
    --run \
    --result_csv "${RESULT_CSV}" \
    --pred_lens 96 192 336 720 \
    --seeds ${SEEDS} \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --num_workers "${NUM_WORKERS}" \
    --gpu "${GPU}"

  "${PYTHON}" -u scripts/build_vardrop_comparison_report.py
  echo "done $(date '+%Y-%m-%d %H:%M:%S %Z')"
} 2>&1 | tee -a "${LOG}"
