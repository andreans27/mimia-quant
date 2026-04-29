#!/bin/bash
# Wrapper for auto_retrain that properly handles output redirection
cd /root/projects/mimia-quant
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
mkdir -p data/retrain_logs
source venv/bin/activate
python -u scripts/training/auto_retrain.py --symbol "$1" --force 2>&1 | tee "data/retrain_logs/${1}.log" | tail -1 >> "data/retrain_logs/_summary.txt"
