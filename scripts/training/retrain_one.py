#!/usr/bin/env python3
"""Run auto_retrain for one symbol, saving output to file and stdout."""
import sys, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(ROOT))

symbol = sys.argv[1]
log_file = ROOT / "data" / "retrain_logs" / f"{symbol}.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

env = {**os.environ, 'OMP_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}

with open(log_file, 'w') as f:
    f.write(f"=== RETRAIN {symbol} ===\n")
    f.write(f"Start: {__import__('datetime').datetime.utcnow().isoformat()}\n\n")
    f.flush()
    
    p = subprocess.Popen(
        [sys.executable, '-u', str(ROOT / "scripts/training/auto_retrain.py"),
         '--symbol', symbol, '--force'],
        stdout=f, stderr=subprocess.STDOUT, env=env,
    )
    p.wait()

print(f"RETRAIN {symbol} DONE (RC={p.returncode})", flush=True)
