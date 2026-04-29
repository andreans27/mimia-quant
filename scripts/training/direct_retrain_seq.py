#!/usr/bin/env python3
"""Run retrain_symbol() from auto_retrain directly. No subprocess, no redirect issues."""
import sys, os, time, json, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

mod = importlib.import_module("scripts.training.auto_retrain")
LOG_DIR = ROOT / "data" / "retrain_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

symbols = sys.argv[1:] if len(sys.argv) > 1 else [
    'FETUSDT', 'TIAUSDT', 'WIFUSDT', 'DOGEUSDT', 'SOLUSDT',
    'SEIUSDT', '1000PEPEUSDT', 'ARBUSDT', 'INJUSDT', 'AVAXUSDT',
    'BNBUSDT', 'ETHUSDT', 'LINKUSDT', 'NEARUSDT', 'ADAUSDT',
    'AAVEUSDT', 'WLDUSDT',
]

for sym in symbols:
    print(f"\n{'='*60}")
    print(f"  RETRAIN: {sym}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    t0 = time.time()
    try:
        result = mod.retrain_symbol(sym, force=True)
        elapsed = time.time() - t0
        
        if result and result.get('status') == 'ok':
            val = result.get('validation', {})
            deployed = result.get('deployed', False)
            wr = val.get('wr', '?')
            pf = val.get('pf', '?')
            dep_str = '🚀 DEPLOYED' if deployed else '✓ saved'
            print(f"  ✅ {sym} {dep_str} | WR={wr} PF={pf} ({elapsed:.0f}s)")
        else:
            reason = result.get('reason', 'unknown') if result else 'no result'
            print(f"  ❌ {sym} failed: {reason} ({elapsed:.0f}s)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ❌ {sym} exception: {e} ({time.time()-t0:.0f}s)")
    
    sys.stdout.flush()

print(f"\n✅ All done!")
