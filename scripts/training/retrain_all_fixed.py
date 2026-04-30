#!/usr/bin/env python3
"""Retrain ALL 20 LIVE_SYMBOLS sequentially with fixed features."""
import sys, os, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

from scripts.training.auto_retrain import retrain_symbol
from src.trading.state import LIVE_SYMBOLS

LOG_DIR = ROOT / "data" / "retrain_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"retrain_all_{time.strftime('%Y%m%d_%H%M%S')}.log"

results = []
for i, sym in enumerate(LIVE_SYMBOLS):
    print(f"\n{'='*60}")
    print(f"  [{i+1}/{len(LIVE_SYMBOLS)}] RETRAIN: {sym}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    t0 = time.time()
    try:
        result = retrain_symbol(sym, force=True)
        elapsed = time.time() - t0
        if result and result.get('status') == 'ok':
            n_models = result.get('n_models', 0)
            deployed = result.get('deployed', False)
            dep_str = '🚀 DEPLOYED' if deployed else '✓ saved'
            results.append((sym, 'OK', elapsed, deployed, n_models))
            print(f"  ✅ {sym} {dep_str} | {n_models} models ({elapsed:.0f}s)")
        else:
            reason = result.get('reason', 'unknown') if result else 'no result'
            results.append((sym, 'FAIL', elapsed, False, 0))
            print(f"  ❌ {sym} failed: {reason} ({elapsed:.0f}s)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append((sym, 'ERR', time.time()-t0, False, 0))
        print(f"  ❌ {sym} exception: {e} ({time.time()-t0:.0f}s)")
    
    sys.stdout.flush()

print(f"\n{'='*60}")
print(f"  RETRAIN COMPLETE")
print(f"{'='*60}")
print(f"  {sum(1 for r in results if r[1]=='OK')}/{len(results)} successful")
print(f"  Total time: {sum(r[2] for r in results):.0f}s")
for sym, status, elapsed, deployed, n in results:
    print(f"  {sym:>15s}: {status} {n} models {'🚀' if deployed else ''} [{elapsed:.0f}s]")

# Save log
with open(LOG_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Log saved: {LOG_FILE}")
