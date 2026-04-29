#!/usr/bin/env python3
"""
Direct Retrain — calls auto_retrain.retrain_symbol() directly.
No subprocess, no pipe issues, no OMP env problems.
3 symbols parallel via multiprocessing.
"""
import sys, os, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

# Set OMP BEFORE importing anything (affects XGBoost)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import multiprocessing as mp
from datetime import datetime

# Import retrain function
sys.path.insert(0, str(ROOT / "scripts/training"))
import importlib
auto_retrain_mod = importlib.import_module("scripts.training.auto_retrain")

ALL_SYMBOLS = [
    'ENAUSDT', 'SUIUSDT', 'OPUSDT', 'FETUSDT', 'TIAUSDT',
    'WIFUSDT', 'DOGEUSDT', 'SOLUSDT', 'SEIUSDT',
    '1000PEPEUSDT', 'ARBUSDT', 'INJUSDT', 'AVAXUSDT', 'BNBUSDT',
    'ETHUSDT', 'LINKUSDT', 'NEARUSDT', 'ADAUSDT', 'AAVEUSDT', 'WLDUSDT',
]

LOG_DIR = Path("data/retrain_direct_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def retrain_one(symbol: str) -> dict:
    """Train + validate one symbol. Returns result dict."""
    start = time.time()
    log_file = LOG_DIR / f"{symbol}.log"
    
    # Redirect stdout to log file
    old_stdout = sys.stdout
    f = open(log_file, 'w')
    sys.stdout = f
    
    try:
        print(f"=== RETRAIN {symbol} ===")
        print(f"Start: {datetime.utcnow().isoformat()}\n")
        
        result = auto_retrain_mod.retrain_symbol(symbol, force=True)
        
        elapsed = time.time() - start
        print(f"\nElapsed: {elapsed:.0f}s")
        
        success = result and result.get('status') == 'ok'
        deployed = result and result.get('deployed', False)
        
        if result:
            val = result.get('validation', {})
            print(f"Result: {'✅' if success else '❌'} | Deployed: {deployed}")
            print(f"WR={val.get('wr','?'):s} PF={val.get('pf','?'):s} DD={val.get('dd','?'):s}")
        
        return {
            'symbol': symbol, 'elapsed': elapsed, 'success': success,
            'deployed': deployed,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed = time.time() - start
        return {'symbol': symbol, 'elapsed': elapsed, 'success': False, 'error': str(e)}
    finally:
        sys.stdout = old_stdout
        f.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help='Comma-separated')
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')] if args.symbols else ALL_SYMBOLS
    parallel = min(args.parallel, len(symbols))
    
    # Remove ENAUSDT if already done
    done_file = LOG_DIR / "_done_symbols.json"
    if done_file.exists():
        done = json.loads(done_file.read_text())
        symbols = [s for s in symbols if s not in done]
    
    print(f"{'='*60}")
    print(f"  DIRECT BATCH RETRAIN")
    print(f"  Symbols: {len(symbols)} | Parallel: {parallel}")
    print(f"{'='*60}")
    
    total_start = time.time()
    results = []
    
    # Process in parallel batches
    for batch_start in range(0, len(symbols), parallel):
        batch = symbols[batch_start:batch_start + parallel]
        print(f"\nBatch {batch_start//parallel + 1}: {', '.join(batch)}")
        
        with mp.Pool(processes=len(batch)) as pool:
            batch_results = pool.map(retrain_one, batch)
        
        for r in batch_results:
            sym = r['symbol']
            elapsed = r['elapsed']
            if r['success']:
                dep = '🚀' if r.get('deployed') else '✓'
                print(f"  {dep} {sym} done ({elapsed:.0f}s)")
            else:
                print(f"  ❌ {sym} FAILED ({elapsed:.0f}s): {r.get('error', '?')}")
            
            results.append(r)
        
        # Save progress
        done = list(set(r['symbol'] for r in results))
        done_file.write_text(json.dumps(done))
    
    total_elapsed = time.time() - total_start
    successes = sum(1 for r in results if r['success'])
    failures = sum(1 for r in results if not r['success'])
    deployed = sum(1 for r in results if r.get('deployed'))
    
    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  ✅ {successes} success | 🚀 {deployed} deployed | ❌ {failures} failed")
    print(f"  Wall time: {total_elapsed:.0f}s ({total_elapsed/60:.0f}m)")
    
    result_file = LOG_DIR / "_results.json"
    result_file.write_text(json.dumps({
        'timestamp': datetime.utcnow().isoformat(),
        'total': len(results), 'success': successes,
        'failed': failures, 'deployed': deployed,
        'results': results,
    }, indent=2))
    print(f"  Logs: {LOG_DIR}")


if __name__ == '__main__':
    # Required for multiprocessing on Linux
    mp.set_start_method('spawn', force=True)
    main()
