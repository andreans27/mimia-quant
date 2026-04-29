#!/usr/bin/env python3
"""
Parallel Batch Retrain v2 — file-based stdout (no pipe buffer issues).
3 processes parallel, stdout langsung ke file.
"""
import sys, os, time, subprocess, json, argparse, signal
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

ALL_SYMBOLS = [
    'ENAUSDT', 'SUIUSDT', 'OPUSDT', 'FETUSDT', 'TIAUSDT',
    'WIFUSDT', 'DOGEUSDT', 'SOLUSDT', 'SEIUSDT',
    '1000PEPEUSDT', 'ARBUSDT', 'INJUSDT', 'AVAXUSDT', 'BNBUSDT',
    'ETHUSDT', 'LINKUSDT', 'NEARUSDT', 'ADAUSDT', 'AAVEUSDT', 'WLDUSDT',
]

LOG_DIR = Path("data/retrain_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_batch(symbols: list, parallel: int = 3, timeout: int = 1800) -> list:
    """Run retrain for symbols in parallel batches."""
    results = []
    
    for batch_start in range(0, len(symbols), parallel):
        batch = symbols[batch_start:batch_start + parallel]
        print(f"\nBatch {batch_start//parallel + 1}: {', '.join(batch)}")
        
        processes = []
        for sym in batch:
            log_file = str(LOG_DIR / f"{sym}.log")
            print(f"  Starting {sym} → {log_file}")
            
            # Open log file and redirect stdout/stderr directly
            f = open(log_file, 'w')
            f.write(f"=== RETRAIN {sym} ===\n")
            f.write(f"Start: {datetime.utcnow().isoformat()}\n\n")
            f.flush()
            
            p = subprocess.Popen(
                [sys.executable, "-u", "scripts/training/auto_retrain.py", "--symbol", sym, "--force"],
                stdout=f, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '2'},
            )
            processes.append({'sym': sym, 'proc': p, 'file': f, 'start': time.time()})
        
        # Wait for all in batch
        for p_info in processes:
            sym = p_info['sym']
            proc = p_info['proc']
            f = p_info['file']
            start = p_info['start']
            
            try:
                proc.wait(timeout=timeout)
                elapsed = time.time() - start
                
                f.write(f"\nElapsed: {elapsed:.0f}s | RC={proc.returncode}\n")
                f.close()
                
                status = '✅' if proc.returncode == 0 else '❌'
                print(f"  {status} {sym} RC={proc.returncode} ({elapsed:.0f}s)")
                
                results.append({
                    'symbol': sym, 'elapsed': elapsed, 'rc': proc.returncode,
                    'success': proc.returncode == 0,
                })
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                elapsed = time.time() - start
                f.write(f"\nTIMEOUT ({timeout}s) — KILLED\n")
                f.close()
                print(f"  ⚠️ {sym} TIMEOUT ({elapsed:.0f}s)")
                results.append({
                    'symbol': sym, 'elapsed': elapsed, 'rc': -1,
                    'success': False, 'error': 'timeout',
                })
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help='Comma-separated')
    parser.add_argument('--parallel', type=int, default=3)
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')] if args.symbols else ALL_SYMBOLS
    
    print(f"{'='*60}")
    print(f"  PARALLEL BATCH RETRAIN v2")
    print(f"  {len(symbols)} symbols × {args.parallel} parallel")
    print(f"{'='*60}")
    
    total_start = time.time()
    results = run_batch(symbols, parallel=args.parallel)
    total_elapsed = time.time() - total_start
    
    successes = sum(1 for r in results if r['success'])
    failures = sum(1 for r in results if not r['success'])
    
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {successes}✅ / {failures}❌")
    print(f"  Wall: {total_elapsed:.0f}s ({total_elapsed/60:.0f}m)")
    if successes:
        avg = sum(r['elapsed'] for r in results if r['success']) / successes
        print(f"  Avg/sym: {avg:.0f}s")
    
    # Save results
    (LOG_DIR / "_batch_results.json").write_text(json.dumps({
        'timestamp': datetime.utcnow().isoformat(),
        'total': len(results), 'success': successes, 'failed': failures,
        'results': results,
    }, indent=2))
    
    print(f"  Results: {LOG_DIR / '_batch_results.json'}")


if __name__ == '__main__':
    main()
