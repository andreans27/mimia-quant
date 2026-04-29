#!/usr/bin/env python3
"""
Parallel Batch Retrain — runs auto_retrain for multiple symbols in parallel.
3 concurrent processes to maximize CPU utilization without overload.

Usage:
  python scripts/training/batch_retrain.py                           # all symbols
  python scripts/training/batch_retrain.py --symbols ENAUSDT,SOLUSDT # specific
  python scripts/training/batch_retrain.py --parallel 4              # 4 at a time
"""
import sys, os, time, subprocess, json, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

# Symbols that need retraining (all LIVE_SYMBOLS)
ALL_SYMBOLS = [
    'ENAUSDT', 'SUIUSDT', 'OPUSDT', 'FETUSDT', 'TIAUSDT',
    'WIFUSDT', 'DOGEUSDT', 'SOLUSDT', 'SEIUSDT',
    '1000PEPEUSDT', 'ARBUSDT', 'INJUSDT', 'AVAXUSDT', 'BNBUSDT',
    'ETHUSDT', 'LINKUSDT', 'NEARUSDT', 'ADAUSDT', 'AAVEUSDT', 'WLDUSDT',
]

LOG_DIR = Path("data/retrain_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_symbol(sym: str, timeout: int = 1800) -> dict:
    """Run auto_retrain for one symbol. Returns result dict."""
    log_file = LOG_DIR / f"{sym}.log"
    start = time.time()
    
    with open(log_file, 'w') as f:
        f.write(f"=== RETRAIN {sym} ===\n")
        f.write(f"Start: {datetime.utcnow().isoformat()}\n\n")
    
    result = subprocess.run(
        [sys.executable, "-u", "scripts/training/auto_retrain.py", "--symbol", sym, "--force"],
        capture_output=True, text=True, timeout=timeout,
    )
    
    elapsed = time.time() - start
    
    # Write full output to log
    with open(log_file, 'a') as f:
        f.write(f"\n== STDOUT ==\n{result.stdout}\n")
        if result.stderr:
            f.write(f"\n== STDERR ==\n{result.stderr}\n")
        f.write(f"\nElapsed: {elapsed:.0f}s | RC={result.returncode}\n")
    
    return {
        'symbol': sym, 'elapsed': elapsed, 'rc': result.returncode,
        'stdout_tail': result.stdout[-500:],
        'stderr_tail': result.stderr[-500:] if result.stderr else '',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--parallel', type=int, default=3, help='Concurrent retrains')
    parser.add_argument('--timeout', type=int, default=1800, help='Per-symbol timeout (s)')
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')] if args.symbols else ALL_SYMBOLS
    parallel = min(args.parallel, len(symbols))
    
    print(f"{'='*60}")
    print(f"  PARALLEL BATCH RETRAIN")
    print(f"  Symbols: {len(symbols)} | Parallel: {parallel} | Timeout: {args.timeout}s")
    print(f"{'='*60}")
    
    results = []
    total_start = time.time()
    
    # Process in batches
    for batch_start in range(0, len(symbols), parallel):
        batch = symbols[batch_start:batch_start + parallel]
        print(f"\n{'─'*60}")
        print(f"  Batch {batch_start//parallel + 1}: {', '.join(batch)}")
        print(f"{'─'*60}")
        
        processes = []
        for sym in batch:
            print(f"  🚀 Starting {sym}...")
            p = subprocess.Popen(
                [sys.executable, "-u", "scripts/training/auto_retrain.py", "--symbol", sym, "--force"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            processes.append({'sym': sym, 'proc': p, 'start': time.time()})
        
        # Wait for all in batch
        for p_info in processes:
            sym = p_info['sym']
            proc = p_info['proc']
            start = p_info['start']
            
            try:
                stdout, stderr = proc.communicate(timeout=args.timeout)
                elapsed = time.time() - start
                
                # Write log
                log_file = LOG_DIR / f"{sym}.log"
                with open(log_file, 'w') as f:
                    f.write(f"=== RETRAIN {sym} ===\n")
                    f.write(f"Start: {datetime.utcfromtimestamp(start).isoformat()}\n")
                    f.write(f"Elapsed: {elapsed:.0f}s | RC={proc.returncode}\n\n")
                    f.write(f"== STDOUT ==\n{stdout}\n")
                    if stderr:
                        f.write(f"\n== STDERR ==\n{stderr}\n")
                
                if proc.returncode == 0:
                    print(f"  ✅ {sym} done ({elapsed:.0f}s)")
                else:
                    print(f"  ❌ {sym} FAILED (rc={proc.returncode}, {elapsed:.0f}s)")
                
                results.append({
                    'symbol': sym, 'elapsed': elapsed, 'rc': proc.returncode,
                    'success': proc.returncode == 0,
                })
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                print(f"  ⚠️ {sym} TIMEOUT (> {args.timeout}s)")
                results.append({
                    'symbol': sym, 'elapsed': args.timeout, 'rc': -1,
                    'success': False, 'error': 'timeout',
                })
    
    # Summary
    total_elapsed = time.time() - total_start
    successes = sum(1 for r in results if r['success'])
    failures = sum(1 for r in results if not r['success'])
    
    print(f"\n{'='*60}")
    print(f"  BATCH RETRAIN COMPLETE")
    print(f"{'='*60}")
    print(f"  Total: {len(results)} symbols")
    print(f"  ✅ Success: {successes}")
    print(f"  ❌ Failed: {failures}")
    print(f"  ⏱  Wall time: {total_elapsed:.0f}s ({total_elapsed/60:.0f}m)")
    if successes:
        avg_time = sum(r['elapsed'] for r in results if r['success']) / successes
        print(f"  📊 Avg per symbol: {avg_time:.0f}s")
    
    # Save results
    result_file = LOG_DIR / "_batch_results.json"
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': datetime.utcnow().isoformat(),
            'total': len(results),
            'success': successes,
            'failed': failures,
            'results': results,
        }, f, indent=2)
    print(f"\n  Results saved: {result_file}")


if __name__ == '__main__':
    main()
