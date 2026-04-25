#!/usr/bin/env python3
"""
Batch rapid evaluation — runs auto_retrain for multiple symbols sequentially.
Outputs retrain_status.json metrics per symbol.
"""
import sys, os, subprocess, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

# Candidates ranked by volatility + liquidity potential
CANDIDATES = [
    # Meme coins (high vol + high volume)
    "DOGEUSDT",    # Already has 120d cache + models
    "WIFUSDT",     # Top meme coin, high vol
    "POPCATUSDT",  # High vol (4.0%), solana meme
    "NEIROUSDT",   # Very high vol (4.4%), massive volume
    # High-beta altcoins
    "ENAUSDT",     # High vol (3.3%)
    "SEIUSDT",     # High vol (4.1%)
]

def has_feature_cache(symbol):
    cache_dir = ROOT / "data" / "ml_cache"
    for f in cache_dir.glob(f"{symbol}*.parquet"):
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=None, help='Override symbols')
    parser.add_argument('--quick', action='store_true', help='Only symbols with existing cache')
    args = parser.parse_args()
    
    symbols = args.symbols or CANDIDATES
    
    results = {}
    
    for sym in symbols:
        has_cache = has_feature_cache(sym)
        if args.quick and not has_cache:
            print(f"\n⏭️  {sym} — skipping (no cache, --quick mode)")
            continue
        
        print(f"\n{'='*60}")
        print(f"TRAINING: {sym}  [cache={'✅' if has_cache else '❌'}]")
        print(f"{'='*60}")
        
        start = time.time()
        result = subprocess.run(
            [sys.executable, "-u", "scripts/training/auto_retrain.py", "--symbol", sym, "--force"],
            capture_output=True, text=True, timeout=600
        )
        elapsed = time.time() - start
        print(f"  Elapsed: {elapsed:.0f}s")
        
        if result.returncode != 0:
            print(f"  ❌ FAILED (rc={result.returncode})")
            print(f"  Stderr: {result.stderr[-500:]}")
            results[sym] = {"status": "failed", "error": result.stderr[-200:]}
        else:
            print(f"  ✅ SUCCESS")
            print(f"  Stdout[-500]: {result.stdout[-500:]}")
            results[sym] = {"status": "success", "elapsed": elapsed}
        
        # Check retrain status
        status_file = ROOT / "data" / "ml_models" / "_retrain_status.json"
        if status_file.exists():
            status_data = json.loads(status_file.read_text())
            info = status_data.get(sym, {})
            if info:
                wr = info.get('wr', '?')
                pf = info.get('pf', '?')
                sharpe = info.get('sharpe', '?')
                dd = info.get('max_dd', '?')
                nt = info.get('n_trades', '?')
                dep = info.get('deployed', '?')
                print(f"\n  📊 {sym} RETRAIN STATUS:")
                print(f"     WR={wr}% PF={pf} Sharpe={sharpe} DD={dd}% Trades={nt} Deployed={dep}")
                results[sym].update(info)
        
        # Save intermediate results
        (ROOT / "data" / "rapid_eval" / "batch_results.json").write_text(json.dumps(results, indent=2, default=str))
    
    # Final summary
    print(f"\n\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    for sym, info in results.items():
        if 'wr' in info:
            wr = info.get('wr', '?')
            pf = info.get('pf', '?')
            sharpe = info.get('sharpe', '?')
            dd = info.get('max_dd', '?')
            verdict = "✅" if (isinstance(wr, (int,float)) and wr >= 65 and isinstance(pf, (int,float)) and pf >= 1.8) else "⚠️" if (isinstance(wr, (int,float)) and wr >= 55) else "❌"
            print(f"  {verdict} {sym:<12} WR={wr:<6} PF={pf:<6} Sharpe={sharpe:<8} DD={dd}%")
        else:
            print(f"  ❌ {sym:<12} FAILED — {info.get('error','?')}")

if __name__ == '__main__':
    main()
