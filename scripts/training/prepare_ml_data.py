"""
Prepare ML dataset: fetch multi-timeframe data & compute features for all pairs.
Saves to data/ml_cache/ for reuse.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.ml_features import prepare_ml_dataset
import pandas as pd
import gc

# 8 pairs we've been testing on
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT"
]

def main():
    cache_dir = "data/ml_cache"
    days = 120

    print(f"Preparing ML dataset for {len(SYMBOLS)} symbols...")
    print(f"Timeframe: 15m + 1h + 4h, {days} days")
    print()

    all_features = []
    label_info = {}

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing {sym}...")
        print(f"{'='*60}")
        
        try:
            X, y, index_ = prepare_ml_dataset(
                symbol=sym,
                days=days,
                cache_dir=cache_dir,
                target_candle=3  # predict ~45m ahead
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

        if X is None:
            continue

        # Stats
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        total = len(y)
        ratio = pos / max(neg, 1)
        
        print(f"\n  Dataset stats:")
        print(f"    Total samples: {total}")
        print(f"    Positive (up): {pos} ({100*pos/total:.1f}%)")
        print(f"    Negative (down): {neg} ({100*neg/total:.1f}%)")
        print(f"    Ratio pos/neg: {ratio:.3f}")
        print(f"    Features: {len(X.columns)}")
        print(f"    Date range: {index_[0].strftime('%Y-%m-%d')} to {index_[-1].strftime('%Y-%m-%d')}")

        label_info[sym] = {"total": total, "class_balance": ratio}
        all_features.append(sym)

        gc.collect()

    print(f"\n{'='*60}")
    print(f"DONE. Processed {len(all_features)}/{len(SYMBOLS)} symbols.")
    print(f"Cached at: {cache_dir}/<symbol>_features_{days}d_3c.parquet")
    
    # Summary table
    print(f"\n{'Symbol':<10} {'Samples':>10} {'Class Bal':>10}")
    print("-" * 32)
    for sym in all_features:
        info = label_info[sym]
        print(f"{sym:<10} {info['total']:>10,} {info['class_balance']:>10.3f}")

if __name__ == "__main__":
    main()
