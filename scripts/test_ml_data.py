"""Quick test: fetch data for BTCUSDT and verify pagination + features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.strategies.ml_features import prepare_ml_dataset
import pandas as pd

print("Testing BTCUSDT data pipeline...")
result = prepare_ml_dataset("BTCUSDT", days=60, cache_dir="data/ml_cache", target_candle=3)
if result[0] is not None:
    X, y, idx = result
    print(f"\n✅ SUCCESS!")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Class balance: pos={y.sum()}/{len(y)} ({100*y.mean():.1f}%)")
    print(f"  Date range: {idx[0]} to {idx[-1]}")
    print(f"  Feature names (first 10): {list(X.columns[:10])}")
    print(f"  Feature names (last 10): {list(X.columns[-10:])}")
else:
    print("\n❌ FAILED to prepare dataset")
