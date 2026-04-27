"""Direct feature alignment check between live inference and model expectations."""
import sys, json, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import MODEL_DIR, SEEDS, TF_GROUPS
from src.trading.signals import SignalGenerator
import xgboost as xgb

# Test on a few symbols - both 1000x and non-1000x
test_symbols = ['SUIUSDT', '1000PEPEUSDT', 'FETUSDT', 'INJUSDT']

for symbol in test_symbols:
    print(f"\n{'='*60}")
    print(f"  SYMBOL: {symbol}")
    print(f"{'='*60}")
    
    gen = SignalGenerator(symbol)
    
    # Load models and compute fresh features
    cached = gen._load_models(symbol)
    if cached is None:
        print(f"  ❌ Could not load models or features")
        continue
    
    feat_df = cached['features']
    groups = cached['groups']
    
    print(f"  Features computed: {len(feat_df.columns)} columns, {len(feat_df)} rows")
    print(f"  TF groups loaded: {list(groups.keys())}")
    
    for tf, models in groups.items():
        print(f"\n  --- TF Group: {tf} ({len(models)} models) ---")
        for seed, m, mf in models:
            # Check feature availability
            available = [c for c in mf if c in feat_df.columns]
            missing = [c for c in mf if c not in feat_df.columns]
            
            pct_avail = len(available) / max(len(mf), 1) * 100
            print(f"  Seed {seed}: {len(available)}/{len(mf)} features available ({pct_avail:.1f}%)")
            
            if missing:
                # Show first 5 missing features as sample
                sample_missing = missing[:5]
                print(f"    Missing sample: {sample_missing}")
            
            if pct_avail < 80:
                print(f"    ⚠️ CRITICAL: Only {pct_avail:.1f}% features available!")
            
            # Try to actually predict
            X = feat_df[available].fillna(0).clip(-10, 10)
            X_row = X.iloc[-1:].fillna(0)
            try:
                probs = m.predict_proba(X_row)[:, 1]
                print(f"    Prediction: {probs[0]:.4f}")
            except Exception as e:
                print(f"    ❌ Prediction failed: {e}")
