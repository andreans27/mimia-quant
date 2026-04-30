"""Analyze model output distribution and test percentile-based filtering."""
import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.trading.backtest import run_backtest_live_aligned

r = run_backtest_live_aligned('WIFUSDT', test_hours=24, verbose=False)
if not r:
    print("Backtest failed")
    sys.exit(1)

probas = np.array(r['long_probas'])
signals = np.array(r['signals'])

print('=== Probability Distribution ===')
for pctl in [80, 85, 90, 95, 99]:
    thr = np.percentile(probas[probas > 0], pctl)
    above = (probas >= thr).sum()
    print(f'Top {100-pctl}%: threshold={thr:.4f}, bars={above}/{len(probas)}')

print()
print('=== Signal Distribution by Proba Bucket ===')
buckets = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
for lo, hi in buckets:
    mask = (signals != 0) & (probas >= lo) & (probas < hi)
    n = mask.sum()
    if n > 0:
        dirs = signals[mask]
        longs = (dirs == 1).sum()
        shorts = (dirs == -1).sum()
        print(f'  [{lo:.2f}, {hi:.2f}): {n} signals ({longs}L, {shorts}S)')

print()
print(f'Total non-flat signals: {(signals != 0).sum()}')
print(f'Signal ratio: {(signals != 0).sum()}/{len(signals)} = {(signals != 0).mean()*100:.1f}%')
