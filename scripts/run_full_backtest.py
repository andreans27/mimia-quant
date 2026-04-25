"""Full backtest — round 2 including meme coins + missed OOS symbols."""
import sys
sys.path.insert(0, ".")

from scripts.threshold_scan import scan_symbol, best_threshold
import json, numpy as np

SYMBOLS = [
    # Meme coins (previously failed)
    "1000PEPEUSDT", "1000BONKUSDT",
    # Previous OOS-only symbols — need full comparison
    "SOLUSDT", "UNIUSDT", "AVAXUSDT",
    # Already done but re-include for completeness
    "APTUSDT", "FETUSDT", "TIAUSDT", "OPUSDT",
    "SUIUSDT", "ARBUSDT", "INJUSDT",
]

print(f"{'='*65}")
print(f"  FULL BACKTEST ROUND 2 — {len(SYMBOLS)} Symbols")
print(f"{'='*65}")

all_results = {}
for sym in SYMBOLS:
    print(f"\n{'='*65}")
    print(f"  {sym}")
    print(f"{'='*65}")
    results = scan_symbol(sym)
    if results:
        best_t, best_r = best_threshold(results)
        all_results[sym] = {'threshold': best_t, 'metrics': best_r, 'all': results}
        marker = " 🔥" if best_r and best_r.get('wr', 0) >= 70 else ""
        print(f"  >> {sym}: thresh={best_t:.2f} | WR {best_r['wr']:.1f}% | PF {best_r['pf']:.2f} | "
              f"M:{best_r['monthly']:.1f}% | DD {best_r['dd']:.2f}% | {best_r['trades']} trades{marker}")
    else:
        print(f"  >> {sym}: scan_symbol returned None")

# Summary
print(f"\n{'='*65}")
print(f"  FINAL RESULTS — Sorted by WR")
print(f"{'='*65}")
print(f"  {'Symbol':<14} {'Thresh':<8} {'WR':<10} {'PF':<8} {'Monthly':<10} {'DD':<8} {'Trades':<8}")
print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

sorted_syms = sorted(all_results.keys(), key=lambda s: all_results[s]['metrics']['wr'], reverse=True)
for sym in sorted_syms:
    r = all_results[sym]['metrics']
    t = all_results[sym]['threshold']
    print(f"  {sym:<14} {t:<8.2f} {r['wr']:<9.1f}% {r['pf']:<8.2f} {r['monthly']:<9.1f}% {r['dd']:<7.2f}% {r['trades']:<8}")

# Top 10 selection
print(f"\n{'='*65}")
print(f"  TOP 10 FOR PAPER TRADE")
print(f"{'='*65}")
top10 = sorted_syms[:10]
for i, sym in enumerate(top10, 1):
    r = all_results[sym]['metrics']
    t = all_results[sym]['threshold']
    print(f"  {i:2d}. {sym:<14} thresh={t:.2f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f} | "
          f"M:{r['monthly']:.1f}% | DD {r['dd']:.2f}% | {r['trades']} trades")

# Save
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open("data/full_backtest_results_r2.json", "w") as f:
    json.dump(all_results, f, indent=2, cls=NpEncoder)

print(f"\n✅ Results saved to data/full_backtest_results_r2.json")
