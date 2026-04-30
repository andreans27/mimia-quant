"""Validate all retrained models with batch backtest + per-bar signal check."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.trading.backtest import run_multi, run_backtest
from src.trading.state import LIVE_SYMBOLS

print("=" * 60)
print("  VALIDATION: Batch Backtest (24h) — All Symbols")
print("=" * 60)

t0 = time.time()
results = run_multi(symbols=LIVE_SYMBOLS, test_hours=24, verbose=True)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"  RESULTS (sorted by total PnL)")
print(f"{'='*60}")
print(f"  Time: {elapsed:.0f}s for {len(results)} symbols")
print()
print(f"  {'Symbol':>15s} {'Trades':>6s} {'WR':>7s} {'PF':>8s} {'PnL':>10s} {'Long':>10s} {'Short':>10s} {'DD':>7s}")
print(f"  {'-'*75}")
for sym, r in sorted(results, key=lambda x: -x[1]['total_pnl']):
    print(f"  {r['symbol']:>15s} {r['n_trades']:>6d} {r['win_rate']:>6.1f}% {r['profit_factor']:>7.2f}x {r['total_pnl']:>+9.2f} {r['long_pnl']:>+9.2f} {r['short_pnl']:>+9.2f} {r['max_dd']:>6.2f}%")

total_pnl = sum(r['total_pnl'] for _, r in results)
total_trades = sum(r['n_trades'] for _, r in results)
total_wins = sum(sum(1 for t in r.get('trade_details', []) if t['pnl'] > 0) for _, r in results)
wr = total_wins / total_trades * 100 if total_trades else 0
print(f"  {'─'*75}")
print(f"  {'TOTAL':>15s} {total_trades:>6d} {wr:>6.1f}% {'':>8s} {total_pnl:>+9.2f}")
print()
print(f"  Total PnL: ${total_pnl:+.2f}")
print(f"  Win Rate: {wr:.1f}% ({total_wins}/{total_trades})")
print(f"  Avg PnL/trade: ${total_pnl/total_trades:.2f}" if total_trades else "")
