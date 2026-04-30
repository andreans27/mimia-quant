#!/usr/bin/env python3
"""Run per-bar backtest for all symbols with trade simulation."""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

from src.trading.backtest import run_backtest_live_aligned
from src.trading.state import LIVE_SYMBOLS

import json

results = []
for i, sym in enumerate(LIVE_SYMBOLS):
    t0 = time.time()
    r = run_backtest_live_aligned(sym, test_hours=24, verbose=False)
    elapsed = time.time() - t0
    if r:
        results.append(r)
        sigs = r['signals']
        sig_pct = (sigs.count(1)+sigs.count(-1))/len(sigs)*100
        print(f'{sym:>15s} | {r["n_trades"]:>3d} trades | WR={r["win_rate"]:>5.1f}% | PF={r["profit_factor"]:>7.2f}x | PnL=${r["total_pnl"]:>+8.2f} | DD={r["max_dd"]:>5.2f}% | Sig={sig_pct:>5.1f}% | [{elapsed:.0f}s]')
    else:
        print(f'{sym:>15s} | FAILED')
    sys.stdout.flush()

print()
trades = sum(r['n_trades'] for r in results)
wins = sum(sum(1 for t in r.get('trade_details', []) if t['pnl'] > 0) for r in results)
pnl = sum(r['total_pnl'] for r in results)
lp = sum(r['long_pnl'] for r in results)
sp = sum(r['short_pnl'] for r in results)
wr = wins/trades*100 if trades else 0
print(f'{"TOTAL":>15s} | {trades:>3d} trades | WR={wr:>5.1f}% | {"":>9s} | PnL=${pnl:>+8.2f} | LONG=${lp:+.2f} SHORT=${sp:+.2f}')

# Save
log = Path("data") / "per_bar_results.json"
with open(log, 'w') as f:
    json.dump([{
        'symbol': r['symbol'], 'trades': r['n_trades'],
        'wr': r['win_rate'], 'pf': r['profit_factor'],
        'pnl': r['total_pnl'], 'dd': r['max_dd'],
        'long_pnl': r['long_pnl'], 'short_pnl': r['short_pnl'],
    } for r in results], f, indent=2)
print(f'\nSaved to {log}')
