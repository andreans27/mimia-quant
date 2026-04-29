#!/usr/bin/env python3
"""Backtest all 20 LIVE_SYMBOLS for 1 week (168h)"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.trading.backtest import run_backtest
from src.trading.state import LIVE_SYMBOLS

now = __import__('datetime').datetime.utcnow()

print(f'{"="*80}')
print(f'  MIMIA — BACKTEST 1 MINGGU (168h) — ALL 20 SYMBOLS')
print(f'  Model: latest deployed ({now.strftime("%d %b %H:%M")} UTC)')
print(f'  Capital: $5000 | Fee: 0.04% | Slippage: 0.05% | Hold: 9 bars')
print(f'{"="*80}')
print()

results = []
for sym in LIVE_SYMBOLS:
    t0 = time.time()
    print(f'  🔄 {sym}...', end=' ', flush=True)
    r = run_backtest(sym, test_hours=168, verbose=False)
    t = time.time() - t0
    if r:
        results.append((sym, r))
        n_sig = sum(1 for s in r['signals'] if s != 0)
        bars = len(r['signals'])
        print(f'✅ {t:.0f}s | T:{r["n_trades"]:3d} WR:{r["win_rate"]:4.1f}% PF:{r["profit_factor"]:4.2f} P&L:{r["total_pnl"]:+.1f} DD:{r["max_dd"]:.2f}% Sig:{n_sig}/{bars}')
    else:
        print(f'⏭️')

print()
print(f'{"="*80}')
print(f'  BACKTEST RESULTS — 1 WEEK (168h)')
print(f'{"="*80}')
header = (f'  {"Symbol":>10s} | {"Trades":>6s} | {"WR%":>5s} | {"PF":>5s} | '
          f'{"P&L":>8s} | {"Long$":>8s} | {"Short$":>8s} | {"DD%":>6s} | {"Sig":>4s}')
print(header)
print(f'  {"-"*76}')

total_t = 0
total_p = 0.0
pos_pnl = 0
neg_pnl = 0
pf_gt_1 = 0
dd_lt_5 = 0

for sym, r in sorted(results, key=lambda x: -x[1]['n_trades']):
    total_t += r['n_trades']
    total_p += r['total_pnl']
    n_sig = sum(1 for s in r['signals'] if s != 0)
    print(f'  {sym:>10s} | {r["n_trades"]:6d} | {r["win_rate"]:4.1f}% | '
          f'{r["profit_factor"]:4.2f} | {r["total_pnl"]:7.2f} | '
          f'{r["long_pnl"]:7.2f} | {r["short_pnl"]:7.2f} | '
          f'{r["max_dd"]:5.2f}% | {n_sig:4d}')
    if r['total_pnl'] > 0: pos_pnl += 1
    else: neg_pnl += 1
    if r['profit_factor'] > 1.0: pf_gt_1 += 1  # PF >1 minimal profitable
    if r['max_dd'] < 5: dd_lt_5 += 1

print(f'  {"-"*76}')
print(f'  {"TOTAL":>10s} | {total_t:6d} | {"":>5s} | {"":>5s} | '
      f'{total_p:7.2f} | {"":>8s} | {"":>8s} | {"":>6s} | {"":>4s}')

# Filter by WR >= 30%
wr_ge_30 = sum(1 for _, r in results if r['win_rate'] >= 30)
pf_ge_1 = sum(1 for _, r in results if r['profit_factor'] >= 1)
pf_ge_1_5 = sum(1 for _, r in results if r['profit_factor'] >= 1.5)

print()
print(f'  📊 STATS:')
print(f'     Positive P&L:  {pos_pnl}/{len(results)}')
print(f'     Negative P&L:  {neg_pnl}/{len(results)}')
print(f'     WR >= 30%:     {wr_ge_30}/{len(results)}')
print(f'     PF >= 1.0:     {pf_ge_1}/{len(results)}')
print(f'     PF >= 1.5:     {pf_ge_1_5}/{len(results)}')
print(f'     DD < 5%:       {dd_lt_5}/{len(results)}')
print(f'     Total P&L:     ${total_p:.2f}')
print(f'     Avg P&L:       ${total_p/len(results):.2f}/symbol')
if total_t > 0:
    print(f'     Avg WR:        {sum(r["win_rate"] for _,r in results)/len(results):.1f}%')
    print(f'     Avg PF:        {sum(r["profit_factor"] for _,r in results)/len(results):.2f}')

# Save
with open('data/backtest_1week.json', 'w') as f:
    dump = {}
    for sym, r in results:
        dump[sym] = {
            'n_trades': r['n_trades'], 'win_rate': r['win_rate'],
            'profit_factor': r['profit_factor'], 'total_pnl': r['total_pnl'],
            'long_pnl': r['long_pnl'], 'short_pnl': r['short_pnl'],
            'max_dd': r['max_dd'],
        }
    json.dump({'timestamp': str(now), 'results': dump}, f, indent=2)

print(f'\n  ✅ Saved → data/backtest_1week.json')
