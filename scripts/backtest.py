#!/usr/bin/env python3
"""
Mimia — Walk-Forward Backtest CLI
==================================
Run correct walk-forward backtest with no look-ahead bias.
DateTime-based TF completion + batch inference.

Usage:
  python scripts/backtest.py                              # all symbols, 24h
  python scripts/backtest.py --symbols ENAUSDT,SOLUSDT    # specific
  python scripts/backtest.py --hours 48                   # 48h window
  python scripts/backtest.py --out results.txt            # custom output
"""
import sys, os, time as ttime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
import numpy as np
from datetime import datetime

from src.trading.backtest import run_backtest, LIVE_SYMBOLS

DEFAULT_SYMBOLS = LIVE_SYMBOLS  # all 20 live trading pairs


def fmt_pnl(v: float) -> str:
    return f"${v:+.2f}"


def main():
    parser = argparse.ArgumentParser(description='Mimia Walk-Forward Backtest')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--hours', type=int, default=24, help='Test window (hours)')
    parser.add_argument('--out', type=str, default='data/backtest_results.txt',
                        help='Output file path')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show per-symbol progress')
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',')] if args.symbols \
              else DEFAULT_SYMBOLS
    out = args.out

    # Open output
    with open(out, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"MIMIA WALK-FORWARD BACKTEST\n")
        f.write(f"Window: {args.hours}h | Symbols: {len(symbols)}\n")
        f.write(f"Models: frozen (latest at {datetime.utcnow().strftime('%m/%d %H:%M')} UTC)\n")
        f.write(f"{'='*70}\n")

    results = []
    total_t0 = ttime.time()

    for sym in symbols:
        print(f"\n  🔄 {sym} ({args.hours}h)...", end=' ', flush=True)
        r = run_backtest(sym, test_hours=args.hours, verbose=args.verbose)
        if r:
            results.append((sym, r))
            print(f"✅ {r['n_trades']} trades, WR={r['win_rate']:.0f}%, PnL={fmt_pnl(r['total_pnl'])}")
        else:
            print(f"❌ skipped (no models/data)")

    te = ttime.time() - total_t0

    # Write summary
    with open(out, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"SUMMARY — {args.hours}h ({len(results)}/{len(symbols)} symbols)\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'Symbol':>10s} | {'Trades':>6s} | {'WR':>5s} | {'PF':>5s} | "
                f"{'PnL':>9s} | {'DD':>6s} | {'Long':>8s} | {'Short':>8s} | Gates\n")
        f.write(f"{'-'*10}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*9}-+-{'-'*6}-"
                f"-{'-'*8}-+-{'-'*8}-+-------\n")

        for sym, r in results:
            gates = []
            if r['win_rate'] >= 70: gates.append(f"WR≥{r['win_rate']:.0f}%")
            if r['profit_factor'] >= 2.0: gates.append(f"PF≥{r['profit_factor']:.1f}")
            if r['total_pnl'] > 0: gates.append("PnL+")
            if r['long_pnl'] > 0: gates.append("L+")
            if r['short_pnl'] > 0: gates.append("S+")
            if r['max_dd'] < 10: gates.append(f"DD<{r['max_dd']:.1f}%")
            g = ' '.join(gates) if gates else '❌'

            f.write(f"{sym:>10s} | {r['n_trades']:6d} | {r['win_rate']:4.0f}% | "
                    f"{r['profit_factor']:4.1f} | {fmt_pnl(r['total_pnl']):>9s} | "
                    f"{r['max_dd']:5.2f}% | {fmt_pnl(r['long_pnl']):>8s} | "
                    f"{fmt_pnl(r['short_pnl']):>8s} | {g}\n")

        tpnl = sum(r['total_pnl'] for _, r in results)
        ttr = sum(r['n_trades'] for _, r in results)
        awr = np.mean([r['win_rate'] for _, r in results]) if results else 0
        passed = sum(1 for _, r in results if r['total_pnl'] > 0 and r['win_rate'] >= 30)
        f.write(f"\nTotal: {len(results)} symbols | {ttr} trades | Avg WR: {awr:.0f}%\n")
        f.write(f"Total PnL: {fmt_pnl(tpnl)} | Passed (PnL+ & WR≥30%): {passed}/{len(results)}\n")
        f.write(f"Duration: {te:.0f}s ({te/len(results):.0f}s/sym)\n" if results else "")
        f.write(f"DONE\n")

    print(f"\n{'='*70}")
    print(f"✅ Done — {len(results)} symbols in {te:.0f}s")
    print(f"   Output: {out}")


if __name__ == '__main__':
    main()
