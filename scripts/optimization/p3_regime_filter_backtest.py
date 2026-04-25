#!/usr/bin/env python3
"""
P3: Regime Filter Comparison — test ATR filter, Trend filter, Combined vs Baseline.

Compares 4 modes on 10 symbols:
  1. baseline (no filter)
  2. atr_filter (skip high volatility entries)
  3. trend_filter (long when bullish, short when bearish)
  4. combined (ATR + Trend)

Output: per-symbol + aggregate comparison table.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.backtesting.compare_exit_strategies import run_all_strategies, EXIT_STRATEGIES

SYMBOLS = ['APTUSDT', 'UNIUSDT', 'FETUSDT', 'TIAUSDT', 'SOLUSDT',
           'OPUSDT', '1000PEPEUSDT', 'SUIUSDT', 'ARBUSDT', 'INJUSDT']

FILTER_MODES = [
    {'name': 'baseline',       'atr': False, 'trend': False},
    {'name': 'atr_filter',     'atr': True,  'trend': False},
    {'name': 'trend_filter',   'atr': False, 'trend': True},
    {'name': 'combined',       'atr': True,  'trend': True},
]

METRICS = ['win_rate_pct', 'profit_factor', 'avg_monthly_return_pct',
           'max_drawdown_pct', 'total_trades', 'sharpe_ratio', 'total_return_pct']


def main():
    # ── Run all filter modes on all symbols ──────────────────────────
    all_data = {}  # {symbol: {mode: results}}
    for symbol in SYMBOLS:
        all_data[symbol] = {}
        for mode in FILTER_MODES:
            print(f"\n{'='*70}")
            print(f"  {symbol} — {mode['name'].upper()}")
            print(f"{'='*70}")
            t0 = time.time()
            results = run_all_strategies(
                symbol,
                threshold=0.60,
                hold_bars=9,
                cooldown_bars=3,
                days_data=130,
                warmup_bars=200,
                use_atr_filter=mode['atr'],
                use_trend_filter=mode['trend'],
                filter_name=mode['name'],
            )
            elapsed = time.time() - t0
            if results:
                all_data[symbol][mode['name']] = results
                bl = results.get('baseline', {})
                if bl:
                    print(f"  ⏱ {elapsed:.0f}s | WR={bl['win_rate_pct']:.1f}% PF={bl['profit_factor']:.2f} "
                          f"Return={bl['total_return_pct']:.1f}% DD={bl['max_drawdown_pct']:.2f}% "
                          f"Trades={bl['total_trades']}")
            else:
                print(f"  ❌ Gagal")
                all_data[symbol][mode['name']] = None
        print(f"\n  {'='*70}")

    # ── Aggregate comparison table (baseline strategy only) ─────────
    print(f"\n\n{'='*120}")
    print(f"  P3: REGIME FILTER COMPARISON — Baseline Strategy Across All Filter Modes")
    print(f"{'='*120}")

    # Per-mode aggregate
    mode_aggs = {m['name']: {met: [] for met in METRICS} for m in FILTER_MODES}

    for symbol in SYMBOLS:
        for m in FILTER_MODES:
            mode_name = m['name']
            r = all_data.get(symbol, {}).get(mode_name)
            if r and r.get('baseline'):
                bl = r['baseline']
                for met in METRICS:
                    val = bl.get(met, 0)
                    if val is None:
                        val = 0
                    mode_aggs[mode_name][met].append(val)

    rows = []
    for m in FILTER_MODES:
        mode_name = m['name']
        agg = mode_aggs[mode_name]
        means = {k: np.mean(v) for k, v in agg.items()}
        rows.append({
            'Filter': mode_name,
            'WR%': f"{means['win_rate_pct']:.1f}",
            'PF': f"{means['profit_factor']:.2f}",
            'Mo%': f"{means['avg_monthly_return_pct']:.1f}",
            'DD%': f"{means['max_drawdown_pct']:.2f}",
            'Trades': f"{means['total_trades']:.0f}",
            'Sharpe': f"{means['sharpe_ratio']:.2f}",
            'Return%': f"{means['total_return_pct']:.1f}",
        })

    print(f"\nAverages across {len(SYMBOLS)} symbols:")
    print(tabulate(rows, headers='keys', tablefmt='grid'))

    # ── Per-symbol detail ────────────────────────────────────────────
    print(f"\n\n{'='*120}")
    print(f"  PER-SYMBOL DETAIL (WR / PF / Return%)")
    print(f"{'='*120}")

    for symbol in SYMBOLS:
        sym_data = all_data.get(symbol, {})
        sub_rows = []
        for m in FILTER_MODES:
            r = sym_data.get(m['name'])
            if r and r.get('baseline'):
                bl = r['baseline']
                sub_rows.append({
                    'Filter': m['name'],
                    'WR%': f"{bl['win_rate_pct']:.1f}",
                    'PF': f"{bl['profit_factor']:.2f}",
                    'Return%': f"{bl['total_return_pct']:.1f}",
                    'Mo%': f"{bl['avg_monthly_return_pct']:.1f}",
                    'DD%': f"{bl['max_drawdown_pct']:.2f}",
                    'Trades': bl['total_trades'],
                    'Sharpe': f"{bl['sharpe_ratio']:.2f}",
                })
            else:
                sub_rows.append({'Filter': m['name'], 'WR%': '❌', 'PF': '❌', 'Return%': '❌',
                                 'Mo%': '❌', 'DD%': '❌', 'Trades': 0, 'Sharpe': '❌'})
        print(f"\n  {symbol}")
        print(tabulate(sub_rows, headers='keys', tablefmt='simple'))

    print(f"\n\nP3 comparison complete. Output above.")
    print(f"Best filter: check which row has highest PF and WR while keeping DD < 10%.")


if __name__ == '__main__':
    main()
