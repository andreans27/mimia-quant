#!/usr/bin/env python3
"""
P2: BTC Threshold Scan — test 0.50, 0.55, 0.60, 0.65, 0.70
Compare each to baseline (threshold=0.60) and find optimal.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.backtesting.compare_exit_strategies import (
    run_all_strategies, EXIT_STRATEGIES, INITIAL_CAPITAL,
)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
SYMBOLS = ['BTCUSDT']


def main():
    all_results = {}
    best = {'threshold': None, 'pf': 0, 'wr': 0, 'return_pct': 0}

    for thresh in THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"  THRESHOLD = {thresh}")
        print(f"{'='*60}")
        t0 = time.time()
        results = run_all_strategies('BTCUSDT', threshold=thresh,
                                     hold_bars=9, cooldown_bars=3,
                                     days_data=130, warmup_bars=200)
        elapsed = time.time() - t0
        if results:
            all_results[thresh] = results
            bl = results.get('baseline', {})
            if bl:
                pf = bl.get('profit_factor', 0)
                wr = bl.get('win_rate_pct', 0)
                ret = bl.get('total_return_pct', 0)
                trades = bl.get('total_trades', 0)
                dd = bl.get('max_drawdown_pct', 0)
                sharpe = bl.get('sharpe_ratio', 0)
                print(f"\n  ⏱ {elapsed:.0f}s")
                print(f"  → WR={wr:.1f}% PF={pf:.2f} Return={ret:.1f}% DD={dd:.2f}% Trades={trades} Sharpe={sharpe:.2f}")
                if pf > best['pf']:
                    best = {'threshold': thresh, 'pf': pf, 'wr': wr, 'return_pct': ret,
                            'trades': trades, 'dd': dd, 'sharpe': sharpe}
        else:
            print(f"\n  ❌ Gagal untuk threshold {thresh}")

    # ── Comparison table ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  BTC THRESHOLD COMPARISON — Baseline Strategy Only")
    print(f"{'='*80}")
    rows = []
    for thresh in THRESHOLDS:
        r = all_results.get(thresh, {}).get('baseline')
        if r:
            rows.append({
                'Threshold': f"{thresh:.2f}",
                'WR%': f"{r['win_rate_pct']:.1f}",
                'PF': f"{r['profit_factor']:.2f}",
                'Return%': f"{r['total_return_pct']:.1f}",
                'Mo%': f"{r['avg_monthly_return_pct']:.1f}",
                'DD%': f"{r['max_drawdown_pct']:.2f}",
                'Trades': r['total_trades'],
                'Sharpe': f"{r['sharpe_ratio']:.2f}",
                'Sortino': f"{r['sortino_ratio']:.2f}",
                'Wins': sum(1 for t in r.get('exit_reasons', {}).keys() if 'hold' in t.lower()),
                'Losses': sum(v for k, v in r.get('exit_reasons', {}).items() if 'hold' not in k.lower()),
            })
    print(f"\n{tabulate(rows, headers='keys', tablefmt='grid')}")

    # ── Recommendation ───────────────────────────────────────────────
    print(f"\n  BEST: threshold = {best['threshold']} (PF={best['pf']:.2f}, WR={best['wr']:.1f}%)")
    baseline = THRESHOLDS.index(0.60)
    best_idx = THRESHOLDS.index(best['threshold'])
    thresh_delta = best['threshold'] - 0.60
    print(f"  Delta from 0.60: {thresh_delta:+.2f}")
    if thresh_delta > 0:
        print(f"  → RECOMMEND: Raise BTC threshold to {best['threshold']}")
    elif thresh_delta < 0:
        print(f"  → RECOMMEND: Lower BTC threshold to {best['threshold']}")
    else:
        print(f"  → 0.60 already optimal for BTC")

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
