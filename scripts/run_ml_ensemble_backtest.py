"""
Run ensemble ML backtest across all 8 pairs.
Tests various thresholds to find optimal balance.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import json
from pathlib import Path

from scripts.ml_ensemble_backtest import run_ensemble_backtest

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT"
]

INITIAL_CAPITAL = 5000.0
TAKER_FEE = 0.0004
SLIPPAGE = 0.0005


def run_sweep(thresholds=[0.55, 0.60, 0.65], hold_bars=9, cooldown_bars=3):
    """Run ensemble backtest at multiple thresholds on 5m data."""
    all_results = {}
    
    for thresh in thresholds:
        print(f"\n{'='*80}")
        print(f"  ENSEMBLE BACKTEST — 5m base (45min hold) | threshold={thresh} hold={hold_bars} cool={cooldown_bars}")
        print(f"{'='*80}")
        
        results = []
        for i, symbol in enumerate(SYMBOLS):
            print(f"\n  [{i+1}/{len(SYMBOLS)}] {symbol} (thr={thresh})...", end=" ")
            
            result = run_ensemble_backtest(
                symbol,
                confidence_threshold=thresh,
                hold_bars=hold_bars,
                cooldown_bars=cooldown_bars,
                initial_capital=INITIAL_CAPITAL,
                commission_rate=TAKER_FEE,
                slippage_rate=SLIPPAGE,
                max_position_pct=0.10,
                days_data=90,
                warmup_bars=200,
            )
            
            if result:
                passed = result['passed']
                icon = '✅' if passed else '❌'
                m = result['metrics']
                trade_freq = m['total_trades'] / max(1, (m['days'] if 'days' in m else 60))
                print(f"{icon} Trades={m['total_trades']} WR={m['win_rate']:.1f}% PF={m['profit_factor']:.2f} "
                      f"PnL=${m['total_pnl']:.2f} DD={m['max_drawdown']:.1f}% Sharpe={m['sharpe']:.2f}")
                results.append(result)
            else:
                print("❌ failed")
        
        all_results[thresh] = results
        
        # Summary table for this threshold
        print(f"\n  {'='*50}")
        print(f"  SUMMARY @ threshold={thresh}")
        print(f"  {'='*50}")
        print(f"  {'Symbol':<10} {'Trades':>7} {'WR':>6} {'PF':>6} {'DD':>7} {'Sharpe':>6} {'Long$':>9} {'Short$':>9} Pass")
        print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*9} {'-'*9} {'-'*4}")
        
        passed_count = 0
        for r in results:
            m = r['metrics']
            p = '✅' if r['passed'] else '❌'
            if r['passed']: passed_count += 1
            print(f"  {r['symbol']:<10} {m['total_trades']:>7} {m['win_rate']:>5.1f}% {m['profit_factor']:>5.2f} "
                  f"{m['max_drawdown']:>6.2f}% {m['sharpe']:>5.2f} "
                  f"${m['long_pnl']:>7.2f} ${m['short_pnl']:>7.2f} {p}")
        
        avg_wr = np.mean([r['metrics']['win_rate'] for r in results])
        avg_pf = np.mean([r['metrics']['profit_factor'] for r in results])
        avg_trades = np.mean([r['metrics']['total_trades'] for r in results])
        avg_pnl = np.mean([r['metrics']['total_pnl'] for r in results])
        avg_dd = np.mean([abs(r['metrics']['max_drawdown']) for r in results])
        
        print(f"\n  Averages:  {avg_trades:>7.0f} {avg_wr:>5.1f}% {avg_pf:>5.2f} {avg_dd:>6.2f}%")
        print(f"  Passed: {passed_count}/{len(results)}")
    
    return all_results


if __name__ == "__main__":
    # Run sweep
    results = run_sweep(thresholds=[0.55, 0.60, 0.65], hold_bars=9, cooldown_bars=3)
    
    # Save all results
    output = {}
    for thresh, res_list in results.items():
        output[str(thresh)] = []
        for r in res_list:
            m = r['metrics']
            output[str(thresh)].append({
                'symbol': r['symbol'],
                'passed': r['passed'],
                'trade_count': r['trade_count'],
                'win_rate': float(m['win_rate']),
                'profit_factor': float(m['profit_factor']),
                'max_drawdown': float(m['max_drawdown']),
                'sharpe': float(m['sharpe']),
                'total_pnl': float(m['total_pnl']),
                'avg_monthly_return': float(m['avg_monthly_return']),
                'long_pnl': float(m['long_pnl']),
                'short_pnl': float(m['short_pnl']),
            })
    
    with open('data/ml_ensemble_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved: data/ml_ensemble_results.json")
