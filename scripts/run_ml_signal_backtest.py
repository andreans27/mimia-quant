"""
Run pure-signal ML backtest across all 8 pairs.
No SL/TP — just enter when model fires, hold for N bars, exit.
This tests the model's true predictive ability.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from pathlib import Path

from scripts.ml_signal_backtest import run_pure_signal_backtest_15m

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT"
]

INITIAL_CAPITAL = 5000.0
TAKER_FEE = 0.0004
SLIPPAGE = 0.0005


def run_batch(threshold=0.55, hold_bars=3, cooldown_bars=2, pos_pct=0.10):
    """Run pure-signal backtest with given params."""
    results = []
    print(f"\n{'='*80}")
    print(f"  PURE SIGNAL ML BACKTEST — 15m | thresh={threshold} hold={hold_bars} cool={cooldown_bars}")
    print(f"  No SL/TP — model + fixed holding period only")
    print(f"{'='*80}")
    
    for i, symbol in enumerate(SYMBOLS):
        print(f"\n  [{i+1}/{len(SYMBOLS)}] {symbol}...", end=" ")
        
        result = run_pure_signal_backtest_15m(
            symbol,
            confidence_threshold=threshold,
            hold_bars=hold_bars,
            cooldown_bars=cooldown_bars,
            initial_capital=INITIAL_CAPITAL,
            commission_rate=TAKER_FEE,
            slippage_rate=SLIPPAGE,
            max_position_pct=pos_pct,
            days_data=90,
            warmup_bars=120,
        )
        
        if result:
            passed = result['passed']
            icon = '✅' if passed else '❌'
            m = result['metrics']
            print(f"{icon} Trades={m['total_trades']} WR={m['win_rate']:.1f}% PF={m['profit_factor']:.2f} "
                  f"PnL=${m['total_pnl']:.2f} DD={m['max_drawdown']:.1f}% Sharpe={m['sharpe']:.2f}")
            results.append(result)
        else:
            print("❌ failed")
    
    if not results:
        print("\n  ❌ All failed")
        return results
    
    # Summary
    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Symbol':<10} {'Trades':>7} {'WR':>6} {'PF':>6} {'AvgM%':>7} {'MaxDD':>7} {'Sharpe':>6} {'Long$':>9} {'Short$':>9}")
    print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*9} {'-'*9}")
    
    total_passed = 0
    for r in results:
        m = r['metrics']
        p = '✅' if r['passed'] else '❌'
        if r['passed']: total_passed += 1
        print(f"  {r['symbol']:<10} {m['total_trades']:>7} {m['win_rate']:>5.1f}% {m['profit_factor']:>5.2f} "
              f"{m['avg_monthly_return']:>6.2f}% {m['max_drawdown']:>6.2f}% {m['sharpe']:>5.2f} "
              f"${m['long_pnl']:>7.2f} ${m['short_pnl']:>7.2f} {p:>3}")
    
    print(f"\n  Passed: {total_passed}/{len(results)}")
    
    # Print failed details
    failed = [r for r in results if not r['passed']]
    if failed and len(failed) == len(results):
        print(f"\n  Failed criteria analysis (first 3):")
        for f_result in failed[:3]:
            print(f"\n  {f_result['symbol']}:")
            for name, (passed, val) in f_result['criteria'].items():
                icon = '✅' if passed else '❌'
                print(f"    {icon} {name:<35s} {val}")
    
    return results


if __name__ == "__main__":
    import json
    
    # Run with default params: threshold 0.55, hold 3 bars, cooldown 2
    results = run_batch(threshold=0.55, hold_bars=3, cooldown_bars=2, pos_pct=0.10)
    
    output = []
    for r in results:
        m = r['metrics']
        output.append({
            'symbol': r['symbol'], 'passed': r['passed'],
            'trade_count': r['trade_count'],
            'metrics': {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                       for k, v in m.items()},
        })
    
    with open('data/ml_signal_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    if results:
        # Check the model's raw accuracy
        avg_wr = np.mean([r['metrics']['win_rate'] for r in results])
        avg_pf = np.mean([r['metrics']['profit_factor'] for r in results])
        avg_trades = np.mean([r['trade_count'] for r in results])
        print(f"\n  {'='*60}")
        print(f"  RAW MODEL EDGE ASSESSMENT")
        print(f"  {'='*60}")
        print(f"  Avg Win Rate:     {avg_wr:.1f}%")
        print(f"  Avg Profit Factor: {avg_pf:.2f}")
        print(f"  Avg Trade Count:  {avg_trades:.0f}")
        print(f"  Model Training AUC: ~0.79 (test set)")
        print(f"  Conclusion: Model has {'significant' if avg_pf > 2.0 else 'moderate' if avg_pf > 1.5 else 'small'} directional edge")
