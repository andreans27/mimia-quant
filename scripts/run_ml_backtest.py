"""
Run full ML backtest across all 8 pairs — 15m timeframe, native model predictions.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scripts.ml_backtest_integration import run_ml_backtest_15m

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT"
]

INITIAL_CAPITAL = 5000.0
TAKER_FEE = 0.0004
SLIPPAGE = 0.0005

CRITERIA = [
    ("Max Drawdown < 10%",      lambda s: s["max_drawdown"] > -10),
    ("Avg Monthly Return ≥ 10%",lambda s: s["avg_monthly_return"] >= 10),
    ("Win Rate > 70%",          lambda s: s["win_rate"] > 70),
    ("Profit Factor > 2.0",     lambda s: s["profit_factor"] > 2.0),
    ("Sharpe Ratio > 2.0",      lambda s: s["sharpe"] > 2.0),
    ("Sortino Ratio > 2.5",     lambda s: s["sortino"] > 2.5),
    ("Trades/Day ≥ 5",          lambda s: s["trades_per_day"] >= 5),
    ("Monthly Consistency ≥ 80%", lambda s: s["monthly_consistency"] >= 80),
    ("Long Side Profitable",    lambda s: s["long_pnl"] > 0),
    ("Short Side Profitable",   lambda s: s["short_pnl"] > 0),
    ("Trade Count ≥ 300",       lambda s: s["total_trades"] >= 300),
]


def print_results(results: list):
    """Pretty print ML backtest results."""
    print(f"\n{'='*80}")
    print(f"  ML BACKTEST RESULTS — 15m Timeframe")
    print(f"  Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"{'='*80}")
    
    if not results:
        print("  ❌ No results returned")
        return
    
    print(f"\n  {'Symbol':<10} {'Trades':>7} {'WR':>6} {'PF':>6} {'AvgM%':>7} {'MaxDD':>7} {'Sharpe':>7} {'Sortino':>7} {'Long$':>9} {'Short$':>9} {'TP':>6}")
    print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*6}")
    
    total_passed = 0
    for r in results:
        m = r['metrics']
        symbol = r['symbol']
        trades = m['total_trades']
        wr = m['win_rate']
        pf = m['profit_factor']
        avgm = m['avg_monthly_return']
        mdd = m['max_drawdown']
        sharpe = m['sharpe']
        sortino = m['sortino']
        long_pnl = m['long_pnl']
        short_pnl = m['short_pnl']
        passed = r['passed']
        
        if passed:
            total_passed += 1
        
        status = '✅' if passed else '❌'
        print(f"  {symbol:<10} {trades:>7} {wr:>5.1f}% {pf:>5.2f} {avgm:>6.2f}% {mdd:>6.2f}% {sharpe:>6.2f} {sortino:>6.2f} ${long_pnl:>7.2f} ${short_pnl:>7.2f} {status:>5}")
    
    print(f"\n  Total Passed: {total_passed}/{len(results)} ({total_passed/len(results)*100:.0f}%)")
    
    # Show criteria breakdown for failures
    failed = [r for r in results if not r['passed']]
    if failed:
        print(f"\n  {'='*60}")
        print(f"  FAILED STRATEGIES")
        print(f"  {'='*60}")
        for f_result in failed[:3]:
            print(f"\n  {f_result['symbol']}:")
            for name, (passed, val) in f_result['criteria'].items():
                icon = '✅' if passed else '❌'
                print(f"    {icon} {name:<35s} {val}")
    
    # Successful strategies
    passed_results = [r for r in results if r['passed']]
    if passed_results:
        print(f"\n  {'='*60}")
        print(f"  PASSED STRATEGIES")
        print(f"  {'='*60}")
        for p_result in passed_results:
            print(f"\n  {p_result['symbol']} — ✅ ALL PASSED")
            for name, (passed, val) in p_result['criteria'].items():
                print(f"    ✅ {name:<35s} {val}")


def main():
    results = []
    print(f"\n{'='*80}")
    print(f"  ML BACKTEST SUITE — 8 Pairs (15m Data + Model Predictions)")
    print(f"{'='*80}")
    
    for i, symbol in enumerate(SYMBOLS):
        print(f"\n  [{i+1}/{len(SYMBOLS)}] {symbol}...", end=" ")
        
        result = run_ml_backtest_15m(
            symbol,
            confidence_threshold=0.60,
            cooldown_candles=3,
            stop_loss_pct=1.2,
            take_profit_pct=2.0,
            initial_capital=INITIAL_CAPITAL,
            commission_rate=TAKER_FEE,
            slippage_rate=SLIPPAGE,
            max_position_pct=0.10,
            warmup_bars=100,
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
    
    print_results(results)
    
    # Save to JSON
    import json
    output = []
    for r in results:
        output.append({
            'symbol': r['symbol'], 'passed': r['passed'],
            'trade_count': r['trade_count'],
            'metrics': {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                       for k, v in r['metrics'].items()},
        })
    
    with open('data/ml_backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x))
    
    print(f"\n  Results saved: data/ml_backtest_results.json")


if __name__ == "__main__":
    main()
