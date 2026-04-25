#!/usr/bin/env python3
"""
Backtest script for Mimia Quant.
Fetches historical data from Binance Demo API and runs comprehensive backtests.

Usage:
    python scripts/run_backtest.py [--symbol SYMBOL] [--days DAYS] [--strategy STRATEGY]
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_config
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid import GridStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy
from src.strategies.backtester import Backtester, BacktestConfig, BacktestMetrics, BacktestTrade
from src.utils.binance_client import BinanceRESTClient

# Fee structure (Binance Futures)
TAKER_FEE = 0.0004  # 0.04%
MAKER_FEE = 0.0002  # 0.02%


def fetch_klines(symbol: str, interval: str = "1h", days: int = 30) -> pd.DataFrame:
    """Fetch historical klines from Binance Demo API."""
    client = BinanceRESTClient(testnet=True)
    
    # Calculate timestamps
    from datetime import datetime, timedelta
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    raw = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=1000,
        start_time=start_ms,
        end_time=end_ms
    )
    
    if raw is None or len(raw) == 0:
        raise ValueError(f"No data fetched for {symbol}")
    
    # Standard columns: open_time, open, high, low, close, volume, close_time, ...
    df = pd.DataFrame(raw, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    print(f"  Fetched {len(df)} bars for {symbol} ({df.index[0]} to {df.index[-1]})")
    return df


def run_backtest(
    symbol: str,
    strategy_name: str,
    days: int = 30,
    initial_capital: float = 5000.0
) -> Dict:
    """Run a single backtest and return metrics."""
    
    # Load data
    print(f"\nFetching data for {symbol}...")
    df = fetch_klines(symbol, "1h", days)
    
    # Get strategy config
    config = get_config()
    strategy_cfg = config.get_strategy(strategy_name) or {}
    strategy_cfg['enabled'] = True
    
    # Create strategy
    strategy_classes = {
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'grid': GridStrategy,
        'breakout': BreakoutStrategy,
        'multi_timeframe': MultiTimeframeStrategy,
    }
    
    if strategy_name not in strategy_classes:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy = strategy_classes[strategy_name](name=strategy_name, config=strategy_cfg)
    
    # Configure backtest
    bt_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=TAKER_FEE,
        slippage_rate=0.0005,  # 0.05% slippage
        enable_fractional_shares=False,
        max_position_size=0.1,  # 10% max per position
    )
    
    # Create backtester
    backtester = Backtester(config=bt_config)
    backtester.register_strategy(strategy)
    backtester.load_data(symbol, df)
    
    # Run backtest
    print(f"Running backtest for {strategy_name} on {symbol}...")
    start_date = df.index[0] + timedelta(hours=50)  # Skip warmup period
    end_date = df.index[-1]
    
    metrics_dict = backtester.run(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date
    )
    
    return {
        'metrics': metrics_dict[strategy_name] if strategy_name in metrics_dict else None,
        'trades': backtester.completed_trades,
        'equity_curve': backtester.equity_curve,
        'data': df,
        'strategy': strategy_name,
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
    }


def analyze_results(results: Dict) -> Dict:
    """Analyze backtest results against Mimia's 11 pass criteria."""
    
    metrics = results['metrics']
    trades = results['trades']
    equity_curve = results['equity_curve']
    
    if metrics is None or len(trades) == 0:
        return {'passed': False, 'reason': 'No trades or metrics'}
    
    # Calculate additional metrics
    equity = pd.Series(equity_curve)
    returns = equity.pct_change().dropna()
    
    # Rolling metrics
    if len(returns) > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365) if returns.std() > 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(24 * 365) if len(returns[returns < 0]) > 0 and returns[returns < 0].std() > 0 else 0
    else:
        sharpe = sortino = 0
    
    # Drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min() * 100  # as percentage
    
    # Monthly returns
    if len(equity_curve) > 1:
        months = pd.date_range(start=results['start_date'], end=results['end_date'], freq='D')
        monthly_returns = []
        for i in range(len(equity_curve)):
            if i > 0 and i % 720 == 0:  # ~every 30 days
                monthly_returns.append((equity_curve[i] - equity_curve[i-720]) / equity_curve[i-720] * 100)
    else:
        monthly_returns = []
    
    avg_monthly = np.mean(monthly_returns) if monthly_returns else 0
    
    # Win rate
    winning = sum(1 for t in trades if t.pnl > 0)
    total = len(trades)
    win_rate = winning / total * 100 if total > 0 else 0
    
    # Profit factor
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    # Trade frequency (trades per day)
    days_in_test = (results['end_date'] - results['start_date']).total_seconds() / 86400
    trades_per_day = total / days_in_test if days_in_test > 0 else 0
    
    # Separate long/short performance
    long_trades = [t for t in trades if t.side.value == 'long']
    short_trades = [t for t in trades if t.side.value == 'short']
    long_pnl = sum(t.pnl for t in long_trades)
    short_pnl = sum(t.pnl for t in short_trades)
    
    # Monthly consistency
    monthly_pnl = {}
    for t in trades:
        month_key = t.exit_time.strftime('%Y-%m')
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + t.pnl
    
    profitable_months = sum(1 for v in monthly_pnl.values() if v > 0)
    total_months = len(monthly_pnl)
    monthly_consistency = profitable_months / total_months * 100 if total_months > 0 else 0
    
    # Criteria check
    criteria = {
        'Max Drawdown < 10%': (max_dd > -10, f"{max_dd:.2f}%"),
        'Avg Monthly Return ≥ 10%': (avg_monthly >= 10, f"{avg_monthly:.2f}%"),
        'Win Rate > 70%': (win_rate > 70, f"{win_rate:.1f}%"),
        'Profit Factor > 2.0': (profit_factor > 2.0, f"{profit_factor:.2f}"),
        'Sharpe Ratio > 2.0': (sharpe > 2.0, f"{sharpe:.2f}"),
        'Sortino Ratio > 2.5': (sortino > 2.5, f"{sortino:.2f}"),
        'Trades/Day ≥ 5': (trades_per_day >= 5, f"{trades_per_day:.1f}"),
        'Monthly Consistency ≥ 80%': (monthly_consistency >= 80, f"{monthly_consistency:.0f}%"),
        'Long Side Profitable': (long_pnl > 0, f"${long_pnl:.2f}"),
        'Short Side Profitable': (short_pnl > 0, f"${short_pnl:.2f}"),
        'Trade Count ≥ 300': (total >= 300, f"{total}"),
    }
    
    passed = all(c[0] for c in criteria.values())
    
    return {
        'passed': passed,
        'criteria': criteria,
        'summary': {
            'total_trades': total,
            'winning_trades': winning,
            'losing_trades': total - winning,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'avg_monthly_return': avg_monthly,
            'trades_per_day': trades_per_day,
            'monthly_consistency': monthly_consistency,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'total_pnl': metrics.total_pnl,
            'sharpe': sharpe,
            'sortino': sortino,
        }
    }


def print_report(results: Dict, analysis: Dict):
    """Print a formatted backtest report."""
    
    print("\n" + "=" * 70)
    print(f"  BACKTEST REPORT: {results['strategy'].upper()} on {results['symbol']}")
    print("=" * 70)
    
    s = analysis['summary']
    print(f"\n  Period:        {results['start_date'].date()} to {results['end_date'].date()}")
    print(f"  Initial Cap:   $5,000.00")
    print(f"  Final Equity: ${s['total_pnl'] + 5000:.2f}")
    print(f"  Total P&L:    ${s['total_pnl']:.2f} ({s['total_pnl']/5000*100:.2f}%)")
    
    print(f"\n  ── Trade Statistics ──")
    print(f"  Total Trades:      {s['total_trades']}")
    print(f"  Winning Trades:    {s['winning_trades']}")
    print(f"  Losing Trades:     {s['losing_trades']}")
    print(f"  Win Rate:          {s['win_rate']:.1f}%")
    print(f"  Profit Factor:     {s['profit_factor']:.2f}")
    print(f"  Long P&L:          ${s['long_pnl']:.2f}")
    print(f"  Short P&L:         ${s['short_pnl']:.2f}")
    
    print(f"\n  ── Risk Metrics ──")
    print(f"  Max Drawdown:      {s['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:      {s['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:     {s['sortino_ratio']:.2f}")
    print(f"  Avg Monthly Ret:    {s['avg_monthly_return']:.2f}%")
    print(f"  Monthly Consist:    {s['monthly_consistency']:.0f}%")
    print(f"  Trades/Day:        {s['trades_per_day']:.1f}")
    
    print(f"\n  ── 11 Pass Criteria ──")
    for name, (passed, value) in analysis['criteria'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | {name}: {value}")
    
    print("\n" + "=" * 70)
    if analysis['passed']:
        print("  ✅ ALL CRITERIA PASSED — Ready for paper trading")
    else:
        failed = [k for k, v in analysis['criteria'].items() if not v[0]]
        print(f"  ❌ FAILED: {', '.join(failed)}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run Mimia Quant backtest')
    parser.add_argument('--symbol', '-s', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--days', '-d', type=int, default=30, help='Days of historical data')
    parser.add_argument('--strategy', default='momentum', help='Strategy name')
    parser.add_argument('--capital', type=float, default=5000.0, help='Initial capital')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  MIMIA QUANT — BACKTEST ENGINE")
    print(f"  Symbol: {args.symbol} | Strategy: {args.strategy}")
    print(f"  Period: {args.days} days | Capital: ${args.capital:,.2f}")
    print(f"{'='*70}")
    
    # Run backtest
    try:
        results = run_backtest(
            symbol=args.symbol,
            strategy_name=args.strategy,
            days=args.days,
            initial_capital=args.capital,
        )
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Print report
        print_report(results, analysis)
        
        return 0 if analysis['passed'] else 1
        
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())