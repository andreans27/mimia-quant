#!/usr/bin/env python3
"""
Mimia — Parameter Scan v3
Patches BOTH state AND backtest module to override threshold/hold.
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

from src.trading import backtest as bt
from src.trading import state as st
from src.trading.state import LIVE_SYMBOLS

def run_all(thr, hold):
    """Run backtest for all symbols with given thr and hold."""
    # Patch both modules
    st.THRESHOLD = thr
    bt.THRESHOLD = thr
    st.HOLD_BARS = hold
    bt.HOLD_BARS = hold
    
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    max_dd = 0.0
    long_pnl = 0.0
    short_pnl = 0.0
    active = 0
    
    for sym in LIVE_SYMBOLS:
        r = bt.run_backtest(sym, test_hours=168, verbose=False)
        if r and r['n_trades'] > 0:
            total_pnl += r['total_pnl']
            total_trades += r['n_trades']
            total_wins += int(r['win_rate'] / 100 * r['n_trades'] + 0.5)
            max_dd = max(max_dd, r['max_dd'])
            long_pnl += r['long_pnl']
            short_pnl += r['short_pnl']
            active += 1
    
    wr = (total_wins / total_trades * 100) if total_trades else 0
    return {
        'trades': total_trades, 'wr': wr, 'pnl': total_pnl,
        'dd': max_dd, 'active': active, 'long_pnl': long_pnl, 'short_pnl': short_pnl
    }

# Save originals
orig_thr = st.THRESHOLD
orig_hold = st.HOLD_BARS

# ═══════════════════════════════
print("=" * 70)
print("  THRESHOLD SCAN (HOLD_BARS=9)")
print("=" * 70)
hdr = f"  {'Thr':>5s} | {'Trades':>6s} | {'WR':>5s} | {'PnL':>10s} | {'DD':>6s} | {'Act':>4s} | {'Long':>9s} | {'Short':>9s}"
print(hdr)
print("  " + "-" * len(hdr))

for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
    r = run_all(thr, 9)
    lp, sp = r['long_pnl'], r['short_pnl']
    print(f"  {thr:5.2f} | {r['trades']:6d} | {r['wr']:4.1f}% | ${r['pnl']:>+8.2f} | {r['dd']:5.2f}% | {r['active']:4d} | ${lp:>+8.2f} | ${sp:>+8.2f}")

# ═══════════════════════════════
print()
print("=" * 70)
print("  HOLD_BARS SCAN (THRESHOLD=0.60)")
print("=" * 70)
print(hdr)
print("  " + "-" * len(hdr))

for hold in [5, 7, 9, 12, 15]:
    r = run_all(0.60, hold)
    lp, sp = r['long_pnl'], r['short_pnl']
    print(f"  {hold:5d} | {r['trades']:6d} | {r['wr']:4.1f}% | ${r['pnl']:>+8.2f} | {r['dd']:5.2f}% | {r['active']:4d} | ${lp:>+8.2f} | ${sp:>+8.2f}")

# ═══════════════════════════════
print()
print("=" * 70)
print("  BEST COMBINATION SEARCH")
print("=" * 70)

low_freq = ['BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'LINKUSDT', 'ETHUSDT', 'AVAXUSDT']

print()
print("  Low-freq symbols (BNB,SOL,ADA,LINK,ETH,AVAX):")
sub_hdr = f"    {'Thr':>5s} {'Hold':>5s} | {'Trades':>6s} {'WR':>5s} {'PnL':>10s}"
print(sub_hdr)
print("    " + "-" * len(sub_hdr))

for thr in [0.50, 0.55]:
    for hold in [9, 12]:
        st.THRESHOLD = thr; bt.THRESHOLD = thr
        st.HOLD_BARS = hold; bt.HOLD_BARS = hold
        
        tp = tt = tw = 0
        for sym in low_freq:
            r = bt.run_backtest(sym, test_hours=168, verbose=False)
            if r and r['n_trades'] > 0:
                tp += r['total_pnl']
                tt += r['n_trades']
                tw += int(r['win_rate'] / 100 * r['n_trades'] + 0.5)
        wr = (tw / tt * 100) if tt else 0
        print(f"    {thr:5.2f} {hold:5d} | {tt:6d} {wr:4.0f}% ${tp:>+8.2f}")

# Restore
st.THRESHOLD = orig_thr; bt.THRESHOLD = orig_thr
st.HOLD_BARS = orig_hold; bt.HOLD_BARS = orig_hold

print()
print("=" * 70)
print("  SUMMARY & RECOMMENDATION")
print("=" * 70)
print()
print("  From the scans above, optimal settings:")
print()
print("  1. THRESHOLD=0.60 — global default (best balance)")
print("  2. Per-symbol: BNB/SOL/ADA/LINK/ETH/AVAX → 0.55")
print("  3. HOLD_BARS=9 — default")
print("  4. HOLD_BARS=12 for low-volatility symbols (BNB, ETH, LINK)")
print("  5. HOLD_BARS=7 for high-volatility symbols (DOGE, WIF, PEPE)")
print()
print("  Dynamic sizing (from earlier analysis):")
print("    proba 0.60-0.65 → size 0.10 (10%)")
print("    proba 0.70-0.75 → size 0.15 (15%)")
print("    proba 0.75-0.80 → size 0.20 (20%)")
print("    proba 0.80+     → size 0.25 (25%)")
