#!/usr/bin/env python3
"""Compare backtest vs live trades for last 6 hours, trade-by-trade."""
import sys; sys.path.insert(0, '.')
import json, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import COOLDOWN_BARS, TAKER_FEE, SLIPPAGE, INITIAL_CAPITAL, LIVE_SYMBOLS
warnings.filterwarnings('ignore')
WARMUP_BARS = 200

THR = 0.50
COOLDOWN = 2
HOLD = 10

def sizing_fn(proba):
    if proba >= 0.80: return 0.35
    if proba >= 0.75: return 0.28
    if proba >= 0.70: return 0.22
    if proba >= 0.65: return 0.18
    return 0.15

class Precomputed:
    def __init__(self, symbol):
        self.symbol = symbol
    def load(self):
        gen = SignalGenerator(self.symbol)
        cached = gen._load_models(self.symbol)
        if not cached: return False
        mg = cached['groups']
        df = gen._ensure_ohlcv_data(self.symbol)
        if df is None or len(df) < 1000: return False
        feat_df = compute_5m_features_5tf(df, for_inference=True)
        if feat_df is None: return False
        self.gen = gen
        self.cached = cached
        self.df_5m = df
        self.feat_df = feat_df
        self.close_arr = df.loc[feat_df.index, 'close'].astype(float).values
        self.feat_index = feat_df.index
        all_feats = sorted(set(f for _,ms in mg.items() for _,m,mf in ms for f in mf))
        mf_idx = {f:i for i,f in enumerate(all_feats)}
        fc_map = {f:i for i,f in enumerate(feat_df.columns)}
        long_r, short_r = [], []
        for tg, ms in mg.items():
            for _,m,mf in ms:
                avail = [mf_idx[f] for f in mf if f in mf_idx]
                if len(avail)>=5:
                    ref = (m, np.array(avail, np.int32))
                    if tg=='long': long_r.append(ref)
                    elif tg=='short': short_r.append(ref)
        now = datetime.utcnow().replace(second=0,microsecond=0)
        self.now = now
        # Get feature matrix for full range
        feat_np = feat_df.values
        self.all_feats = all_feats
        self.feat_matrix_all = np.zeros((len(feat_df), len(all_feats)), np.float64)
        for name, pos in mf_idx.items():
            if name in fc_map:
                self.feat_matrix_all[:,pos] = np.clip(np.nan_to_num(feat_np[:, fc_map[name]], nan=0.0), -10, 10)
        self.long_r = long_r
        self.short_r = short_r
        return True
    
    def batch_predict(self, start_idx, n_bars):
        """Get model probas for a specific range of bars starting at start_idx."""
        mat = self.feat_matrix_all[start_idx:start_idx+n_bars]
        def batch(refs):
            p = np.zeros(n_bars); nv = np.zeros(n_bars, np.int32)
            for m, fi in refs:
                try:
                    preds = m.predict_proba(mat[:,fi])[:,1]
                    p += preds; nv += 1
                except: pass
            return np.where(nv>0, p/nv, 0.5)
        def cal(probas, side):
            cal_path = Path('data/ml_models') / f'{self.symbol}_{side}_calibrator.json'
            if cal_path.exists():
                try:
                    with open(cal_path) as f: c = json.load(f)
                    z = c['coef'] * probas + c['intercept']
                    return np.clip(1.0/(1.0+np.exp(-z)), 0.0, 1.0)
                except: pass
            return probas
        return cal(batch(self.long_r), 'long'), cal(batch(self.short_r), 'short')
    
    def simulate(self, start_dt, stop_dt, test_hours=6):
        """Simulate over exact time range, return trades with timestamps + probas."""
        feat_df = self.feat_df
        close_arr = self.close_arr
        tr_start = max(WARMUP_BARS, int(np.searchsorted(feat_df.index, start_dt)))
        tr_stop = int(np.searchsorted(feat_df.index, stop_dt))
        if tr_stop <= tr_start:
            return []
        tr_range = list(range(tr_start, tr_stop))
        n = len(tr_range)
        if n == 0: return []
        
        # Get probas for this range
        long_p, short_p = self.batch_predict(tr_start, n)
        
        sigs = np.zeros(n, np.int32)
        lm = long_p >= THR; sm = short_p >= THR
        both = lm & sm; pick = long_p >= short_p
        sigs[both & pick]=1; sigs[both & ~pick]=-1
        sigs[lm & ~sm]=1; sigs[sm & ~lm]=-1
        probas = np.where(sigs==1, long_p, np.where(sigs==-1, short_p, np.maximum(long_p, short_p)))
        
        cap=5000.0;peak=5000.0
        pos=0;ep=0.0;eq=0.0;hr=0;cool=0;eproba=0.0;ps=0;pproba=0.0
        bt_trades=[]
        for ii, bi in enumerate(tr_range):
            price=float(close_arr[bi])
            bar_ts = feat_df.index[bi]
            hr=max(0,hr-1);cool=max(0,cool-1)
            if pos!=0 and hr<=0:
                xp=price*(1-SLIPPAGE) if pos==1 else price*(1+SLIPPAGE)
                raw=eq*(xp-ep) if pos==1 else eq*(ep-xp)
                net=raw-xp*eq*TAKER_FEE
                cap+=net;peak=max(peak,cap)
                bt_trades.append({
                    'symbol': self.symbol,
                    'direction': 'LONG' if pos==1 else 'SHORT',
                    'entry_proba': eproba,
                    'pnl': net,
                    'entry_time': entry_ts.isoformat() if entry_ts else None,
                    'exit_time': bar_ts.isoformat(),
                    'entry_price': entry_price_bt,
                    'exit_price': float(xp),
                })
                pos=0;cool=COOLDOWN
            if ps!=0 and pos==0 and cool<=0:
                slip=(1+SLIPPAGE) if ps==1 else (1-SLIPPAGE)
                ep=price*slip
                pct=sizing_fn(pproba)
                eq=(cap*pct)/ep
                cap-=ep*eq*TAKER_FEE
                peak=max(peak,cap)
                pos=ps;eproba=pproba;hr=HOLD
                entry_ts=bar_ts; entry_price_bt=ep
            ps=sigs[ii];pproba=probas[ii]
        return bt_trades

# Load
symbols = LIVE_SYMBOLS
data = {}
print("Loading...")
for i, sym in enumerate(symbols):
    print(f"  {sym}... ", end='', flush=True)
    pc = Precomputed(sym)
    if pc.load():
        data[sym] = pc
        print("OK")
    else:
        print("SKIP")
print(f"Loaded {len(data)}/{len(symbols)}\n")

# Live trades from DB
import sqlite3
db = sqlite3.connect('data/live_trading.db')
six_h_ago = int(time.time() * 1000) - 6*3600*1000
live_trades = db.execute("""
    SELECT id, symbol, direction, entry_time, exit_time, entry_price, exit_price, pnl_net, entry_proba, exit_reason
    FROM live_trades WHERE entry_time > ? AND exit_reason != 'history_sync'
    ORDER BY entry_time
""", (six_h_ago,)).fetchall()
db.close()

print(f"=== LIVE TRADES (last 6h, no history_sync) ===")
print(f"Total: {len(live_trades)}")
live_wins = sum(1 for t in live_trades if t[7] > 0)
live_pnl = sum(t[7] for t in live_trades)
print(f"WR: {live_wins}/{len(live_trades)} ({live_wins/len(live_trades)*100:.1f}%) PnL: ${live_pnl:.2f}")
print()
for t in live_trades:
    d = 'LONG' if t[2]==1 else 'SHORT'
    et = datetime.fromtimestamp(t[3]/1000).strftime('%H:%M') if t[3] else '?'
    xt = datetime.fromtimestamp(t[4]/1000).strftime('%H:%M') if t[4] else 'OPEN'
    print(f"  {t[1]:<12} {d} entry=${t[5]:.4f} exit=${t[6]:.4f} PnL=${t[7]:+.2f} proba={t[8]:.4f} [{et}->{xt}]")

# Now run backtest for exact same period
print(f"\n=== BACKTEST (exact same 6h window) ===")
now = datetime.utcnow().replace(second=0, microsecond=0)
start_dt = now - timedelta(hours=6)
print(f"  Window: {start_dt} → {now}")

all_bt_trades = []
for sym in symbols:
    if sym not in data: continue
    bt = data[sym].simulate(start_dt, now)
    all_bt_trades.extend(bt)

bt_wins = sum(1 for t in all_bt_trades if t['pnl'] > 0)
bt_pnl = sum(t['pnl'] for t in all_bt_trades)
print(f"Total BT trades: {len(all_bt_trades)}")
print(f"WR: {bt_wins}/{len(all_bt_trades)} ({bt_wins/len(all_bt_trades)*100:.1f}%) PnL: ${bt_pnl:.2f}")
print()
for t in all_bt_trades:
    print(f"  {t['symbol']:<12} {t['direction']:>5} entry=${t['entry_price']:.4f} exit=${t['exit_price']:.4f} PnL=${t['pnl']:+.2f} proba={t['entry_proba']:.4f} [{t['entry_time']}->{t['exit_time']}]")

# Compare: signal-level (before trade execution)
print(f"\n=== SIGNAL COMPARISON (entry proba distribution) ===")
bt_probas = [t['entry_proba'] for t in all_bt_trades]
live_probas = [t[8] for t in live_trades]

print(f"Backtest entry probas:")
for p in sorted(bt_probas):
    print(f"  {p:.4f}")
print(f"Live entry probas:")
for p in sorted(live_probas):
    print(f"  {p:.4f}")

print(f"\nSummary:")
print(f"  Backtest: {len(all_bt_trades)} trades, avg proba={np.mean(bt_probas):.4f}, WR={bt_wins/len(all_bt_trades)*100:.1f}%, PnL=${bt_pnl:.2f}")
print(f"  Live:     {len(live_trades)} trades, avg proba={np.mean(live_probas):.4f}, WR={live_wins/len(live_trades)*100:.1f}%, PnL=${live_pnl:.2f}")
