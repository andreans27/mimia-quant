"""
Compare Live-Aligned Backtest with Live Trade History (Optimized)
=================================================================
For each live trade in the last 24h: reconstruct the feature state at
the entry bar (using available_until), run model inference, and compare
the signal direction with the actual live trade.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlite3
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.trading.state import DB_PATH, get_symbol_threshold
from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf


def load_models_and_data(gen, symbol):
    """Load models and OHLCV data for a symbol, return refs for inference."""
    cached = gen._load_models(symbol)
    if cached is None:
        return None
    df_5m = gen._ensure_ohlcv_data(symbol)
    if df_5m is None or len(df_5m) < 1000:
        return None
    
    model_groups = cached['groups']
    all_model_feats = sorted(set(
        f for _, models in model_groups.items()
        for _, m, mf in models for f in mf
    ))
    mf_to_idx = {f: i for i, f in enumerate(all_model_feats)}
    n_mf = len(all_model_feats)
    
    long_refs = []
    short_refs = []
    for tg, models in model_groups.items():
        for seed, m, mf in models:
            avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
            if len(avail) >= 5:
                arr = np.array(avail, dtype=np.int32)
                if tg == 'long':
                    long_refs.append((m, arr))
                elif tg == 'short':
                    short_refs.append((m, arr))
    
    return {
        'df_5m': df_5m,
        'mf_to_idx': mf_to_idx,
        'n_mf': n_mf,
        'long_refs': long_refs,
        'short_refs': short_refs,
    }


def main():
    print("=" * 70)
    print("  LIVE vs BACKTEST (Live-Aligned) COMPARISON")
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    # Fetch live trades
    cutoff_ms = int((datetime.utcnow() - timedelta(hours=48)).timestamp() * 1000)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
        SELECT symbol, direction, entry_time, entry_price, exit_price, pnl_net, entry_proba
        FROM live_trades
        WHERE entry_time >= ?
        ORDER BY entry_time DESC
    """, (cutoff_ms,))
    trades = c.fetchall()
    conn.close()
    
    print(f"\n📊 Total live trades in last 48h: {len(trades)}")
    if len(trades) == 0:
        print("No live trades found.")
        return

    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t[0]].append(t)
    print(f"  Symbols with trades: {len(by_symbol)}")
    
    total_matched = 0
    total_mismatched = 0
    total_flat = 0
    total_skipped = 0

    for sym in sorted(by_symbol.keys()):
        symbol_trades = by_symbol[sym]
        print(f"\n{'─'*60}")
        print(f"  {sym}: {len(symbol_trades)} live trades")
        
        t0 = time.time()
        gen = SignalGenerator(sym)
        model_data = load_models_and_data(gen, sym)
        
        if model_data is None:
            print(f"  ❌ Failed to load models/data for {sym}")
            total_skipped += len(symbol_trades)
            continue
        
        df_5m = model_data['df_5m']
        mf_to_idx = model_data['mf_to_idx']
        n_mf = model_data['n_mf']
        long_refs = model_data['long_refs']
        short_refs = model_data['short_refs']
        
        def _infer(refs, row_np):
            probas = []
            for model, feat_idx in refs:
                try:
                    preds = model.predict_proba(row_np[feat_idx].reshape(1, -1))[:, 1]
                    probas.append(preds[0])
                except Exception:
                    continue
            return float(np.mean(probas)) if probas else 0.5
        
        def _apply_cal(proba, side):
            cal_path = Path("data/ml_models") / f"{sym}_{side}_calibrator.json"
            if cal_path.exists():
                try:
                    with open(cal_path) as f:
                        cal = json.load(f)
                    z = cal['coef'] * proba + cal['intercept']
                    return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0))
                except Exception:
                    pass
            return proba
        
        sym_matched = 0
        sym_mismatched = 0
        sym_flat = 0
        sym_skipped = 0
        details = []
        trade_bars_cache = {}
        
        for trade in symbol_trades:
            sym_name, direction, entry_ms = trade[:3]
            entry_proba = trade[6]
            
            entry_dt = datetime.fromtimestamp(entry_ms / 1000.0)
            entry_5m = entry_dt.replace(
                minute=(entry_dt.minute // 5) * 5,
                second=0, microsecond=0
            )
            
            # Deferred entry: signal at bar N → executed at bar N+1
            signal_bar_dt = entry_5m - timedelta(minutes=5)
            idx_arr = int(np.searchsorted(df_5m.index, signal_bar_dt))
            if idx_arr >= len(df_5m):
                idx_arr = len(df_5m) - 1
            
            bar_index = idx_arr
            
            if bar_index not in trade_bars_cache:
                try:
                    feat_row = compute_5m_features_5tf(
                        df_5m, for_inference=True, available_until=bar_index
                    )
                except Exception as e:
                    trade_bars_cache[bar_index] = None
                    continue
                
                if feat_row is None or len(feat_row) == 0:
                    trade_bars_cache[bar_index] = None
                else:
                    feat_cols = list(feat_row.columns)
                    row_np = np.zeros(n_mf, dtype=np.float64)
                    for mf_name, mf_pos in mf_to_idx.items():
                        if mf_name in feat_cols:
                            val = feat_row[mf_name].values[0]
                            row_np[mf_pos] = float(np.clip(0.0 if np.isnan(val) else val, -10, 10))
                    
                    long_proba = _apply_cal(_infer(long_refs, row_np), 'long')
                    short_proba = _apply_cal(_infer(short_refs, row_np), 'short')
                    
                    threshold = get_symbol_threshold(sym)
                    if long_proba >= threshold and long_proba >= short_proba:
                        bt_signal = 1
                    elif short_proba >= threshold:
                        bt_signal = -1
                    else:
                        bt_signal = 0
                    
                    trade_bars_cache[bar_index] = {
                        'signal': bt_signal,
                        'long_proba': long_proba,
                        'short_proba': short_proba,
                    }
            
            pred = trade_bars_cache.get(bar_index)
            if pred is None:
                sym_skipped += 1
                continue
            
            live_dir = 1 if direction == 'long' else -1
            bt_dir = pred['signal']
            
            match = (live_dir == bt_dir)
            if match:
                sym_matched += 1
                total_matched += 1
            elif bt_dir == 0:
                sym_flat += 1
                total_flat += 1
            else:
                sym_mismatched += 1
                total_mismatched += 1
            
            details.append((match, bt_dir, live_dir, pred, entry_proba, entry_dt))
        
        elapsed = time.time() - t0
        total_check = sym_matched + sym_mismatched + sym_flat
        rate = (sym_matched / max(1, total_check)) * 100
        print(f"  ⏱ {elapsed:.1f}s | {sym_matched}✅ {sym_mismatched}❌ {sym_flat}⬜ {sym_skipped}⚠️")
        print(f"     Match rate: {rate:.1f}% | checked: {total_check}/{len(symbol_trades)}")
        
        for i in range(min(5, len(details))):
            d = details[i]
            status = "✅" if d[0] else ("⬜" if d[1] == 0 else "❌")
            live_lbl = "LONG" if d[2] == 1 else "SHORT"
            p = d[3]
            bt_lbl = f"LONG({p['long_proba']:.3f})" if d[1] == 1 else (f"SHORT({p['short_proba']:.3f})" if d[1] == -1 else "FLAT")
            print(f"    {status} live={live_lbl} bt={bt_lbl} proba={d[4]:.4f} ts={d[5].strftime('%H:%M')}")

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    total_check = total_matched + total_mismatched + total_flat
    rate = (total_matched / max(1, total_check)) * 100
    print(f"  Matched:    {total_matched} ({rate:.1f}%)")
    print(f"  Mismatched: {total_mismatched}")
    print(f"  Flat (bt=0):{total_flat}")
    print(f"  Skipped:    {total_skipped}")
    print(f"  Checked:    {total_check}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
