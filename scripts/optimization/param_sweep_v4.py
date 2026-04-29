#!/usr/bin/env python3
"""
Parameter Sweep v4 — Systematic optimization of per-symbol parameters.
Pre-computes features once, then varies parameters without recompute.

Optimizes:
  1. GLOBAL_THRESHOLD  (trial values: 0.50, 0.55, 0.60, 0.65, 0.70)
  2. GLOBAL_HOLD_BARS  (trial values: 8, 9, 10)
  3. Per-symbol threshold adjustments for low-freq symbols
  4. Dynamic sizing proba breakpoints
  5. Combined optimal config
"""
import sys; sys.path.insert(0, '.')
import json, time, warnings
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
warnings.filterwarnings('ignore')

from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import (
    THRESHOLD, HOLD_BARS, COOLDOWN_BARS, TAKER_FEE, SLIPPAGE,
    POSITION_PCT, INITIAL_CAPITAL, LIVE_SYMBOLS,
)

WARMUP_BARS = 200
OUT_DIR = Path("data/param_sweep_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Precomputed data per symbol ───────────────────────────────

class PrecomputedSymbol:
    """Cached per-symbol data to avoid recomputing features + inference."""
    def __init__(self, symbol: str):
        self.symbol = symbol

    def load(self, test_hours: int = 168, verbose: bool = False) -> bool:
        """Pre-compute and cache feature matrix, close prices, and model inference."""
        log = (lambda msg: print(f"  {msg}")) if verbose else (lambda _: None)
        symbol = self.symbol

        gen = SignalGenerator(symbol)
        cached = gen._load_models(symbol)
        if cached is None:
            log(f"❌ No models for {symbol}")
            return False
        model_groups = cached['groups']

        df_5m = gen._ensure_ohlcv_data(symbol)
        if df_5m is None or len(df_5m) < 1000:
            log(f"❌ No data for {symbol}")
            return False

        feat_df = compute_5m_features_5tf(df_5m, for_inference=True)
        if feat_df is None or len(feat_df) < 500:
            log(f"❌ Feature computation failed for {symbol}")
            return False

        close_arr = df_5m.loc[feat_df.index, 'close'].astype(float).values

        # Build model references
        all_model_feats = sorted(set(
            f for _, models in model_groups.items()
            for _, m, mf in models for f in mf
        ))
        mf_to_idx = {f: i for i, f in enumerate(all_model_feats)}
        feat_cols = list(feat_df.columns)
        feat_to_col = {f: i for i, f in enumerate(feat_cols)}

        long_refs, short_refs = [], []
        for tg, models in model_groups.items():
            for seed, m, mf in models:
                avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
                if len(avail) >= 5:
                    ref = (m, np.array(avail, dtype=np.int32))
                    if tg == 'long': long_refs.append(ref)
                    elif tg == 'short': short_refs.append(ref)

        # Test range
        now = datetime.utcnow().replace(second=0, microsecond=0)
        start_dt = now - timedelta(hours=test_hours)
        feat_index = feat_df.index
        test_start = max(WARMUP_BARS, int(np.searchsorted(feat_index, start_dt)))
        test_range = list(range(test_start, len(feat_index)))
        n_test = len(test_range)
        if n_test == 0:
            log(f"⚠️ No bars in test window for {symbol}")
            return False

        # Feature matrix
        feat_np = feat_df.values
        feat_slice = feat_np[test_start:test_start + n_test]
        feat_matrix = np.zeros((n_test, n_mf := len(all_model_feats)), dtype=np.float64)
        for mf_name, mf_pos in mf_to_idx.items():
            if mf_name in feat_to_col:
                col_pos = feat_to_col[mf_name]
                feat_matrix[:, mf_pos] = np.clip(
                    np.nan_to_num(feat_slice[:, col_pos], nan=0.0), -10, 10
                )

        # Batch inference
        def _batch_infer(refs, fm):
            probas = np.zeros(n_test, dtype=np.float64)
            nv = np.zeros(n_test, dtype=np.int32)
            for model, feat_idx in refs:
                try:
                    preds = model.predict_proba(fm[:, feat_idx])[:, 1]
                    probas += preds
                    nv += 1
                except Exception:
                    continue
            return np.where(nv > 0, probas / nv, 0.5)

        def _apply_cal(probas, side):
            cal_path = Path("data/ml_models") / f"{symbol}_{side}_calibrator.json"
            if cal_path.exists():
                try:
                    with open(cal_path) as f:
                        cal = json.load(f)
                    z = cal['coef'] * probas + cal['intercept']
                    return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)
                except Exception:
                    pass
            return probas

        self.long_probas = _apply_cal(_batch_infer(long_refs, feat_matrix), 'long')
        self.short_probas = _apply_cal(_batch_infer(short_refs, feat_matrix), 'short')
        self.close_arr = close_arr
        self.feat_index = feat_index
        self.test_range = test_range
        self.n_test = n_test
        self.gen = gen
        return True

    def simulate(self, threshold: float, hold_bars: int,
                 sizing_fn: Callable = None) -> dict:
        """
        Run trade simulation with given parameters.
        sizing_fn(proba) -> position_pct (fraction of capital). None = 0.15 fixed.
        """
        n_test = self.n_test
        close_arr = self.close_arr
        feat_index = self.feat_index
        test_range = self.test_range

        # Generate signals with this threshold
        long_probas = self.long_probas
        short_probas = self.short_probas
        signals = np.zeros(n_test, dtype=np.int32)
        long_mask = long_probas >= threshold
        short_mask = short_probas >= threshold
        both = long_mask & short_mask
        pick_long = long_probas >= short_probas
        signals[both & pick_long] = 1
        signals[both & ~pick_long] = -1
        signals[long_mask & ~short_mask] = 1
        signals[short_mask & ~long_mask] = -1
        probas = np.where(signals == 1, long_probas,
                          np.where(signals == -1, short_probas,
                                   np.maximum(long_probas, short_probas)))

        # Trade simulation
        capital = INITIAL_CAPITAL
        peak_cap = INITIAL_CAPITAL
        pos = 0; ep = 0.0; eq = 0.0; hr = 0; cd = 0; eproba = 0.0
        entry_ts = None; entry_price_bt = 0.0
        ps = 0; pproba = 0.0
        trades = []

        for test_ii, bi in enumerate(test_range):
            price = float(close_arr[bi])
            bar_ts = feat_index[bi]
            hr = max(0, hr - 1)
            cd = max(0, cd - 1)

            if pos != 0 and hr <= 0:
                xp = price * (1 - SLIPPAGE) if pos == 1 else price * (1 + SLIPPAGE)
                raw = eq * (xp - ep) if pos == 1 else eq * (ep - xp)
                xc = xp * eq * TAKER_FEE
                net = raw - xc
                capital += raw - xc
                peak_cap = max(peak_cap, capital)
                trades.append({
                    'direction': 'LONG' if pos == 1 else 'SHORT',
                    'entry_proba': eproba, 'pnl': net,
                    'entry_price': entry_price_bt,
                    'exit_price': float(xp),
                })
                pos = 0; cd = COOLDOWN_BARS

            if ps != 0 and pos == 0 and cd <= 0:
                slip = (1 + SLIPPAGE) if ps == 1 else (1 - SLIPPAGE)
                ep = price * slip
                entry_price_bt = ep
                entry_ts = bar_ts
                pos_pct = sizing_fn(pproba) if sizing_fn else POSITION_PCT
                eq = (capital * pos_pct) / ep
                capital -= ep * eq * TAKER_FEE
                peak_cap = max(peak_cap, capital)
                pos = ps; eproba = pproba; hr = hold_bars

            ps = signals[test_ii]
            pproba = probas[test_ii]

        # Metrics
        nt = len(trades)
        w_ = sum(1 for t in trades if t['pnl'] > 0)
        wr = (w_ / nt * 100) if nt else 0
        tp = sum(t['pnl'] for t in trades)
        gp_ = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gl_ = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        pf = gp_ / gl_ if gl_ else float('inf')
        lp = sum(t['pnl'] for t in trades if t['direction'] == 'LONG')
        sp_ = sum(t['pnl'] for t in trades if t['direction'] == 'SHORT')

        run = INITIAL_CAPITAL; rp = INITIAL_CAPITAL; mdd = 0.0
        for t in trades:
            run += t['pnl']
            rp = max(rp, run)
            mdd = max(mdd, (rp - run) / rp * 100)

        return {
            'n_trades': nt, 'win_rate': wr, 'total_pnl': tp,
            'max_dd': mdd, 'profit_factor': pf,
            'long_pnl': lp, 'short_pnl': sp_,
            'trades': trades,
        }


# ─── Sizing function factories ──────────────────────────────

def fixed_sizing(pct: float = POSITION_PCT):
    """Return a sizing function that always uses the given pct."""
    return lambda proba: pct

def step_sizing(breakpoints: Dict[float, float], default: float = POSITION_PCT):
    """Return a sizing function with proba breakpoints.
    breakpoints: {min_proba: position_pct, ...}
    e.g. {0.75: 0.20, 0.80: 0.25}
    Only increases above default, never decreases.
    """
    def fn(proba: float) -> float:
        result = default
        for min_p, pct in sorted(breakpoints.items()):
            if proba >= min_p:
                result = max(result, pct)
        return result
    return fn


# ─── Sweep functions ─────────────────────────────────────────

def run_sweep(symbol_data: Dict[str, PrecomputedSymbol],
              param_name: str, param_values: list,
              symbols: List[str] = None,
              fixed_threshold: float = THRESHOLD,
              fixed_hold: int = HOLD_BARS,
              sizing_fn: Callable = None,
              verbose: bool = True) -> List[dict]:
    """Run a parameter sweep across one parameter dimension."""
    if symbols is None:
        symbols = list(symbol_data.keys())

    results = []
    for val in param_values:
        t0 = time.time()
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        max_dd = 0.0
        all_profitable = True

        per_sym = {}
        for sym in symbols:
            sd = symbol_data[sym]
            thr = fixed_threshold
            hld = fixed_hold

            # Apply the parameter being swept
            if param_name == 'threshold':
                thr = val
            elif param_name == 'hold_bars':
                hld = val
            # For per-symbol threshold, val is a dict of {symbol: threshold}
            elif param_name == 'sym_threshold' and isinstance(val, dict):
                thr = val.get(sym, fixed_threshold)
            elif param_name == 'sym_hold' and isinstance(val, dict):
                hld = val.get(sym, fixed_hold)

            result = sd.simulate(thr, hld, sizing_fn=sizing_fn)
            per_sym[sym] = result
            total_pnl += result['total_pnl']
            total_trades += result['n_trades']
            total_wins += int(result['n_trades'] * result['win_rate'] / 100)
            max_dd = max(max_dd, result['max_dd'])
            if result['total_pnl'] <= 0:
                all_profitable = False

        agg_wr = (total_wins / total_trades * 100) if total_trades else 0
        elapsed = time.time() - t0

        label = str(val)
        if param_name == 'sym_threshold':
            label = f"sym={val}"
        elif param_name == 'sym_hold':
            label = f"sym={val}"

        entry = {
            'param': param_name,
            'value': val,
            'label': label,
            'n_trades': total_trades,
            'win_rate': round(agg_wr, 1),
            'total_pnl': round(total_pnl, 2),
            'max_dd': round(max_dd, 2),
            'all_profitable': all_profitable,
            'elapsed': round(elapsed, 1),
            'per_symbol': per_sym,
        }
        results.append(entry)

        if verbose:
            print(f"  {label:<35} | {total_trades:>4d} trades | WR {agg_wr:>5.1f}% | "
                  f"PnL ${total_pnl:>+8.2f} | DD {max_dd:.2f}% | "
                  f"{'✅' if all_profitable else '❌'}")

    return results


def format_table(results: List[dict], highlight_best: bool = True):
    """Print a formatted table of results."""
    print(f"\n{'='*90}")
    print(f"  {'Parameter':<35} | {'Trades':>6} | {'WR%':>5} | {'PnL':>12} | {'DD%':>6} | {'All+':>5}")
    print(f"{'='*90}")
    best_idx = max(range(len(results)), key=lambda i: results[i]['total_pnl'])
    for i, r in enumerate(results):
        marker = " ★" if i == best_idx else "  "
        print(f"  {r['label']:<35}{marker} | {r['n_trades']:>6d} | {r['win_rate']:>5.1f} | "
              f"${r['total_pnl']:>+9.2f} | {r['max_dd']:>5.2f}% | "
              f"{'✅' if r['all_profitable'] else '❌'}")


def print_winner(results: List[dict], param_name: str):
    """Print the best config details."""
    best = max(results, key=lambda r: r['total_pnl'])
    print(f"\n🏆 BEST {param_name.upper()}: {best['label']}")
    print(f"   Trades: {best['n_trades']} | WR: {best['win_rate']}% | "
          f"PnL: ${best['total_pnl']:.2f} | DD: {best['max_dd']}% | "
          f"All Profitable: {'✅' if best['all_profitable'] else '❌'}")
    return best


# ─── Main sweep orchestration ─────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        MIMIA PARAMETER SWEEP v4 — 1 Week Backtest         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Phase 0: Load all symbols (pre-compute features + inference)
    print("📦 PRE-COMPUTING features & inference for all symbols...")
    print()

    symbols = LIVE_SYMBOLS
    symbol_data = {}
    t0 = time.time()
    for i, sym in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {sym}... ", end='', flush=True)
        sd = PrecomputedSymbol(sym)
        ok = sd.load(test_hours=168, verbose=False)
        if ok:
            symbol_data[sym] = sd
            print(f"✅ ({sd.n_test} bars)")
        else:
            print("❌ SKIP")
    print(f"\n✅ Loaded {len(symbol_data)}/{len(symbols)} symbols in {time.time()-t0:.1f}s")
    print()

    if len(symbol_data) == 0:
        print("❌ No symbols loaded, aborting.")
        return

    all_results = {}

    # ──────────────────────────────────────────────────────────
    # PHASE 1: Sweep GLOBAL THRESHOLD
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  🔬 PHASE 1: GLOBAL THRESHOLD SWEEP")
    print("="*70)
    print(f"  Fixed: hold_bars={HOLD_BARS}, sizing=fixed {POSITION_PCT}")
    print()

    thr_results = run_sweep(
        symbol_data, 'threshold',
        param_values=[0.50, 0.55, 0.60, 0.65, 0.70],
        fixed_hold=HOLD_BARS, sizing_fn=fixed_sizing(POSITION_PCT)
    )
    format_table(thr_results)
    best_thr = print_winner(thr_results, 'threshold')
    all_results['phase1_threshold'] = thr_results

    # Find best threshold value
    best_thr_val = best_thr['value']

    # ──────────────────────────────────────────────────────────
    # PHASE 2: Sweep GLOBAL HOLD BARS
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  🔬 PHASE 2: GLOBAL HOLD BARS SWEEP (threshold={best_thr_val})")
    print("="*70)
    print(f"  Fixed: threshold={best_thr_val}, sizing=fixed {POSITION_PCT}")
    print()

    hold_results = run_sweep(
        symbol_data, 'hold_bars',
        param_values=[7, 8, 9, 10, 11, 12],
        fixed_threshold=best_thr_val, sizing_fn=fixed_sizing(POSITION_PCT)
    )
    format_table(hold_results)
    best_hold = print_winner(hold_results, 'hold_bars')
    all_results['phase2_hold'] = hold_results
    best_hold_val = best_hold['value']

    # ──────────────────────────────────────────────────────────
    # PHASE 3: Sweep DYNAMIC SIZING VARIANTS
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  🔬 PHASE 3: DYNAMIC SIZING SWEEP (thr={best_thr_val}, hold={best_hold_val})")
    print("="*70)
    print()

    sizing_variants = [
        ("fixed 15%", fixed_sizing(0.15)),
        ("fixed 10%", fixed_sizing(0.10)),
        ("fixed 12%", fixed_sizing(0.12)),
        ("fixed 18%", fixed_sizing(0.18)),
        ("fixed 20%", fixed_sizing(0.20)),
        ("step: ≥0.75→20%, ≥0.80→25%", step_sizing({0.75: 0.20, 0.80: 0.25})),
        ("step: ≥0.70→18%, ≥0.75→22%, ≥0.80→28%", step_sizing({0.70: 0.18, 0.75: 0.22, 0.80: 0.28})),
        ("step: ≥0.70→20%, ≥0.80→30%", step_sizing({0.70: 0.20, 0.80: 0.30})),
        ("aggressive: ≥0.65→18%, ≥0.70→22%, ≥0.75→28%, ≥0.80→35%",
         step_sizing({0.65: 0.18, 0.70: 0.22, 0.75: 0.28, 0.80: 0.35})),
        ("conservative: ≥0.75→17%, ≥0.80→20%", step_sizing({0.75: 0.17, 0.80: 0.20})),
    ]

    sz_results = []
    for label, sfn in sizing_variants:
        t0 = time.time()
        total_pnl = total_trades = total_wins = 0
        max_dd = 0.0
        all_prof = True
        for sym in symbols:
            sd = symbol_data.get(sym)
            if not sd: continue
            r = sd.simulate(best_thr_val, best_hold_val, sizing_fn=sfn)
            total_pnl += r['total_pnl']
            total_trades += r['n_trades']
            total_wins += int(r['n_trades'] * r['win_rate'] / 100)
            max_dd = max(max_dd, r['max_dd'])
            if r['total_pnl'] <= 0: all_prof = False
        agg_wr = (total_wins / total_trades * 100) if total_trades else 0
        sz_results.append({
            'label': label,
            'n_trades': total_trades, 'win_rate': round(agg_wr, 1),
            'total_pnl': round(total_pnl, 2), 'max_dd': round(max_dd, 2),
            'all_profitable': all_prof,
        })
        print(f"  {label:<45} | {total_trades:>4d} | {agg_wr:>5.1f}% | "
              f"${total_pnl:>+9.2f} | {max_dd:>5.2f}% | {'✅' if all_prof else '❌'}")

    best_sz = max(sz_results, key=lambda r: r['total_pnl'])
    print(f"\n🏆 BEST SIZING: {best_sz['label']}")
    all_results['phase3_sizing'] = sz_results

    # ──────────────────────────────────────────────────────────
    # PHASE 4: Per-symbol threshold optimization
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  🔬 PHASE 4: PER-SYMBOL THRESHOLD (base thr={best_thr_val})")
    print("="*70)
    print()

    # Test which low-freq symbols benefit from lower threshold
    low_freq_symbols = ['BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'LINKUSDT', 'ETHUSDT', 'AVAXUSDT']
    thr_configs = []

    # Config 1: all at best global
    thr_configs.append(("all at " + str(best_thr_val), {}))

    # Config 2: all low-freq at 0.50
    thr_configs.append((f"low@0.50, others@{best_thr_val}",
                        {s: 0.50 for s in low_freq_symbols}))

    # Config 3: all low-freq at 0.55
    thr_configs.append((f"low@0.55, others@{best_thr_val}",
                        {s: 0.55 for s in low_freq_symbols}))

    # Config 4: all low-freq at 0.50, high-vol at 0.65
    high_vol = ['WIFUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'INJUSDT']
    sym_map = {s: 0.50 for s in low_freq_symbols}
    sym_map.update({s: 0.65 for s in high_vol})
    thr_configs.append((f"low@0.50, high@0.65, others@{best_thr_val}", sym_map))

    # Config 5: more nuanced - try 0.50 for BNB/SOL/ADA, keep others at best
    thr_configs.append((f"BNB/SOL/ADA@0.50, others@{best_thr_val}",
                        {'BNBUSDT': 0.50, 'SOLUSDT': 0.50, 'ADAUSDT': 0.50}))

    sym_thr_results = run_sweep(
        symbol_data, 'sym_threshold',
        param_values=[cfg[1] for cfg in thr_configs],
        fixed_threshold=best_thr_val, fixed_hold=best_hold_val,
        sizing_fn=step_sizing({0.75: 0.20, 0.80: 0.25})  # use the dynamic sizing from best
    )
    # Add labels
    for i, r in enumerate(sym_thr_results):
        r['label'] = thr_configs[i][0]
    format_table(sym_thr_results)
    best_sym_thr = max(sym_thr_results, key=lambda r: r['total_pnl'])
    all_results['phase4_sym_threshold'] = sym_thr_results

    # ──────────────────────────────────────────────────────────
    # PHASE 5: Per-symbol hold bars
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  🔬 PHASE 5: PER-SYMBOL HOLD BARS")
    print("="*70)
    print()

    hold_configs = []
    # Config 1: all at best global
    hold_configs.append((f"all@{best_hold_val}", {}))
    # Config 2: high vol = best_hold_val - 1, low vol = best_hold_val + 2
    hold_configs.append((f"high@{max(6,best_hold_val-1)}, low@{best_hold_val+2}",
                         {s: max(6, best_hold_val-1) for s in high_vol} |
                         {s: best_hold_val+2 for s in ['BNBUSDT','ETHUSDT','LINKUSDT','SOLUSDT']}))
    # Config 3: high vol = best_hold_val - 2, low vol = best_hold_val + 3
    hold_configs.append((f"high@{max(5,best_hold_val-2)}, low@{best_hold_val+3}",
                         {s: max(5, best_hold_val-2) for s in high_vol} |
                         {s: best_hold_val+3 for s in ['BNBUSDT','ETHUSDT','LINKUSDT','SOLUSDT']}))
    # Config 4: high vol = best_hold_val - 1 only
    hold_configs.append((f"high@{max(6,best_hold_val-1)}, others@{best_hold_val}",
                         {s: max(6, best_hold_val-1) for s in high_vol}))
    # Config 5: more aggressive high vol reduction
    hold_configs.append((f"high@{max(5,best_hold_val-2)}, others@{best_hold_val}",
                         {s: max(5, best_hold_val-2) for s in high_vol}))

    sym_hold_results = run_sweep(
        symbol_data, 'sym_hold',
        param_values=[cfg[1] for cfg in hold_configs],
        fixed_threshold=best_thr_val, fixed_hold=best_hold_val,
        sizing_fn=step_sizing({0.75: 0.20, 0.80: 0.25})
    )
    for i, r in enumerate(sym_hold_results):
        r['label'] = hold_configs[i][0]
    format_table(sym_hold_results)
    best_sym_hold = max(sym_hold_results, key=lambda r: r['total_pnl'])
    all_results['phase5_sym_hold'] = sym_hold_results

    # ──────────────────────────────────────────────────────────
    # PHASE 6: COMBINED BEST CONFIGURATION
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  🏆 PHASE 6: COMBINED BEST CONFIGURATION")
    print("="*70)
    print()

    # Extract the best per-symbol threshold config
    best_sym_thr_config = best_sym_thr['value'] if isinstance(best_sym_thr['value'], dict) else {}
    best_sym_hold_config = best_sym_hold['value'] if isinstance(best_sym_hold['value'], dict) else {}
    best_sizing = best_sz['label']

    # Build the sizing function from the best sizing config
    if "step" in best_sizing:
        if "0.70→18" in best_sizing:
            best_sizing_fn = step_sizing({0.70: 0.18, 0.75: 0.22, 0.80: 0.28})
        elif "0.70→20" in best_sizing and "0.80→30" in best_sizing:
            best_sizing_fn = step_sizing({0.70: 0.20, 0.80: 0.30})
        elif "0.65→18" in best_sizing:
            best_sizing_fn = step_sizing({0.65: 0.18, 0.70: 0.22, 0.75: 0.28, 0.80: 0.35})
        elif "0.75→17" in best_sizing:
            best_sizing_fn = step_sizing({0.75: 0.17, 0.80: 0.20})
        else:
            best_sizing_fn = step_sizing({0.75: 0.20, 0.80: 0.25})
    else:
        best_sizing_fn = fixed_sizing(POSITION_PCT)

    # Run combined best
    t0 = time.time()
    total_pnl = total_trades = total_wins = 0
    max_dd = 0.0
    all_prof = True
    ensemble_per_sym = {}
    for sym in symbols:
        sd = symbol_data.get(sym)
        if not sd: continue
        thr = best_sym_thr_config.get(sym, best_thr_val)
        hld = best_sym_hold_config.get(sym, best_hold_val)
        r = sd.simulate(thr, hld, sizing_fn=best_sizing_fn)
        ensemble_per_sym[sym] = r
        total_pnl += r['total_pnl']
        total_trades += r['n_trades']
        total_wins += int(r['n_trades'] * r['win_rate'] / 100)
        max_dd = max(max_dd, r['max_dd'])
        if r['total_pnl'] <= 0: all_prof = False

    agg_wr = (total_wins / total_trades * 100) if total_trades else 0
    print(f"  ✅ COMBINED BEST CONFIG")
    print(f"     Threshold: {best_thr_val} (sym-adj: {best_sym_thr_config})")
    print(f"     Hold Bars: {best_hold_val} (sym-adj: {best_sym_hold_config})")
    print(f"     Sizing: {best_sizing}")
    print(f"     ─────────────────────────────────────")
    print(f"     Trades: {total_trades} | WR: {agg_wr:.1f}% | "
          f"PnL: ${total_pnl:.2f} | DD: {max_dd:.2f}% | "
          f"All+ : {'✅' if all_prof else '❌'}")
    print(f"     ⏱ {time.time()-t0:.1f}s")

    # ──────────────────────────────────────────────────────────
    # GRAND FINALE: Compare baseline vs final best
    # ──────────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  🎯 GRAND FINALE: BASELINE vs BEST CONFIG")
    print("="*70)
    print()

    # Baseline: best global threshold + hold bars + fixed 15%
    base_thr = best_thr_val
    base_hold = best_hold_val
    base_sizing = fixed_sizing(POSITION_PCT)

    base_trades = base_wins = base_pnl = 0
    best_trades = best_wins = best_pnl = 0
    base_dd = best_dd = 0.0
    base_all_prof = best_all_prof = True

    for sym in symbols:
        sd = symbol_data.get(sym)
        if not sd: continue

        # Baseline
        br = sd.simulate(base_thr, base_hold, sizing_fn=base_sizing)
        base_trades += br['n_trades']
        base_wins += int(br['n_trades'] * br['win_rate'] / 100)
        base_pnl += br['total_pnl']
        base_dd = max(base_dd, br['max_dd'])
        if br['total_pnl'] <= 0: base_all_prof = False

        # Best
        thr = best_sym_thr_config.get(sym, best_thr_val)
        hld = best_sym_hold_config.get(sym, best_hold_val)
        er = sd.simulate(thr, hld, sizing_fn=best_sizing_fn)
        best_trades += er['n_trades']
        best_wins += int(er['n_trades'] * er['win_rate'] / 100)
        best_pnl += er['total_pnl']
        best_dd = max(best_dd, er['max_dd'])
        if er['total_pnl'] <= 0: best_all_prof = False

    base_wr = (base_wins / base_trades * 100) if base_trades else 0
    best_wr = (best_wins / best_trades * 100) if best_trades else 0

    print(f"  {'Metric':<22} {'BASELINE':>14} | {'BEST':>14} | {'Δ':>10}")
    print(f"  {'─'*22} {'─'*14} | {'─'*14} | {'─'*10}")
    print(f"  {'Threshold':<22} {base_thr:>14.2f} | {best_thr_val:>14.2f} |")
    print(f"  {'Hold Bars':<22} {base_hold:>14d} | {best_hold_val:>14d} |")
    print(f"  {'Sizing':<22} {'fixed 15%':>14} | {best_sizing:>14} |")
    print(f"  {'Sym Thresholds':<22} {'no':>14} | {'yes':>14} |")
    print(f"  {'Sym Hold Bars':<22} {'no':>14} | {'yes':>14} |")
    print(f"  {'─'*22} {'─'*14} | {'─'*14} | {'─'*10}")
    print(f"  {'Trades':<22} {base_trades:>14d} | {best_trades:>14d} | "
          f"{best_trades - base_trades:>+10d}")
    print(f"  {'Win Rate':<22} {base_wr:>13.1f}% | {best_wr:>13.1f}% | "
          f"{best_wr - base_wr:>+9.1f}pp")
    print(f"  {'Total PnL':<22} ${base_pnl:>+11.2f} | ${best_pnl:>+11.2f} | "
          f"${best_pnl - base_pnl:>+9.2f}")
    print(f"  {'Max DD':<22} {base_dd:>13.2f}% | {best_dd:>13.2f}% | "
          f"{best_dd - base_dd:>+9.2f}pp")
    print(f"  {'All Profitable':<22} {'✅' if base_all_prof else '❌':>14} | "
          f"{'✅' if best_all_prof else '❌':>14} |")

    delta_pnl = best_pnl - base_pnl
    delta_pct = (delta_pnl / abs(base_pnl) * 100) if base_pnl else 0
    print(f"\n  🏆 RESULT: Best config outperforms baseline by +${delta_pnl:.2f} ({delta_pct:+.1f}%)")

    # Save all results
    summary = {
        'best_threshold': best_thr_val,
        'best_hold_bars': best_hold_val,
        'best_sizing': best_sizing,
        'best_sym_threshold_config': {str(k): v for k, v in best_sym_thr_config.items()},
        'best_sym_hold_config': {str(k): v for k, v in best_sym_hold_config.items()},
        'baseline': {
            'threshold': base_thr, 'hold_bars': base_hold,
            'n_trades': base_trades, 'win_rate': round(base_wr, 1),
            'total_pnl': round(base_pnl, 2), 'max_dd': round(base_dd, 2),
        },
        'best': {
            'threshold_config': f"global={best_thr_val}, sym_adj={best_sym_thr_config}",
            'hold_config': f"global={best_hold_val}, sym_adj={best_sym_hold_config}",
            'sizing': best_sizing,
            'n_trades': best_trades, 'win_rate': round(best_wr, 1),
            'total_pnl': round(best_pnl, 2), 'max_dd': round(best_dd, 2),
        },
        'delta': {
            'n_trades': best_trades - base_trades,
            'win_rate': round(best_wr - base_wr, 1),
            'total_pnl': round(best_pnl - base_pnl, 2),
            'max_dd': round(best_dd - base_dd, 2),
        },
    }
    out_path = OUT_DIR / "param_sweep_v4_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n💾 Results saved to {out_path}")


if __name__ == '__main__':
    main()
