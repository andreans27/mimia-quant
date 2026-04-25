"""
P4: Kelly Position Sizer — calculate per-symbol optimal sizing based on historical trade data.

Formula:
  f* = (p * b - q) / b
  where:
    p = win_rate (historical)
    q = 1 - p
    b = avg_win / avg_loss (net odds ratio)

Fractional Kelly: f_frac * f* (default 0.25x for safety)

Usage:
  from src.strategies.kelly_sizer import KellySizer
  sizer = KellySizer()
  positions = sizer.calc_positions(trade_stats)  # from backtest data
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

KELLY_DB = Path("data/kelly_sizing.json")
DEFAULT_FRACTION = 0.25
MAX_FRACTION = 0.5
MIN_POSITION_PCT = 0.05  # 5% min position
MAX_POSITION_PCT = 0.25  # 25% max position (capped for safety)
MIN_TRADES_FOR_KELLY = 30


def kelly_formula(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate full Kelly fraction.
    f* = (p * b - q) / b
    where b = |avg_win / avg_loss|.
    
    Returns:
      f* as fraction of capital (0.0 to 1.0).
      Returns 0.0 if inputs are invalid.
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    
    return max(0.0, min(1.0, kelly))


def half_kelly(win_rate: float, avg_win: float, avg_loss: float,
               fraction: float = DEFAULT_FRACTION) -> float:
    """Return fractional Kelly: fraction * full Kelly."""
    return kelly_formula(win_rate, avg_win, avg_loss) * fraction


class KellySizer:
    """
    Per-symbol Kelly position sizer.
    
    Maintains a JSON database of historical trade stats per symbol.
    On each recalculation, updates Kelly fractions based on most recent trades.
    """
    
    def __init__(self, db_path: str = None, fraction: float = DEFAULT_FRACTION):
        self.db_path = Path(db_path or KELLY_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fraction = fraction
        self.data = self._load()
    
    def _load(self) -> dict:
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception):
                pass
        # Default structure
        return {
            'version': 2,
            'fraction': self.fraction,
            'symbols': {},
        }
    
    def _save(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def update_from_trades(self, symbol: str, trades: list) -> dict:
        """
        Update Kelly stats from a list of trade dicts.
        
        trades: list of dicts with keys:
          'pnl_net' (required), 'direction' (optional)
        
        Returns:
          stats dict with kelly info.
        """
        if not trades or len(trades) < MIN_TRADES_FOR_KELLY:
            return {'status': 'insufficient_data', 'n_trades': len(trades or [])}
        
        df = pd.DataFrame(trades)
        
        # Separate wins and losses
        wins = df[df['pnl_net'] > 0]
        losses = df[df['pnl_net'] < 0]
        
        n_total = len(df)
        n_wins = len(wins)
        n_losses = len(losses)
        
        if n_wins == 0 or n_losses == 0:
            return {'status': 'no_diversity', 'n_trades': n_total,
                    'n_wins': n_wins, 'n_losses': n_losses}
        
        win_rate = n_wins / n_total
        avg_win = float(wins['pnl_net'].mean())
        avg_loss = float(abs(losses['pnl_net'].mean()))  # absolute value
        
        full_kelly = kelly_formula(win_rate, avg_win, avg_loss)
        frac_kelly = full_kelly * self.fraction
        
        # Clip to safe range
        position_pct = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, frac_kelly))
        
        # Also compute per-direction stats if available
        dir_stats = {}
        if 'direction' in df.columns:
            for direction in ['long', 'short']:
                sub = df[df['direction'] == direction]
                if len(sub) >= MIN_TRADES_FOR_KELLY // 2:
                    w = sub[sub['pnl_net'] > 0]
                    l = sub[sub['pnl_net'] < 0]
                    if len(w) > 0 and len(l) > 0:
                        wr = len(w) / len(sub)
                        aw = float(w['pnl_net'].mean())
                        al = float(abs(l['pnl_net'].mean()))
                        fk = kelly_formula(wr, aw, al)
                        dir_stats[direction] = {
                            'win_rate': round(wr, 4),
                            'avg_win': round(aw, 2),
                            'avg_loss': round(al, 2),
                            'full_kelly': round(fk, 4),
                            'n_trades': len(sub),
                        }
        
        stats = {
            'symbol': symbol,
            'n_trades': n_total,
            'win_rate': round(win_rate, 4),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(float(wins['pnl_net'].sum()) / float(abs(losses['pnl_net'].sum())), 4),
            'full_kelly': round(full_kelly, 4),
            'fraction': self.fraction,
            'frac_kelly': round(frac_kelly, 4),
            'position_pct': round(position_pct, 4),
            'direction_stats': dir_stats,
            'updated': str(pd.Timestamp.now()),
        }
        
        self.data['symbols'][symbol] = stats
        self._save()
        return stats
    
    def get_position_pct(self, symbol: str, direction: str = None) -> float:
        """
        Get recommended position size % for a symbol.
        
        Returns fraction of capital (0.0 to 1.0).
        Falls back to DEFAULT_POSITION_PCT if no Kelly data.
        """
        DEFAULT_PCT = 0.15  # default 15% position size
        
        stats = self.data.get('symbols', {}).get(symbol)
        if not stats:
            return DEFAULT_PCT
        
        # If direction-specific data available, use it
        if direction and stats.get('direction_stats', {}).get(direction):
            dir_s = stats['direction_stats'][direction]
            dir_fk = dir_s.get('full_kelly', 0) * self.fraction
            return max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, dir_fk))
        
        # Use aggregated Kelly
        return stats.get('position_pct', DEFAULT_PCT)
    
    def get_all_positions(self, symbols: list = None) -> dict:
        """Get position sizes for all tracked symbols."""
        if symbols is None:
            symbols = list(self.data.get('symbols', {}).keys())
        
        result = {}
        for sym in symbols:
            result[sym] = {
                'position_pct': self.get_position_pct(sym),
                'long_position_pct': self.get_position_pct(sym, 'long'),
                'short_position_pct': self.get_position_pct(sym, 'short'),
            }
        return result
    
    def summary(self) -> str:
        """Return a human-readable summary of all Kelly stats."""
        symbols = self.data.get('symbols', {})
        if not symbols:
            return "  No Kelly data yet."
        
        lines = [f"  Kelly Fraction: {self.fraction}x"]
        lines.append(f"  {'Symbol':<12} {'WR':>6} {'PF':>6} {'FullKelly':>10} {'Pos%':>6} {'Trades':>7}")
        lines.append(f"  {'-'*47}")
        
        for sym, s in sorted(symbols.items()):
            wr = f"{s.get('win_rate', 0)*100:.1f}%"
            pf = f"{s.get('profit_factor', 0):.2f}"
            fk = f"{s.get('full_kelly', 0):.4f}"
            pp = f"{s.get('position_pct', 0)*100:.1f}%"
            nt = s.get('n_trades', 0)
            lines.append(f"  {sym:<12} {wr:>6} {pf:>6} {fk:>10} {pp:>6} {nt:>7}")
        
        return '\n'.join(lines)


# ─── Standalone runner ─────────────────────────────────────────────

def recalculate_from_csv(csv_path: str, fraction: float = DEFAULT_FRACTION) -> KellySizer:
    """
    Recalculate Kelly from a CSV file of trades.
    CSV must have columns: 'symbol', 'pnl_net', 'direction' (optional)
    """
    df = pd.read_csv(csv_path)
    required = {'symbol', 'pnl_net'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}, got {list(df.columns)}")
    
    sizer = KellySizer(fraction=fraction)
    for symbol, grp in df.groupby('symbol'):
        trades = grp.to_dict('records')
        sizer.update_from_trades(symbol, trades)
    return sizer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Kelly Position Sizer')
    parser.add_argument('--from-trades', type=str, help='CSV file with trade data')
    parser.add_argument('--symbol', type=str, help='Single symbol to update')
    parser.add_argument('--fraction', type=float, default=DEFAULT_FRACTION,
                       help=f'Fractional Kelly multiplier (default: {DEFAULT_FRACTION})')
    parser.add_argument('--list', action='store_true', help='List all sized positions')
    args = parser.parse_args()
    
    sizer = KellySizer(fraction=args.fraction)
    
    if args.from_trades:
        sizer = recalculate_from_csv(args.from_trades, args.fraction)
        print(f"✅ Recalculated Kelly from {args.from_trades}")
        print(sizer.summary())
    
    if args.list:
        print(sizer.summary())
    
    if not args.from_trades and not args.list:
        parser.print_help()
