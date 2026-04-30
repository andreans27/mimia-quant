import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
from src.trading.signals import SignalGenerator
t = 0.50
for sym in ['DOGEUSDT', 'FETUSDT', 'INJUSDT', 'ARBUSDT', 'ENAUSDT']:
    sig = SignalGenerator(sym).generate_signal(sym)
    if sig:
        lp = sig['long_proba']
        sp = sig['short_proba']
        s = sig['signal']
        p = sig['proba']
        if lp >= t and lp >= sp:
            dec = 'LONG'
        elif sp >= t:
            dec = 'SHORT'
        else:
            dec = 'FLAT'
        print(f'sym={sym}: long={lp:.4f} short={sp:.4f} signal={s:>2} proba={p:.4f} -> {dec}')
