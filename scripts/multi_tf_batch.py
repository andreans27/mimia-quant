#!/usr/bin/env python3
"""
Multi-TF batch: test across diverse tokens (meme, high vol, majors).
"""

import sys, os, warnings, json, time, subprocess
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.getcwd())

SYMBOLS = [
    ("BTCUSDT",    "King — low vol, efficient"),
    ("DOGEUSDT",   "Meme — classic, high vol"),
    ("1000PEPEUSDT", "Meme — extreme vol, recent"),
    ("WIFUSDT",    "Meme — dog-themed, high vol"),
    ("ENAUSDT",    "New gen — high beta"),
    ("INJUSDT",    "Mid-cap — high volatility"),
]

TARGET_BARS = 6  # 30m target

all_results = {}

for symbol, desc in SYMBOLS:
    print(f"\n{'='*70}")
    print(f"📡 {symbol:>12s} — {desc}")
    print(f"{'='*70}")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, 'scripts/multi_tf_experiment.py', symbol],
        capture_output=True, text=True, timeout=300
    )
    elapsed = time.time() - t0
    
    # Parse output for summary
    summary = []
    for line in result.stdout.split('\n'):
        if 'SUMMARY' in line:
            summary.append(line)
        if line.strip().startswith(('long:', 'short:')) and 'AUC' in line:
            summary.append(line)
    
    print(f"  ⏱ {elapsed:.0f}s")
    print('\n'.join(summary[-8:]))
    
    all_results[symbol] = {
        'desc': desc,
        'time_s': round(elapsed, 1),
        'stdout': result.stdout,
    }
    
    # Check for errors
    if result.returncode != 0:
        print(f"  ❌ ERROR: {result.stderr[-500:]}")

# ── Final comparison ──
print(f"\n\n{'='*70}")
print("FINAL COMPARISON — All Tokens")
print('='*70)
print(f"{'Symbol':>12s} {'Desc':>20s} {'L_AUC':>6s} {'L_WR':>5s} {'L_n':>4s} {'S_AUC':>6s} {'S_WR':>5s} {'S_n':>4s}")
print('-' * 65)

for symbol, desc in SYMBOLS:
    r = all_results.get(symbol, {})
    stdout = r.get('stdout', '')
    
    # Parse results
    lines = stdout.split('\n')
    l_auc = l_wr = l_n = '-'
    s_auc = s_wr = s_n = '-'
    
    for i, line in enumerate(lines):
        if 'SUMMARY' in line:
            # Get next 2 lines
            for j in range(i+1, min(i+3, len(lines))):
                if 'long:' in lines[j]:
                    parts = lines[j].split()
                    for k, p in enumerate(parts):
                        if p.startswith('AUC='): l_auc = p.replace('AUC=', '')
                        if p.startswith('WR='): l_wr = p.replace('WR=', '')
                        if p.startswith('(n='): l_n = p.replace('(n=', '').replace(')', '')
                        if p.startswith('n='): l_n = p.split('n=')[-1] if 'n=' in p and '(n=' not in p else l_n
                if 'short:' in lines[j]:
                    parts = lines[j].split()
                    for k, p in enumerate(parts):
                        if p.startswith('AUC='): s_auc = p.replace('AUC=', '')
                        if p.startswith('WR='): s_wr = p.replace('WR=', '')
                        if p.startswith('(n='): s_n = p.replace('(n=', '').replace(')', '')
                        if p.startswith('n='): s_n = p.split('n=')[-1] if 'n=' in p and '(n=' not in p else s_n
    
    print(f"{symbol:>12s} {desc:>20s} {l_auc:>6s} {l_wr:>5s} {l_n:>4s} {s_auc:>6s} {s_wr:>5s} {s_n:>4s}")

# Save
Path('data/multi_tf_batch_results.json').write_text(json.dumps(
    {k: {'desc': v['desc'], 'time_s': v['time_s']} for k, v in all_results.items()},
    indent=2
))
print(f"\n✅ Saved to data/multi_tf_batch_results.txt")
