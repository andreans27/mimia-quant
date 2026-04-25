"""Train missing models for paper trade symbols (APT, FET, TIA, OP, PEPE, SUI, ARB, INJ)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import subprocess

MISSING = ["APTUSDT", "FETUSDT", "TIAUSDT", "OPUSDT",
           "1000PEPEUSDT", "SUIUSDT", "ARBUSDT", "INJUSDT"]

ROOT = Path("/root/projects/mimia-quant")
MODEL_DIR = ROOT / "data/ml_models"

def needs_full_ensemble(sym):
    meta = MODEL_DIR / f"{sym}_ensemble_meta.json"
    if not meta.exists():
        return True
    for seed in [42, 101, 202, 303, 404]:
        if not (MODEL_DIR / f"{sym}_xgb_ens_{seed}.json").exists():
            return True
    return False

def needs_tf_models(sym):
    for tf in ['m15', 'm30', 'h1', 'h4']:
        meta = MODEL_DIR / f"{sym}_{tf}_ensemble_meta.json"
        if not meta.exists():
            return True
        for seed in [42, 101, 202, 303, 404]:
            if not (MODEL_DIR / f"{sym}_{tf}_xgb_ens_{seed}.json").exists():
                return True
    return False

for sym in MISSING:
    if needs_full_ensemble(sym):
        print(f"\n{'='*60}")
        print(f"Training FULL ensemble for {sym}")
        print(f"{'='*60}")
        r = subprocess.run(
            ["python", "-c", f"""
import sys; sys.path.insert(0, '{ROOT}')
from src.training.train_ml_ensemble import train_ensemble_symbol
from src.training.train_tf_specific import train_tf_specific
print('DONE' if meta else 'FAILED')
"""],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600
        )
        print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
        if r.stderr:
            print(f"STDERR: {r.stderr[-500:]}")
    else:
        print(f"  ✓ Full ensemble for {sym} exists, skipping")

    if needs_tf_models(sym):
        for tf_prefix in ['m15', 'm30', 'h1', 'h4']:
            print(f"\n  Training {sym} {tf_prefix}...")
            r = subprocess.run(
                ["python", "-c", f"""
import sys; sys.path.insert(0, '{ROOT}')
from src.training.train_tf_specific import train_tf_specific, load_cached_features, get_tf_features, select_features
import numpy as np; import pandas as pd; import json; import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path

MODEL_DIR = Path('data/ml_models')
SEEDS = [42, 101, 202, 303, 404]
HPARAMS = {{
    42:  {{'max_depth': 6,  'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5}},
    101: {{'max_depth': 5,  'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3}},
    202: {{'max_depth': 7,  'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7}},
    303: {{'max_depth': 4,  'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4}},
    404: {{'max_depth': 8,  'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6}},
}}

meta = train_tf_specific('{sym}', '{tf_prefix}')
print('DONE' if meta else 'FAILED')
"""],
                cwd=str(ROOT), capture_output=True, text=True, timeout=600
            )
            print(r.stdout[-300:] if len(r.stdout) > 300 else r.stdout)
            if r.stderr:
                print(f"STDERR: {r.stderr[-300:]}")
    else:
        print(f"  ✓ TF models for {sym} exist, skipping")

print("\n\n✅ Training complete for missing symbols")
