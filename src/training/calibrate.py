"""
Platt scaling calibration for XGBoost ensemble predictions.

After training, we fit a logistic regression (Platt scaler) on the
validation set to calibrate the ensemble's raw proba outputs.
This spreads the narrow proba range (~0.47-0.52) to the full [0, 1]
range, enabling proper threshold-based trading with a dead zone.

Usage:
  from src.training.calibrate import calibrate_ensemble
  
  # After training 25 models, calibrate on validation set
  calibrators = calibrate_ensemble(models, X_val, y_val)
  
  # During inference:
  raw_proba = ensemble_predict(models, X)
  calibrated_proba = apply_calibration(raw_proba, calibrators)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from typing import Dict, List, Optional, Tuple, Callable


def calibrate_platt(
    raw_probas: np.ndarray,
    y_true: np.ndarray,
) -> LogisticRegression:
    """
    Fit Platt scaling (logistic regression) on validation predictions.
    
    Maps raw proba -> calibrated proba via:
        calibrated = 1 / (1 + exp(a * raw + b))
    
    This preserves ranking order (monotonic) while spreading probas.
    """
    calibrator = LogisticRegression(C=1e10, solver='lbfgs')  # C large = no regularization
    calibrator.fit(raw_probas.reshape(-1, 1), y_true)
    return calibrator


def calibrate_isotonic(
    raw_probas: np.ndarray,
    y_true: np.ndarray,
) -> IsotonicRegression:
    """
    Fit isotonic regression for calibration (non-parametric).
    Can improve calibration over Platt but more prone to overfitting
    with small validation sets.
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probas, y_true)
    return calibrator


def evaluate_calibration(
    raw_probas: np.ndarray,
    calibrated_probas: np.ndarray,
    y_true: np.ndarray,
    label: str = "",
) -> dict:
    """Compare raw vs calibrated metrics."""
    raw_auc = roc_auc_score(y_true, raw_probas)
    cal_auc = roc_auc_score(y_true, calibrated_probas)
    raw_brier = brier_score_loss(y_true, raw_probas)
    cal_brier = brier_score_loss(y_true, calibrated_probas)
    
    print(f"  {label} Calibration:")
    print(f"    Raw:      AUC={raw_auc:.4f}  Brier={raw_brier:.4f}")
    print(f"    Calibrated: AUC={cal_auc:.4f}  Brier={cal_brier:.4f}")
    print(f"    Raw range:   [{raw_probas.min():.4f}, {raw_probas.max():.4f}]")
    print(f"    Cal range:   [{calibrated_probas.min():.4f}, {calibrated_probas.max():.4f}]")
    
    return {
        'raw_auc': float(raw_auc),
        'cal_auc': float(cal_auc),
        'raw_brier': float(raw_brier),
        'cal_brier': float(cal_brier),
        'raw_range': [float(raw_probas.min()), float(raw_probas.max())],
        'cal_range': [float(calibrated_probas.min()), float(calibrated_probas.max())],
    }


def save_calibrators(
    calibrators: Dict[str, List],
    symbol: str,
    model_dir: Path,
) -> None:
    """Save calibrator parameters to JSON."""
    cal_data = {}
    for tf_group, cal_list in calibrators.items():
        cal_data[tf_group] = []
        for cal in cal_list:
            if isinstance(cal, LogisticRegression):
                cal_data[tf_group].append({
                    'type': 'platt',
                    'coef': float(cal.coef_[0][0]),
                    'intercept': float(cal.intercept_[0]),
                })
            elif isinstance(cal, IsotonicRegression):
                cal_data[tf_group].append({
                    'type': 'isotonic',
                    'thresholds': cal.thresholds_.tolist(),
                    'y': cal.y_.tolist(),
                })
    
    path = model_dir / f"{symbol}_calibrators.json"
    with open(path, 'w') as f:
        json.dump(cal_data, f, indent=2)
    print(f"  💾 Calibrators saved → {path}")


def load_calibrators(
    symbol: str,
    model_dir: Path,
) -> Optional[Dict[str, List]]:
    """Load calibrators from JSON."""
    path = model_dir / f"{symbol}_calibrators.json"
    if not path.exists():
        return None
    
    with open(path) as f:
        cal_data = json.load(f)
    
    # Reconstruct calibrator objects
    calibrators = {}
    for tf_group, cal_list in cal_data.items():
        calibrators[tf_group] = []
        for cal_dict in cal_list:
            if cal_dict['type'] == 'platt':
                cal = LogisticRegression(C=1e10, solver='lbfgs')
                cal.coef_ = np.array([[cal_dict['coef']]])
                cal.intercept_ = np.array([cal_dict['intercept']])
                cal.classes_ = np.array([0, 1])
                calibrators[tf_group].append(cal)
            elif cal_dict['type'] == 'isotonic':
                cal = IsotonicRegression(out_of_bounds='clip')
                cal.thresholds_ = np.array(cal_dict['thresholds'])
                cal.y_ = np.array(cal_dict['y'])
                calibrators[tf_group].append(cal)
    
    return calibrators


def calibrate_ensemble(
    model_groups: Dict[str, List],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    symbol: str,
    model_dir: Path,
    method: str = 'platt',
) -> Dict[str, List]:
    """
    Calibrate all models in the ensemble on validation data.
    
    Each model gets its own calibrator (per-seed calibration).
    This is more flexible than a single ensemble-level calibrator
    and handles different model confidence levels.
    
    Args:
        model_groups: {tf_group: [(seed, model, features)]}
        X_val: Validation features
        y_val: Validation targets
        symbol: For saving
        model_dir: For saving
        method: 'platt' or 'isotonic'
    
    Returns:
        calibrators: {tf_group: [calibrator_objects]}
    """
    calibrators = {}
    
    for tf_group, models in model_groups.items():
        cal_list = []
        for seed, model, features in models:
            # Get raw predictions on validation set
            available = [c for c in features if c in X_val.columns]
            if len(available) < 5:
                continue
            
            X_val_sub = X_val[available].fillna(0).clip(-10, 10).values
            raw_probas = model.predict_proba(X_val_sub)[:, 1]
            
            # Calibrate
            if method == 'platt':
                cal = calibrate_platt(raw_probas, y_val.values)
            else:
                cal = calibrate_isotonic(raw_probas, y_val.values)
            
            cal_list.append(cal)
            
            # Show calibration effect for first model
            if seed == models[0][0]:
                cal_probas = cal.predict_proba(raw_probas.reshape(-1, 1))[:, 1]
                evaluate_calibration(raw_probas, cal_probas, y_val.values, f"{tf_group} seed {seed}")
        
        calibrators[tf_group] = cal_list
    
    save_calibrators(calibrators, symbol, model_dir)
    return calibrators
