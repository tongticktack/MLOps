"""
Drift metrics and decision logic.
"""

import math

import numpy as np
import pandas as pd


def compute_rmse(errors: pd.Series | np.ndarray) -> float:
    """RMSE = sqrt(mean(error²))"""
    e = np.asarray(errors, dtype=float)
    return math.sqrt(np.mean(e ** 2))


def compute_mae(errors: pd.Series | np.ndarray) -> float:
    e = np.asarray(errors, dtype=float)
    return float(np.mean(np.abs(e)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rolling_rmse(errors: pd.Series, window: int) -> pd.Series:
    """
    Vectorised rolling RMSE over a Series of errors.
    Returns a Series of the same index; first (window-1) values are NaN.
    """
    return errors.pow(2).rolling(window).mean().apply(np.sqrt)


def drift_decision(rmse: float, threshold: float) -> str:
    return "normal" if rmse < threshold else "drift"


def build_report(predictions: pd.DataFrame, threshold: float) -> dict:
    """
    Build the full metrics dict from a predictions DataFrame.

    Parameters
    ----------
    predictions : DataFrame with columns [y_true, y_pred, error]
    threshold   : RMSE threshold for drift/normal classification

    Returns
    -------
    dict compatible with metrics.json output format
    """
    errors  = predictions["error"].values
    y_true  = predictions["y_true"].values
    y_pred  = predictions["y_pred"].values

    rmse   = compute_rmse(errors)
    mae    = compute_mae(errors)
    mape   = compute_mape(y_true, y_pred)
    status = drift_decision(rmse, threshold)

    return {
        "n_predictions": len(predictions),
        "window_size":   168,
        "RMSE":          round(rmse,  4),
        "MAE":           round(mae,   4),
        "MAPE":          round(mape,  4),
        "threshold":     threshold,
        "status":        status,
    }
