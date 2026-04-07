"""
Rolling dual-RMSE evaluator for drift detection.

Sliding window logic
--------------------
For N rows of known (datetime, power_demand):

  Step k: window[k : k+168] → predict step k+168  (1-step ahead, no leakage)

This produces (N - 168) one-step-ahead predictions.
From those predictions we compute two RMSE signals:

  RMSE_24h  : rolling RMSE over the last 24 predictions
              → sensitive to sudden, short-lived spikes
              → triggers "warning" status early

  RMSE_168h : rolling RMSE over the last 168 predictions (7 days)
              → smoothed signal, resistant to single-event noise
              → triggers "drift" status only when degradation is sustained

We sample both metrics every 24 hours to produce one log row per day,
keeping the drift log compact while still capturing intra-day variation.

Why two metrics?
  A single threshold would cause either too many false alarms (if set low)
  or miss real drift (if set high). Using short-term as early warning and
  long-term as confirmation filters noise while catching true distribution shift.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from drift.metrics import rolling_rmse
from drift.sliding_window import run_inference

# How many initial predictions to use as the baseline RMSE reference
BASELINE_STEPS = 168


def evaluate(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    external_baseline: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling one-step-ahead evaluation and attach dual RMSE signals.

    Parameters
    ----------
    model : pretrained XGBRegressor
    df    : validated DataFrame (DatetimeIndex, 'power_demand').
            Must have at least WINDOW_SIZE + BASELINE_STEPS rows.

    Returns
    -------
    predictions : per-hour DataFrame
        columns — y_true, y_pred, error, rmse_24h, rmse_168h
    blocks : per-24h-block DataFrame (sampled from predictions)
        columns — rmse_24h, rmse_168h
        (used by decision.py to assign status and write the drift log)
    """
    # ------------------------------------------------------------------
    # 1. One-step-ahead predictions (vectorised, no leakage)
    # ------------------------------------------------------------------
    predictions = run_inference(model, df)   # columns: y_true, y_pred, error

    # ------------------------------------------------------------------
    # 2. Rolling RMSE signals (fully vectorised via metrics.rolling_rmse)
    # ------------------------------------------------------------------
    predictions["rmse_24h"]  = rolling_rmse(predictions["error"], window=24)
    predictions["rmse_168h"] = rolling_rmse(predictions["error"], window=168)

    # ------------------------------------------------------------------
    # 3. Baseline RMSE — computed from the first BASELINE_STEPS predictions
    #    (treated as the "healthy" reference period)
    # ------------------------------------------------------------------
    if external_baseline is not None:
        baseline_rmse = external_baseline
    else:
        baseline_errors = predictions["error"].iloc[:BASELINE_STEPS]
        baseline_rmse   = float(np.sqrt(np.mean(baseline_errors ** 2)))

    # ------------------------------------------------------------------
    # 4. Sample one row per 24h block for the drift log
    # ------------------------------------------------------------------
    # Take the last row of each 24-hour group so both rolling windows
    # have accumulated the full block before we record them.
    blocks = (
        predictions[["rmse_24h", "rmse_168h"]]
        .dropna()
        .resample("24h")
        .last()
    )

    return predictions, blocks, baseline_rmse
