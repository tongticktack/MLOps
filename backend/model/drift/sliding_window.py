"""
Sliding window inference — no data leakage, fully vectorised.

How it works
------------
Given N rows of (datetime, power_demand):

  Window 0 : rows[0 : 168]  →  predict rows[168]
  Window 1 : rows[1 : 169]  →  predict rows[169]
  ...
  Window k : rows[k : k+168] →  predict rows[k+168]

Total predictions = N - 168.

Each prediction uses ONLY the actual demand values inside its own window
(no future leakage, no auto-regressive substitution of prior predictions).

Implementation
--------------
Feature extraction is fully vectorised via pandas shift() and rolling().
The model then receives the entire feature matrix in one batch call —
no Python loop over individual windows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.preprocess import WINDOW_SIZE, build_features


def read_input(csv_path: str | Path) -> pd.DataFrame:
    """
    Read and validate the user-supplied CSV.

    Expected columns: datetime, power_demand  (case-insensitive, flexible names).
    Returns a DataFrame with DatetimeIndex sorted ascending.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # Accept common column name variants
    col_map: dict[str, str] = {}
    for col in df.columns:
        if col in ("datetime", "date", "timestamp", "time"):
            col_map[col] = "datetime"
        elif col in ("power_demand", "power demand", "power usage (mwh)", "demand", "value", "values"):
            col_map[col] = "power_demand"
        elif col in ("기온(c)", "기온"):
            col_map[col] = "기온"
        elif col in ("강수량(mm)", "강수량"):
            col_map[col] = "강수량"
        elif col in ("습도(%)", "습도"):
            col_map[col] = "습도"
        elif col in ("적설(cm)", "적설"):
            col_map[col] = "적설"
    df = df.rename(columns=col_map)

    if "datetime" not in df.columns or "power_demand" not in df.columns:
        raise ValueError(
            "Input CSV must have columns 'datetime' and 'power_demand'.\n"
            f"Found: {df.columns.tolist()}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df["power_demand"] = pd.to_numeric(df["power_demand"], errors="coerce")

    n_nan = df["power_demand"].isna().sum()
    if n_nan > 0:
        df["power_demand"] = df["power_demand"].interpolate(method="time").bfill()

    if len(df) <= WINDOW_SIZE:
        raise ValueError(
            f"Input has {len(df)} rows — need at least {WINDOW_SIZE + 1} "
            f"to produce any predictions."
        )

    return df


def forecast_24h(model: xgb.XGBRegressor, df: pd.DataFrame) -> pd.Series:
    """
    Auto-regressive 24-step-ahead forecast from the last WINDOW_SIZE rows of df.

    Each step:
      1. Extract features from the current 168-row window
      2. Predict the next hour
      3. Append prediction to window, drop oldest row (slide forward)

    No ground-truth leakage — only actual history is used as the seed;
    predicted values feed subsequent steps.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex and 'power_demand'. Must have >= WINDOW_SIZE rows.

    Returns
    -------
    pd.Series of 24 predicted values, indexed by future timestamps.
    """
    # Seed window: last WINDOW_SIZE actual observations
    window = df.iloc[-WINDOW_SIZE:].copy()

    preds, timestamps = [], []
    for _ in range(24):
        # Need exactly WINDOW_SIZE + 1 rows so build_features returns 1 row
        # Add a placeholder row for the target timestamp
        next_ts = window.index[-1] + pd.Timedelta(hours=1)
        placeholder = pd.DataFrame({"power_demand": [np.nan]}, index=[next_ts])
        extended = pd.concat([window, placeholder])

        X = build_features(extended)          # returns 1 row (the placeholder)
        pred = float(model.predict(X)[0])

        preds.append(pred)
        timestamps.append(next_ts)

        # Slide: replace placeholder with prediction, drop oldest
        new_row = pd.DataFrame({"power_demand": [pred]}, index=[next_ts])
        window = pd.concat([window.iloc[1:], new_row])

    return pd.Series(preds, index=pd.DatetimeIndex(timestamps), name="y_pred")


def run_inference(model: xgb.XGBRegressor, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run sliding-window one-step-ahead prediction on the entire DataFrame.

    Parameters
    ----------
    model : pretrained XGBRegressor
    df    : validated DataFrame (DatetimeIndex, column 'power_demand')

    Returns
    -------
    DataFrame with columns [y_true, y_pred, error] indexed by target timestamp.
    Length = len(df) - WINDOW_SIZE.
    """
    # Build the full feature matrix in one vectorised pass
    X = build_features(df)                          # shape: (N - 168, n_features)
    y_true = df["power_demand"].iloc[WINDOW_SIZE:]   # aligned targets

    # Batch inference — XGBoost handles the whole matrix at once
    y_pred = model.predict(X)

    result = pd.DataFrame(
        {
            "y_true": y_true.values,
            "y_pred": y_pred,
            "error": y_true.values - y_pred,
        },
        index=X.index,
    )
    return result
