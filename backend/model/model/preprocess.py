"""
Feature extraction for the window-based model.

Features produced
-----------------
Lag values         : lag_1, 2, 3, 6, 12, 24, 48, 72, 168  (from power_demand)
Rolling statistics : roll_mean/std over last 24h and 168h
Climate (current)  : 기온, 강수량, 습도, 적설  (when available)
Temporal (cyclic)  : hour_sin/cos, dow_sin/cos, month_sin/cos
Temporal (binary)  : is_weekend, is_holiday, is_october, long_weekend_flag

All features are computed with pandas vectorised operations — no Python loops.
The first WINDOW_SIZE rows are dropped (insufficient lag history).

Climate columns are optional: if the DataFrame does not contain them,
FEATURE_COLS falls back to the lag-only set (for drift detector compatibility).
"""

import holidays
import numpy as np
import pandas as pd

WINDOW_SIZE = 168

LAG_STEPS = [1, 2, 3, 6, 12, 24, 48, 72, 168]

CLIMATE_COLS = ["기온", "강수량", "습도", "적설"]

# Base features (always present)
_BASE_FEATURE_COLS = (
    [f"lag_{l}" for l in LAG_STEPS]
    + ["roll_mean_24h", "roll_std_24h", "roll_mean_168h", "roll_std_168h"]
    + [
        "hour_sin", "hour_cos",
        "dow_sin",  "dow_cos",
        "month_sin", "month_cos",
        "is_weekend", "is_holiday", "is_october", "long_weekend_flag",
    ]
)

# Full feature set including climate
FEATURE_COLS = CLIMATE_COLS + _BASE_FEATURE_COLS

_KR_HOLIDAYS: holidays.HolidayBase | None = None


def _get_kr_holidays() -> holidays.HolidayBase:
    global _KR_HOLIDAYS
    if _KR_HOLIDAYS is None:
        kr = holidays.KR(years=range(2013, 2031))
        for y in range(2013, 2031):
            kr[f"{y}-05-01"] = "근로자의날"
        _KR_HOLIDAYS = kr
    return _KR_HOLIDAYS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised feature matrix construction.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex and column 'power_demand'.
         Optionally contains climate columns: 기온, 강수량, 습도, 적설.
         Must be sorted by time with no gaps (hourly frequency).

    Returns
    -------
    DataFrame of shape (len(df) - WINDOW_SIZE, n_features).
    Index is the target timestamp being predicted.
    """
    has_climate = all(c in df.columns for c in CLIMATE_COLS)
    active_cols = FEATURE_COLS if has_climate else _BASE_FEATURE_COLS

    pw = df["power_demand"]
    feat = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # Climate features (current hour — no leakage, these are known inputs)
    # ------------------------------------------------------------------
    if has_climate:
        feat["기온"]   = df["기온"]
        feat["강수량"] = df["강수량"]
        feat["습도"]   = df["습도"]
        feat["적설"]   = df["적설"]

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------
    for lag in LAG_STEPS:
        feat[f"lag_{lag}"] = pw.shift(lag)

    # ------------------------------------------------------------------
    # Rolling statistics (shift(1): exclude target row itself)
    # ------------------------------------------------------------------
    pw_s = pw.shift(1)
    feat["roll_mean_24h"]  = pw_s.rolling(24).mean()
    feat["roll_std_24h"]   = pw_s.rolling(24).std()
    feat["roll_mean_168h"] = pw_s.rolling(WINDOW_SIZE).mean()
    feat["roll_std_168h"]  = pw_s.rolling(WINDOW_SIZE).std()

    # ------------------------------------------------------------------
    # Temporal cyclic encodings
    # ------------------------------------------------------------------
    feat["hour_sin"]  = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * df.index.hour / 24)
    feat["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feat["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # ------------------------------------------------------------------
    # Calendar binary flags
    # ------------------------------------------------------------------
    kr = _get_kr_holidays()
    date_s = pd.Series(df.index.date, index=df.index)

    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    feat["is_october"] = (df.index.month == 10).astype(int)
    feat["is_holiday"] = date_s.isin(kr).astype(int)
    feat["long_weekend_flag"] = (
        (date_s - pd.Timedelta(days=1)).isin(kr)
        | (date_s + pd.Timedelta(days=1)).isin(kr)
    ).astype(int)

    # Drop first WINDOW_SIZE rows (incomplete lag history)
    feat = feat.iloc[WINDOW_SIZE:][active_cols]
    return feat
