"""
Retraining on the most recent 3 months of data.

Retraining trigger flow
-----------------------
1. drift/detector.py detects drift (RMSE_168h > baseline * 1.2)
2. scheduler.can_retrain() confirms ≥ 7 days since last retrain
3. retrain() is called:
   a. Load latest 3 months of demand + climate data
   b. Build sliding-window feature matrix (same preprocessing as original)
   c. Fit XGBoost with RandomizedSearchCV + TimeSeriesSplit
   d. Overwrite model/window_model.pkl
   e. Record retrain timestamp via scheduler.record_retrain()
4. detector reloads the new model and continues

Why 3 months?
  Short enough to capture recent distribution shifts (seasonality, trend).
  Long enough (≈2160 h) to provide sufficient training samples after the
  168h lag-window burn-in is discarded.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from model.preprocess import WINDOW_SIZE, build_features
from model.train import build_training_df, load_climate, load_demand
from retrain.scheduler import record_retrain

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "model" / "window_model.pkl"

RECENT_HOURS = 91 * 24   # 2184h ≈ 3 months

PARAM_DIST = {
    "n_estimators":     [400, 600, 800],
    "learning_rate":    [0.02, 0.05],
    "max_depth":        [4, 5, 6],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 1.5, 2],
}
N_ITER    = 15
CV_SPLITS = 3
SEED      = 42


def retrain(cutoff_dt: datetime | None = None) -> xgb.XGBRegressor:
    """
    Retrain the model on the most recent 3 months of 2025 data.

    Parameters
    ----------
    cutoff_dt : Upper bound for training data (defaults to now).
                Data from [cutoff - 3 months, cutoff] is used.

    Returns
    -------
    Fitted XGBRegressor (also saved to model/window_model.pkl).
    """
    t0 = time.time()
    np.random.seed(SEED)

    cutoff = cutoff_dt or datetime.now()
    logger.info("Retrain triggered at %s", cutoff.strftime("%Y-%m-%d %H:%M"))
    logger.info("Training window: last %dh (~3 months)", RECENT_HOURS)

    # 1. Load full 2025 demand + climate
    logger.info("Loading 2025 demand …")
    demand_full  = load_demand(config.DEMAND_2025_CSV)
    climate_full = load_climate(config.CLIMATE_2025_CSV)

    # 2. Slice to [cutoff - RECENT_HOURS, cutoff], with WINDOW_SIZE lag rows
    cutoff_ts   = pd.Timestamp(cutoff)
    demand_trim = demand_full[demand_full.index <= cutoff_ts]

    needed = RECENT_HOURS + WINDOW_SIZE
    if len(demand_trim) < needed:
        shortfall = needed - len(demand_trim)
        logger.info(
            "2025 data: %dh — supplementing with %dh from 2024 …",
            len(demand_trim), shortfall,
        )
        demand_24  = load_demand(config.DEMAND_2024_CSV)
        climate_24 = load_climate(config.CLIMATE_2024_CSV)
        supplement = build_training_df(demand_24, climate_24).iloc[-shortfall:]
        df = pd.concat([supplement, build_training_df(demand_trim, climate_full)])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        demand_trim = demand_trim.iloc[-needed:]
        df = build_training_df(demand_trim, climate_full)

    logger.info("Training df: %s  NaNs=%d", df.shape, df.isna().sum().sum())

    # 3. Feature matrix
    X = build_features(df)
    y = df["power_demand"].iloc[WINDOW_SIZE:].loc[X.index]
    logger.info("X=%s  y=%s", X.shape, y.shape)

    # 4. Fit
    logger.info("Fitting …")
    search = RandomizedSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED),
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="neg_root_mean_squared_error",
        verbose=0,
        n_jobs=-1,
        random_state=SEED,
    )
    search.fit(X, y)
    best = search.best_estimator_

    logger.info("Best params : %s", search.best_params_)
    logger.info("CV RMSE     : %.2f MWh", -search.best_score_)

    # 5. Save model + record timestamp
    joblib.dump(best, MODEL_PATH)
    record_retrain()
    logger.info("Model saved → %s  (%.1fs)", MODEL_PATH, time.time() - t0)

    return best
