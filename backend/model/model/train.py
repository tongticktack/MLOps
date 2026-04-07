"""
One-time training script — 2024 power demand + climate → predict 2025.

Reads:
  - 한국전력거래소_시간별 전국 전력수요량_20241231.csv  (2024 hourly demand, wide format)
  - 24ClimateByHour.csv                              (2024 hourly climate)

Builds sliding-window feature matrix, fits XGBoost via RandomizedSearchCV,
and saves the model to model/window_model.pkl.

Usage
-----
    python model/train.py
"""

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from model.preprocess import WINDOW_SIZE, build_features

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "window_model.pkl"

PARAM_DIST = {
    "n_estimators":     [400, 600, 800, 1000],
    "learning_rate":    [0.01, 0.02, 0.05],
    "max_depth":        [4, 5, 6, 7],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 1.5, 2],
}
N_ITER    = 20
CV_SPLITS = 3
SEED      = 42


def load_demand(path: Path) -> pd.Series:
    """
    Load wide-format demand CSV (날짜, 1시 ~ 24시) → hourly Series.
    The '24시' column (midnight end-of-day) is mapped to 00:00 of the next day.
    """
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=["날짜"])
    df["날짜"] = pd.to_datetime(df["날짜"])

    if "24시" in df.columns:
        next_day = df[["날짜", "24시"]].copy()
        next_day["날짜"] = next_day["날짜"] + pd.Timedelta(days=1)
        next_day = next_day.rename(columns={"24시": "0시"})
        df = df.drop(columns=["24시"])
        df = pd.merge(df, next_day, on="날짜", how="outer")

    melted = df.melt(id_vars="날짜", var_name="hour", value_name="power_demand")
    melted["hour"] = melted["hour"].str.replace("시", "").astype(int)
    melted["datetime"] = melted["날짜"] + pd.to_timedelta(melted["hour"], unit="h")
    demand = (
        melted[["datetime", "power_demand"]]
        .dropna()
        .sort_values("datetime")
        .set_index("datetime")["power_demand"]
    )
    full_idx = pd.date_range(demand.index.min(), demand.index.max(), freq="h")
    demand = demand.reindex(full_idx).interpolate(method="time").bfill()
    return demand


def load_climate(path: Path) -> pd.DataFrame:
    """
    Load hourly climate CSV → DataFrame indexed by datetime.
    Columns: 기온, 강수량, 습도, 적설
    NaN handling: 강수량/적설 → 0 (no-event default), 기온/습도 → interpolate.
    """
    df = pd.read_csv(path, encoding="cp949")
    df["일시"] = pd.to_datetime(df["일시"])
    df = df.set_index("일시").sort_index()
    df = df.rename(columns={
        "기온(°C)":  "기온",
        "강수량(mm)": "강수량",
        "습도(%)":   "습도",
        "적설(cm)":  "적설",
    })
    df = df[["기온", "강수량", "습도", "적설"]]
    df["강수량"] = df["강수량"].fillna(0)
    df["적설"]   = df["적설"].fillna(0)
    df["기온"]   = df["기온"].interpolate(method="time").bfill().ffill()
    df["습도"]   = df["습도"].interpolate(method="time").bfill().ffill()
    return df


def build_training_df(demand: pd.Series, climate: pd.DataFrame) -> pd.DataFrame:
    """Merge demand + climate onto a common complete hourly index."""
    full_idx = pd.date_range(demand.index.min(), demand.index.max(), freq="h")
    df = pd.DataFrame({"power_demand": demand}, index=full_idx)
    df["power_demand"] = df["power_demand"].interpolate(method="time").bfill()
    climate_aligned = climate.reindex(full_idx, method="ffill")
    df = df.join(climate_aligned)
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    t0 = time.time()
    np.random.seed(SEED)

    logger.info("Loading 2024 demand: %s", config.DEMAND_2024_CSV.name)
    demand = load_demand(config.DEMAND_2024_CSV)
    logger.info("Demand: %s ~ %s  (%d rows)", demand.index[0], demand.index[-1], len(demand))

    logger.info("Loading 2024 climate: %s", config.CLIMATE_2024_CSV.name)
    climate = load_climate(config.CLIMATE_2024_CSV)

    df = build_training_df(demand, climate)
    logger.info("Merged shape: %s  NaNs: %d", df.shape, df.isna().sum().sum())

    logger.info("Building feature matrix …")
    X = build_features(df)
    y = df["power_demand"].iloc[WINDOW_SIZE:].loc[X.index]
    logger.info("X: %s  y: %s  features: %d", X.shape, y.shape, len(X.columns))

    logger.info("Fitting RandomizedSearchCV …")
    search = RandomizedSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED),
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="neg_root_mean_squared_error",
        verbose=2,
        n_jobs=-1,
        random_state=SEED,
    )
    search.fit(X, y)

    logger.info("Best params : %s", search.best_params_)
    logger.info("CV RMSE     : %.2f MWh", -search.best_score_)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_PATH)
    logger.info("Model saved → %s  (%.1fs)", MODEL_PATH, time.time() - t0)


if __name__ == "__main__":
    main()
