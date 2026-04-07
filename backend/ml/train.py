from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from backend.ml.model import TARGET_COLUMN


MLFLOW_TRACKING_URI = f"file://{Path('mlruns').resolve()}"
EXPERIMENT_NAME = "power-demand-forecast"
FEATURE_COLUMNS = ["hour", "dayofweek", "month", "day", "lag_24", "rolling_24"]


@dataclass
class TrainingResult:
    model: LinearRegression
    rmse: float
    rmse_percent: float
    holdout_frame: pd.DataFrame
    run_id: str
    model_uri: str


def train_and_log_model(hourly_df: pd.DataFrame, source_name: str) -> TrainingResult:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    holdout_size = min(24, max(1, len(hourly_df) // 10))
    train_df = hourly_df.iloc[:-holdout_size].copy()
    holdout_df = hourly_df.iloc[-holdout_size:].copy()

    if train_df.empty or holdout_df.empty:
        raise ValueError("모델 학습에 필요한 데이터가 부족합니다.")

    model = LinearRegression()
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    predictions = model.predict(holdout_df[FEATURE_COLUMNS])
    rmse = mean_squared_error(holdout_df[TARGET_COLUMN], predictions, squared=False)
    baseline = max(holdout_df[TARGET_COLUMN].mean(), 1.0)
    rmse_percent = round((rmse / baseline) * 100, 2)

    with mlflow.start_run(run_name=f"train-{source_name}") as run:
        mlflow.log_param("source_name", source_name)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("holdout_rows", len(holdout_df))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("rmse_percent", float(rmse_percent))
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

    holdout_df = holdout_df.assign(prediction=predictions)
    return TrainingResult(
        model=model,
        rmse=float(rmse),
        rmse_percent=float(rmse_percent),
        holdout_frame=holdout_df,
        run_id=run_id,
        model_uri=model_uri,
    )
