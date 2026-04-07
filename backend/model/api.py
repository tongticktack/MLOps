"""
Power Demand MLOps — Backend API
=================================

Single entry point for backend integration.
All public functions accept plain Python types and return dataclasses
that can be serialised with dataclasses.asdict() or Pydantic models.

Typical usage
-------------
    from api import run_evaluation, get_forecast, get_drift_log, trigger_retrain

    result = run_evaluation("data.csv", baseline_rmse=801.11)
    # result.status  → "normal" | "warning" | "drift"
    # result.retrained → True if retrain was triggered
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from drift.decision import LOG_PATH, append_drift_log, build_log_rows, has_drift, summary
from drift.evaluator import evaluate
from drift.metrics import build_report
from drift.sliding_window import forecast_24h, read_input
from model.load_model import load_model
from retrain.scheduler import can_retrain, time_since_last

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class BlockSummary:
    total: int
    normal: int
    warning: int
    drift: int


@dataclass
class EvaluationResult:
    """Result of one evaluation run (one CSV window)."""
    status: str           # "normal" | "warning" | "drift"
    baseline_rmse: float
    threshold: float
    period_rmse: float
    mae: float
    mape: float
    n_predictions: int
    blocks: BlockSummary
    retrained: bool
    log_appended: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForecastPoint:
    timestamp: str        # ISO-8601
    y_pred: float


@dataclass
class ForecastResult:
    """24-step-ahead auto-regressive forecast."""
    forecast: list[ForecastPoint]
    generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrainResult:
    """Result of a manual or automatic retraining run."""
    success: bool
    cv_rmse: Optional[float]
    model_path: str
    retrained_at: Optional[str]
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftLogEntry:
    timestamp: str
    rmse_24h: float
    rmse_168h: float
    baseline_rmse: float
    status: str           # "normal" | "warning" | "drift"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(
    input_csv: str | Path,
    *,
    baseline_rmse: Optional[float] = None,
    allow_retrain: bool = True,
    model_path: Optional[str | Path] = None,
) -> EvaluationResult:
    """
    Run the full MLOps evaluation pipeline on a single CSV window.

    Parameters
    ----------
    input_csv     : Path to input CSV.
                    Required columns: datetime (hourly), power_demand (MWh).
                    Optional columns: 기온(°C), 강수량(mm), 습도(%), 적설(cm).
    baseline_rmse : Reference RMSE from a known-healthy period.
                    If None, computed from the first 168 predictions internally.
    allow_retrain : If True and drift is detected with cooldown elapsed,
                    retraining is triggered automatically.
    model_path    : Override model .pkl location.

    Returns
    -------
    EvaluationResult dataclass — use .to_dict() for JSON serialisation.
    """
    logger.info("Loading model …")
    model = load_model(Path(model_path) if model_path else None)

    logger.info("Reading input: %s", input_csv)
    df = read_input(input_csv)
    logger.info("Rows: %d  evaluable: %d", len(df), len(df) - 168)

    logger.info("Running rolling evaluation …")
    predictions, blocks, computed_baseline = evaluate(
        model, df, external_baseline=baseline_rmse
    )
    effective_baseline = baseline_rmse if baseline_rmse is not None else computed_baseline
    threshold = effective_baseline * 1.2

    log_rows    = build_log_rows(blocks, effective_baseline)
    run_summary = summary(log_rows)
    append_drift_log(log_rows)

    status = run_summary["overall_status"]
    logger.info(
        "Status: %s  |  normal=%d  warning=%d  drift=%d",
        status.upper(), run_summary["normal"], run_summary["warning"], run_summary["drift"],
    )

    report = build_report(predictions, threshold=threshold)

    retrained = False
    if has_drift(log_rows) and allow_retrain:
        last = time_since_last()
        logger.info("DRIFT detected. Last retrain: %s", last or "never")
        if can_retrain():
            logger.info("Cooldown elapsed — triggering retrain …")
            retrain_result = trigger_retrain()
            retrained = retrain_result.success
        else:
            logger.info("Cooldown active — skipping retrain.")

    return EvaluationResult(
        status=status,
        baseline_rmse=round(effective_baseline, 2),
        threshold=round(threshold, 2),
        period_rmse=report["RMSE"],
        mae=report["MAE"],
        mape=report["MAPE"],
        n_predictions=report["n_predictions"],
        blocks=BlockSummary(
            total=run_summary["total_blocks"],
            normal=run_summary["normal"],
            warning=run_summary["warning"],
            drift=run_summary["drift"],
        ),
        retrained=retrained,
        log_appended=True,
    )


def get_forecast(
    input_csv: str | Path,
    *,
    model_path: Optional[str | Path] = None,
) -> ForecastResult:
    """
    Generate a 24-step-ahead auto-regressive forecast.

    Parameters
    ----------
    input_csv  : Must contain at least 168 rows of power_demand history.
    model_path : Override model .pkl location.

    Returns
    -------
    ForecastResult with 24 hourly predictions starting from the next hour
    after the last timestamp in input_csv.
    """
    model = load_model(Path(model_path) if model_path else None)
    df    = read_input(input_csv)

    forecast_series = forecast_24h(model, df)

    points = [
        ForecastPoint(timestamp=ts.isoformat(), y_pred=round(float(v), 2))
        for ts, v in forecast_series.items()
    ]

    return ForecastResult(
        forecast=points,
        generated_at=datetime.now().isoformat(),
    )


def get_drift_log(
    log_path: str | Path = LOG_PATH,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> list[DriftLogEntry]:
    """
    Read the drift log and return structured entries.

    Parameters
    ----------
    log_path      : Override default log file path.
    start         : ISO date string — filter rows >= start.
    end           : ISO date string — filter rows <= end.
    status_filter : "normal" | "warning" | "drift" — filter by status.

    Returns
    -------
    List of DriftLogEntry (empty list if log does not exist yet).
    """
    path = Path(log_path)
    if not path.exists():
        return []

    df = pd.read_csv(path, parse_dates=["timestamp"])

    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    if status_filter:
        df = df[df["status"] == status_filter]

    return [
        DriftLogEntry(
            timestamp=row.timestamp.isoformat(),
            rmse_24h=round(row.RMSE_24h, 2),
            rmse_168h=round(row.RMSE_168h, 2),
            baseline_rmse=round(row.baseline_rmse, 2),
            status=row.status,
        )
        for row in df.itertuples()
    ]


def trigger_retrain(
    cutoff_dt: Optional[datetime] = None,
    *,
    force: bool = False,
) -> RetrainResult:
    """
    Manually trigger model retraining.

    Parameters
    ----------
    cutoff_dt : Upper bound for training data (defaults to now).
    force     : If True, bypass the 7-day cooldown check.

    Returns
    -------
    RetrainResult — success=False if cooldown is active (and force=False).
    """
    from retrain.retrain import retrain as _retrain
    from retrain.scheduler import STATE_PATH

    if not force and not can_retrain():
        last = time_since_last()
        logger.warning("Retrain blocked by cooldown. Last retrain: %s", last)
        return RetrainResult(
            success=False,
            cv_rmse=None,
            model_path="",
            retrained_at=None,
            message=f"Cooldown active. Last retrain: {last}",
        )

    try:
        model = _retrain(cutoff_dt=cutoff_dt)
        from model.load_model import _MODEL_PATH
        return RetrainResult(
            success=True,
            cv_rmse=None,          # cv_rmse available in logs; not returned by retrain()
            model_path=str(_MODEL_PATH),
            retrained_at=datetime.now().isoformat(),
            message="Retrain completed successfully.",
        )
    except Exception as exc:
        logger.error("Retrain failed: %s", exc)
        return RetrainResult(
            success=False,
            cv_rmse=None,
            model_path="",
            retrained_at=None,
            message=str(exc),
        )


def get_model_status() -> dict:
    """
    Return current model and retrain scheduler state.

    Returns
    -------
    dict with keys: model_exists, model_path, last_retrain, can_retrain_now.
    """
    from model.load_model import _MODEL_PATH

    return {
        "model_exists":    _MODEL_PATH.exists(),
        "model_path":      str(_MODEL_PATH),
        "last_retrain":    time_since_last(),
        "can_retrain_now": can_retrain(),
    }
