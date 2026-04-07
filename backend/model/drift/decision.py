"""
Drift status classification and drift log persistence.

Status logic
------------
Given baseline_rmse (computed from the healthy reference period):

  RMSE_168h > baseline * 1.2  →  "drift"    (sustained degradation confirmed)
  RMSE_24h  > baseline * 1.2  →  "warning"  (early signal, may be transient)
  otherwise                   →  "normal"

The long-term check gates the hard "drift" decision to prevent reacting to
isolated spikes (e.g., a single holiday anomaly). The short-term check
surfaces early warnings that an operator can monitor before committing to
a full retrain.

Drift log format (logs/drift_log.csv)
--------------------------------------
  timestamp, RMSE_24h, RMSE_168h, baseline_rmse, status
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DRIFT_MULTIPLIER = 1.2   # RMSE must exceed baseline * this to trigger alert

LOG_DIR  = Path(__file__).parent.parent / "logs"
LOG_PATH = LOG_DIR / "drift_log.csv"
LOG_COLS = ["timestamp", "RMSE_24h", "RMSE_168h", "baseline_rmse", "status"]


def classify_status(rmse_24h: float, rmse_168h: float, baseline: float) -> str:
    """Return 'drift', 'warning', or 'normal'."""
    threshold = baseline * DRIFT_MULTIPLIER
    if rmse_168h > threshold:
        return "drift"
    if rmse_24h > threshold:
        return "warning"
    return "normal"


def build_log_rows(blocks: pd.DataFrame, baseline_rmse: float) -> pd.DataFrame:
    """
    Attach a status column to the per-24h-block DataFrame.

    Parameters
    ----------
    blocks       : DataFrame from evaluator.evaluate(), indexed by timestamp,
                   columns [rmse_24h, rmse_168h]
    baseline_rmse: scalar reference RMSE from the healthy period

    Returns
    -------
    DataFrame ready to append to drift_log.csv.
    """
    log = blocks.copy()
    log["baseline_rmse"] = round(baseline_rmse, 2)
    log["status"] = log.apply(
        lambda r: classify_status(r["rmse_24h"], r["rmse_168h"], baseline_rmse),
        axis=1,
    )
    log = log.rename(columns={"rmse_24h": "RMSE_24h", "rmse_168h": "RMSE_168h"})
    log.index.name = "timestamp"
    log = log.reset_index()[LOG_COLS]
    return log


def append_drift_log(log_rows: pd.DataFrame) -> None:
    """Append rows to logs/drift_log.csv, creating the file if needed."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    header = not LOG_PATH.exists()
    log_rows.to_csv(LOG_PATH, mode="a", index=False, header=header)


def has_drift(log_rows: pd.DataFrame) -> bool:
    """Return True if any block in this run is classified as 'drift'."""
    return (log_rows["status"] == "drift").any()


def summary(log_rows: pd.DataFrame) -> dict:
    counts = log_rows["status"].value_counts().to_dict()
    return {
        "total_blocks":  len(log_rows),
        "normal":        counts.get("normal",  0),
        "warning":       counts.get("warning", 0),
        "drift":         counts.get("drift",   0),
        "overall_status": "drift" if has_drift(log_rows) else
                          ("warning" if counts.get("warning", 0) > 0 else "normal"),
    }
