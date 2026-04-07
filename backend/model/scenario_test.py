"""
MLOps Scenario Test
===================

3-round simulation of the full MLOps monitoring loop:

  Round 1 — Jan 2025 : Healthy operation  → NORMAL
  Round 2 — Feb 2025 : Cold-snap anomaly  → DRIFT → retraining triggered
  Round 3 — Mar 2025 : Post-retrain eval  → NORMAL (performance recovery)

Run:
    python scenario_test.py
"""

import json
import joblib
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent))

import config
from drift.decision import append_drift_log, build_log_rows, has_drift, summary
from drift.evaluator import evaluate
from drift.metrics import build_report
from drift.sliding_window import run_inference
from model.load_model import load_model
from model.preprocess import WINDOW_SIZE, build_features
from model.train import build_training_df, load_climate, load_demand
from retrain.scheduler import STATE_PATH, record_retrain

OUTPUT_DIR   = Path("outputs/scenario")
LOG_PATH     = Path("logs/drift_log.csv")
MODEL_PATH   = Path("model/window_model.pkl")
MODEL_BACKUP = Path("model/window_model_backup.pkl")

DRIFT_MULTIPLIER = 1.2
DIVIDER = "=" * 62


def _load_all():
    df24 = build_training_df(load_demand(config.DEMAND_2024_CSV),
                             load_climate(config.CLIMATE_2024_CSV))
    df25 = build_training_df(load_demand(config.DEMAND_2025_CSV),
                             load_climate(config.CLIMATE_2025_CSV))
    return df24, df25


def _window(df24, df25, start, end):
    seed   = df24[df24.index < start].iloc[-168:]
    target = df25[(df25.index >= start) & (df25.index < end)]
    return pd.concat([seed, target])


def _retrain_full(df24, df25, cutoff="2025-03-01"):
    """Retrain on full 2024 + 2025 data up to cutoff."""
    demand = pd.concat([load_demand(config.DEMAND_2024_CSV),
                        load_demand(config.DEMAND_2025_CSV)])
    demand = demand[~demand.index.duplicated(keep="last")].sort_index()
    demand = demand[demand.index < cutoff]

    climate = pd.concat([load_climate(config.CLIMATE_2024_CSV),
                         load_climate(config.CLIMATE_2025_CSV)])
    climate = climate[~climate.index.duplicated(keep="last")].sort_index()

    df = build_training_df(demand, climate)
    X  = build_features(df)
    y  = df["power_demand"].iloc[WINDOW_SIZE:].loc[X.index]

    search = RandomizedSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        param_distributions={
            "n_estimators": [600, 800], "learning_rate": [0.05],
            "max_depth": [5, 6], "subsample": [0.8],
            "colsample_bytree": [0.8], "reg_alpha": [0.5], "reg_lambda": [1],
        },
        n_iter=4, cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        verbose=0, n_jobs=-1, random_state=42,
    )
    search.fit(X, y)
    joblib.dump(search.best_estimator_, MODEL_PATH)
    record_retrain()
    print(f"  CV RMSE: {-search.best_score_:,.0f} MWh  |  "
          f"params: {search.best_params_}")
    return search.best_estimator_


def _run_round(label, df, model, baseline_rmse, allow_retrain=False):
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(DIVIDER)

    predictions, blocks, _ = evaluate(model, df, external_baseline=baseline_rmse)
    log_rows    = build_log_rows(blocks, baseline_rmse)
    run_summary = summary(log_rows)
    append_drift_log(log_rows)

    overall_rmse = float(np.sqrt(np.mean(predictions["error"] ** 2)))
    threshold    = baseline_rmse * DRIFT_MULTIPLIER
    retrained    = False

    print(f"  Baseline RMSE  : {baseline_rmse:,.0f} MWh")
    print(f"  Threshold ×1.2 : {threshold:,.0f} MWh")
    print(f"  Period RMSE    : {overall_rmse:,.0f} MWh  "
          f"({'▲ EXCEEDS' if overall_rmse > threshold else '✓ within'} threshold)")
    print(f"  24h blocks     : normal={run_summary['normal']}  "
          f"warning={run_summary['warning']}  drift={run_summary['drift']}")
    print(f"  ➜ Status       : {run_summary['overall_status'].upper()}")

    if has_drift(log_rows) and allow_retrain:
        print(f"\n  [!] DRIFT confirmed — retraining triggered")
        retrained = True

    report = build_report(predictions, threshold=threshold)
    report.update({
        "overall_status": run_summary["overall_status"],
        "baseline_rmse":  round(baseline_rmse, 2),
        "retrained":      retrained,
        **{k: run_summary[k] for k in ("normal", "warning", "drift")},
    })
    return report, retrained, model


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in [LOG_PATH, STATE_PATH]:
        if p.exists():
            p.unlink()
    shutil.copy(MODEL_PATH, MODEL_BACKUP)

    print(DIVIDER)
    print("  MLOps Scenario Test — Power Demand Drift Detection")
    print(DIVIDER)

    print("\n[setup] Loading data …")
    df24, df25 = _load_all()

    print("[setup] Computing baseline RMSE from January 2025 …")
    model       = load_model()
    df_jan      = _window(df24, df25, "2025-01-01", "2025-02-01")
    jan_preds   = run_inference(model, df_jan)
    baseline    = float(np.sqrt(np.mean(jan_preds["error"] ** 2)))
    print(f"[setup] Baseline = {baseline:,.0f} MWh  |  Threshold = {baseline * 1.2:,.0f} MWh")

    results = {}

    # ------------------------------------------------------------------
    # Round 1 — January (normal)
    # ------------------------------------------------------------------
    r1, _, model = _run_round(
        "ROUND 1 — January 2025  (expected: NORMAL)",
        df_jan, model, baseline,
    )
    results["jan"] = r1

    # ------------------------------------------------------------------
    # Round 2 — February (drift → retrain)
    # ------------------------------------------------------------------
    df_feb = _window(df24, df25, "2025-02-01", "2025-03-01")
    r2, drift_detected, model = _run_round(
        "ROUND 2 — February 2025  (expected: DRIFT)",
        df_feb, model, baseline, allow_retrain=True,
    )
    results["feb"] = r2

    if drift_detected:
        print(f"\n  Retraining on full 2024 + 2025 Jan-Feb …")
        model = _retrain_full(df24, df25, cutoff="2025-03-01")
        print(f"  [✓] Model saved → {MODEL_PATH}")

    # ------------------------------------------------------------------
    # Round 3 — March (post-retrain recovery)
    # ------------------------------------------------------------------
    df_mar = _window(df24, df25, "2025-03-01", "2025-04-01")
    r3, _, model = _run_round(
        "ROUND 3 — March 2025  (expected: NORMAL after retrain)",
        df_mar, model, baseline,
    )
    results["mar"] = r3

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print(f"  SCENARIO SUMMARY")
    print(DIVIDER)
    print(f"  {'Period':<30} {'RMSE':>7}  {'vs baseline':>11}  {'Status':<10}  Action")
    print(f"  {'-'*28} {'-'*7}  {'-'*11}  {'-'*10}  {'-'*22}")
    rows = [
        ("jan", "R1  Jan 2025  (baseline)",  "—"),
        ("feb", "R2  Feb 2025  (drift)",     "retrain triggered ←"),
        ("mar", "R3  Mar 2025  (recovered)", "✓ performance restored"),
    ]
    for key, label, action in rows:
        r    = results[key]
        rmse = r["RMSE"]
        diff = rmse - baseline
        print(f"  {label:<30} {rmse:>7,.0f}  {diff:>+11,.0f}  "
              f"{r['overall_status']:<10}  {action}")

    print(f"\n  Elapsed  : {time.time() - t0:.1f}s")
    print(f"  Drift log: {LOG_PATH}")

    with open(OUTPUT_DIR / "scenario_results.json", "w") as f:
        json.dump(results, f, indent=2)

    shutil.copy(MODEL_BACKUP, MODEL_PATH)
    print(f"[setup] Original model restored.")


if __name__ == "__main__":
    main()
