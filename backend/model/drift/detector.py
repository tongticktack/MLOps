"""
Drift detector — CLI entry point.

Usage
-----
    python drift/detector.py --input data.csv
    python drift/detector.py --input data.csv --baseline-rmse 801.11
    python drift/detector.py --input data.csv --no-retrain

For programmatic (backend) use, import from api.py instead:
    from api import run_evaluation, get_forecast

Full pipeline
-------------
1.  Load pretrained model
2.  Read + validate input CSV
3.  Rolling evaluation  → per-hour predictions + per-24h-block dual RMSE
4.  Drift classification per block  (normal / warning / drift)
5.  Append to logs/drift_log.csv
6.  If drift detected AND cooldown elapsed → retrain + reload model
7.  24h forecast from the end of the input window
8.  Save outputs/predictions.csv, outputs/metrics.json, outputs/forecast_24h.csv
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from drift.decision import append_drift_log, build_log_rows, has_drift, summary
from drift.evaluator import evaluate
from drift.metrics import build_report
from drift.sliding_window import forecast_24h, read_input
from model.load_model import load_model
from retrain.scheduler import can_retrain, time_since_last

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MLOps drift detector — power demand.")
    p.add_argument("--input",          required=True,  help="Path to input CSV")
    p.add_argument("--output-dir",     default="outputs")
    p.add_argument("--model",          default=None,   help="Override model .pkl path")
    p.add_argument("--baseline-rmse",  type=float, default=None,
                   help="External baseline RMSE (skips internal baseline computation)")
    p.add_argument("--no-retrain",     action="store_true",
                   help="Disable automatic retraining even when drift is detected")
    return p.parse_args(argv)


def run_pipeline(
    input_path: str | Path,
    *,
    output_dir: str | Path = "outputs",
    model_path: str | Path | None = None,
    baseline_rmse: float | None = None,
    allow_retrain: bool = True,
) -> dict:
    """
    Execute the full drift-detection pipeline and save outputs to disk.

    Parameters
    ----------
    input_path    : CSV with columns [datetime, power_demand].
    output_dir    : Directory to write predictions.csv, metrics.json, forecast_24h.csv.
    model_path    : Override model .pkl path.
    baseline_rmse : External healthy-period RMSE. If None, derived from first 168 steps.
    allow_retrain : If True, triggers retraining when drift + cooldown conditions are met.

    Returns
    -------
    Metrics dict (same schema as outputs/metrics.json).
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    logger.info("Loading model …")
    model = load_model(Path(model_path) if model_path else None)

    # 2. Read + validate input
    logger.info("Reading: %s", input_path)
    df = read_input(input_path)
    logger.info("Rows: %d  |  evaluable predictions: %d", len(df), len(df) - 168)

    # 3. Rolling evaluation — dual RMSE
    logger.info("Running rolling evaluation …")
    predictions, blocks, computed_baseline = evaluate(
        model, df, external_baseline=baseline_rmse
    )
    effective_baseline = baseline_rmse if baseline_rmse is not None else computed_baseline
    threshold = effective_baseline * 1.2
    logger.info("Baseline RMSE: %,.1f MWh  |  Threshold (×1.2): %,.1f MWh",
                effective_baseline, threshold)

    # 4. Drift classification
    log_rows    = build_log_rows(blocks, effective_baseline)
    run_summary = summary(log_rows)
    logger.info(
        "Blocks: %d  |  normal=%d  warning=%d  drift=%d",
        run_summary["total_blocks"], run_summary["normal"],
        run_summary["warning"], run_summary["drift"],
    )

    # 5. Append drift log
    append_drift_log(log_rows)
    logger.info("Drift log updated → logs/drift_log.csv")

    # 6. Conditional retraining
    retrained = False
    if has_drift(log_rows) and allow_retrain:
        last = time_since_last()
        logger.warning("DRIFT detected! Last retrain: %s", last or "never")
        if can_retrain():
            logger.info("Cooldown elapsed — triggering retrain …")
            from retrain.retrain import retrain
            model = retrain()
            retrained = True
            logger.info("Retrain complete — model reloaded.")
        else:
            logger.info("Cooldown active — skipping retrain.")
    elif has_drift(log_rows) and not allow_retrain:
        logger.warning("DRIFT detected — retraining disabled.")

    # 7. 24h forecast
    logger.info("Generating 24h forecast …")
    forecast = forecast_24h(model, df)
    forecast_df = forecast.to_frame("y_pred")
    forecast_df.index.name = "timestamp"
    forecast_df.to_csv(output_dir / "forecast_24h.csv")

    # 8. Save outputs
    predictions.index.name = "timestamp"
    predictions.to_csv(output_dir / "predictions.csv")

    report = build_report(predictions, threshold=threshold)
    report.update({
        "baseline_rmse":  round(effective_baseline, 2),
        "threshold":      round(threshold, 2),
        "overall_status": run_summary["overall_status"],
        "retrained":      retrained,
        **{k: run_summary[k] for k in ("total_blocks", "normal", "warning", "drift")},
    })
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t0
    logger.info(
        "Done in %.1fs  |  status=%s  retrained=%s",
        elapsed, run_summary["overall_status"].upper(), retrained,
    )
    logger.info("predictions.csv  → %s", output_dir / "predictions.csv")
    logger.info("forecast_24h.csv → %s", output_dir / "forecast_24h.csv")
    logger.info("metrics.json     → %s", output_dir / "metrics.json")

    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        model_path=args.model,
        baseline_rmse=args.baseline_rmse,
        allow_retrain=not args.no_retrain,
    )


if __name__ == "__main__":
    main()
