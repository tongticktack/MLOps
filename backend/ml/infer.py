from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import WebSocket

from backend.ml.model import parse_power_csv


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from api import trigger_retrain  # type: ignore  # noqa: E402
from drift.decision import classify_status  # type: ignore  # noqa: E402
from drift.evaluator import evaluate  # type: ignore  # noqa: E402
from drift.sliding_window import forecast_24h  # type: ignore  # noqa: E402
from model.load_model import load_model  # type: ignore  # noqa: E402
from retrain.scheduler import STATE_PATH, can_retrain  # type: ignore  # noqa: E402


STREAM_SESSIONS: dict[str, dict[str, Any]] = {}
DEFAULT_STREAM_DELAY_SECONDS = 0.2


def create_prediction_session(contents: bytes, filename: str) -> dict[str, Any]:
    input_df = parse_power_csv(contents)
    model = load_model()
    predictions, blocks, baseline_rmse = evaluate(model, input_df)
    threshold = baseline_rmse * 1.2
    forecast_series = forecast_24h(model, input_df)

    session_id = uuid.uuid4().hex
    STREAM_SESSIONS[session_id] = {
        "session_id": session_id,
        "filename": filename,
        "created_at": datetime.utcnow().isoformat(),
        "predictions": predictions,
        "blocks": blocks,
        "baseline_rmse": float(baseline_rmse),
        "threshold": float(threshold),
        "forecast": forecast_series,
    }

    return {
        "session_id": session_id,
        "record_count": int(len(predictions)),
        "filename": filename,
        "stream_url": f"ws://localhost:7070/ws/mlops?session_id={session_id}",
        "baseline_rmse": round(float(baseline_rmse), 2),
        "threshold": round(float(threshold), 2),
        "forecast_horizon": int(len(forecast_series)),
    }


async def stream_prediction_session(websocket: WebSocket, session_id: str) -> None:
    session = STREAM_SESSIONS.get(session_id)
    if not session:
        await websocket.send_json({"error": {"code": "SESSION_NOT_FOUND", "message": "유효하지 않은 세션입니다."}})
        await websocket.close(code=4404)
        return

    predictions: pd.DataFrame = session["predictions"]
    threshold = float(session["threshold"])
    baseline_rmse = float(session["baseline_rmse"])
    retrain_count = 0
    retrain_emitted = False

    for index, (timestamp, row) in enumerate(predictions.iterrows(), start=1):
        rmse_24h = _safe_round(row.get("rmse_24h"))
        rmse_168h = _safe_round(row.get("rmse_168h"))
        status = _classify_stream_status(rmse_24h, rmse_168h, baseline_rmse)
        retrain = False
        retrain_reason = None

        # Spec alignment: evaluate drift on 24h block boundaries and retrain once per session when allowed.
        if index % 24 == 0 and status == "drift" and not retrain_emitted and can_retrain():
            retrain_result = trigger_retrain(force=False)
            retrain = retrain_result.success
            retrain_emitted = retrain_result.success
            if retrain_result.success:
                retrain_count += 1
            retrain_reason = retrain_result.message

        payload = {
            "timestamp": timestamp.isoformat(),
            "y": round(float(row["y_true"]), 2),
            "y_hat": round(float(row["y_pred"]), 2),
            "error": round(abs(float(row["error"])), 2),
            "rmse": rmse_24h,
            "rmse_24h": rmse_24h,
            "rmse_168h": rmse_168h,
            "record_count": index,
            "current_prediction_time": timestamp.isoformat(),
            "retrain": retrain,
            "retrain_reason": retrain_reason,
            "threshold": round(threshold, 2),
            "baseline_rmse": round(baseline_rmse, 2),
            "session_id": session_id,
            "pipeline_status": "retraining" if retrain else status,
            "llm_report": _build_stream_report(
                timestamp=timestamp,
                y=float(row["y_true"]),
                y_hat=float(row["y_pred"]),
                rmse_24h=rmse_24h,
                rmse_168h=rmse_168h,
                threshold=threshold,
                status=status,
                retrain=retrain,
                retrain_reason=retrain_reason,
                retrain_count=retrain_count,
            ),
        }
        await websocket.send_json(payload)
        await asyncio.sleep(DEFAULT_STREAM_DELAY_SECONDS)

    await websocket.send_json(
        {
            "event": "stream_complete",
            "session_id": session_id,
            "record_count": int(len(predictions)),
            "retrain_count": retrain_count,
            "last_retrain": _read_last_retrain_timestamp(),
        }
    )


def _build_stream_report(
    timestamp: pd.Timestamp,
    y: float,
    y_hat: float,
    rmse_24h: float | None,
    rmse_168h: float | None,
    threshold: float,
    status: str,
    retrain: bool,
    retrain_reason: str | None,
    retrain_count: int,
) -> str:
    direction = "상향" if y_hat >= y else "하향"
    base = (
        f"{timestamp.strftime('%Y-%m-%d %H:%M')} 시점 실제값은 {y:.2f} MWh, 예측값은 {y_hat:.2f} MWh입니다. "
        f"절대 오차는 {abs(y - y_hat):.2f} MWh이며 단기 RMSE_24h는 {_format_metric(rmse_24h)} MWh, "
        f"장기 RMSE_168h는 {_format_metric(rmse_168h)} MWh입니다. 현재 상태는 {status}이고 예측 방향은 {direction}입니다."
    )
    if retrain:
        return (
            f"{base} 장기 RMSE가 임계값 {threshold:.2f} MWh를 초과해 재학습이 실행되었습니다. "
            f"사유는 {retrain_reason}. 현재까지 총 재학습 횟수는 {retrain_count}회입니다."
        )
    return f"{base} 현재 재학습 임계값은 {threshold:.2f} MWh이며 파이프라인은 해당 상태 기준으로 운영 중입니다."


def _classify_stream_status(rmse_24h: float | None, rmse_168h: float | None, baseline_rmse: float) -> str:
    if rmse_24h is None or rmse_168h is None:
        return "warming_up"
    return classify_status(rmse_24h, rmse_168h, baseline_rmse)


def _safe_round(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _read_last_retrain_timestamp() -> str | None:
    if not STATE_PATH.exists():
        return None
    try:
        state = json.loads(STATE_PATH.read_text())
        return state.get("last_retrain")
    except Exception:
        return None
