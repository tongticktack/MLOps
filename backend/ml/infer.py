from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import WebSocket

from backend.ml.model import TARGET_COLUMN, parse_power_csv


STREAM_SESSIONS: dict[str, dict[str, Any]] = {}
DEFAULT_STREAM_DELAY_SECONDS = 0.35


def create_prediction_session(contents: bytes, filename: str) -> dict[str, Any]:
    parsed = parse_power_csv(contents)
    session_id = uuid.uuid4().hex
    stream_frame = parsed.stream_df.tail(24 * 21).reset_index(drop=True)

    STREAM_SESSIONS[session_id] = {
        "session_id": session_id,
        "filename": filename,
        "created_at": datetime.utcnow().isoformat(),
        "record_count": parsed.original_rows,
        "stream_frame": stream_frame,
    }

    return {
        "session_id": session_id,
        "record_count": parsed.original_rows,
        "filename": filename,
        "stream_url": f"ws://localhost:7070/ws/mlops?session_id={session_id}",
    }


async def stream_prediction_session(websocket: WebSocket, session_id: str, threshold: float) -> None:
    session = STREAM_SESSIONS.get(session_id)
    if not session:
        await websocket.send_json({"error": {"code": "SESSION_NOT_FOUND", "message": "유효하지 않은 세션입니다."}})
        await websocket.close(code=4404)
        return

    stream_frame = session["stream_frame"]
    errors: list[float] = []
    retrain_points = 0
    model_bias = 0.025
    threshold = max(float(threshold), 0.1)

    for index in range(24, len(stream_frame)):
        current_row = stream_frame.iloc[index]
        lag_24 = float(stream_frame.iloc[index - 24][TARGET_COLUMN])
        rolling_24 = float(stream_frame.iloc[max(0, index - 24) : index][TARGET_COLUMN].mean())
        y = float(current_row[TARGET_COLUMN])
        y_hat = _predict_next_value(lag_24=lag_24, rolling_24=rolling_24, step=index, model_bias=model_bias)
        error = abs(y - y_hat)
        errors.append(error)
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        baseline = max(float(stream_frame.iloc[: index + 1][TARGET_COLUMN].mean()), 1.0)
        rmse_percent = round((rmse / baseline) * 100, 2)

        retrain = rmse_percent > threshold
        retrain_reason = None
        if retrain:
            retrain_points += 1
            retrain_reason = f"RMSE {rmse_percent:.2f}% exceeded threshold {threshold:.2f}%"
            model_bias = 0.0
            errors = [abs(y - ((y + lag_24 + rolling_24) / 3))]
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            rmse_percent = round((rmse / baseline) * 100, 2)
        else:
            model_bias = min(model_bias + 0.0008, 0.06)

        await websocket.send_json(
            {
                "timestamp": current_row["datetime"].isoformat(),
                "y": round(y, 2),
                "y_hat": round(y_hat, 2),
                "error": round(error, 2),
                "rmse": rmse_percent,
                "record_count": int(index + 1),
                "current_prediction_time": current_row["datetime"].isoformat(),
                "retrain": retrain,
                "retrain_reason": retrain_reason,
                "threshold": round(threshold, 2),
                "session_id": session_id,
                "pipeline_status": "retraining" if retrain else "streaming",
                "llm_report": _build_stream_report(
                    y=y,
                    y_hat=y_hat,
                    rmse_percent=rmse_percent,
                    retrain=retrain,
                    retrain_reason=retrain_reason,
                    retrain_points=retrain_points,
                ),
            }
        )
        await asyncio.sleep(DEFAULT_STREAM_DELAY_SECONDS)

    await websocket.send_json(
        {
            "event": "stream_complete",
            "session_id": session_id,
            "record_count": int(len(stream_frame)),
            "retrain_count": retrain_points,
        }
    )


def _predict_next_value(lag_24: float, rolling_24: float, step: int, model_bias: float) -> float:
    seasonal_boost = np.sin(step / 6.0) * 0.018
    return (lag_24 * (0.58 + seasonal_boost)) + (rolling_24 * (0.42 - seasonal_boost)) + (rolling_24 * model_bias)


def _build_stream_report(
    y: float,
    y_hat: float,
    rmse_percent: float,
    retrain: bool,
    retrain_reason: str | None,
    retrain_points: int,
) -> str:
    direction = "상향" if y_hat >= y else "하향"
    base = (
        f"현재 예측값은 {y_hat:.2f}, 실제값은 {y:.2f}이며 오차는 {abs(y - y_hat):.2f}입니다. "
        f"누적 RMSE는 {rmse_percent:.2f}%로 계산되고 있으며 예측 방향은 {direction}입니다."
    )
    if retrain:
        return (
            f"{base} 재학습이 즉시 트리거되었습니다. 사유는 {retrain_reason}. "
            f"현재까지 총 재학습 횟수는 {retrain_points}회입니다."
        )
    return f"{base} 현재 파이프라인은 정상 스트리밍 상태이며 임계치 이내에서 운영 중입니다."
