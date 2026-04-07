from fastapi import APIRouter, File, UploadFile, WebSocket
from fastapi.responses import JSONResponse

from backend.ml.infer import create_prediction_session, stream_prediction_session


router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "INVALID_FILE_TYPE", "message": "CSV 파일만 업로드할 수 있습니다."}},
        )

    contents = await file.read()
    if not contents:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "EMPTY_FILE", "message": "빈 파일은 업로드할 수 없습니다."}},
        )

    try:
        return create_prediction_session(contents=contents, filename=file.filename)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "INVALID_CSV", "message": str(exc)}},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "PREDICTION_FAILED", "message": f"예측 처리 중 오류가 발생했습니다: {exc}"}},
        )


@router.websocket("/ws/mlops")
async def mlops_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id", "")
    try:
        threshold = float(websocket.query_params.get("threshold", "5.0"))
    except ValueError:
        threshold = 5.0
    await stream_prediction_session(websocket=websocket, session_id=session_id, threshold=threshold)
