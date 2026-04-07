# FE Overview

이 문서는 현재 `FE` 프론트엔드가 실제로 어떤 방식으로 백엔드와 통신하는지 설명합니다. 백엔드 구현자 입장에서 보면, 이 문서를 기준으로 API 요청 형식과 응답 스키마를 바로 맞출 수 있습니다.

## 현재 FE 역할

프론트는 React 단일 페이지 애플리케이션(SPA)이며, 사용자는 1년치 전력 CSV를 업로드한 뒤 예측 요청을 보냅니다. 응답을 받으면 한 화면에서 아래 3가지를 렌더링합니다.

- RMSE(%)
- 예측 Figure
- LLM 분석 보고서

실제 메인 화면 구현 파일은 [`App.jsx`](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/FE/src/App.jsx) 입니다.

## 현재 화면 구성

### 1. CSV 업로드 섹션

사용자 동작:

- Drag & Drop으로 CSV 업로드
- 또는 파일 선택 버튼으로 CSV 업로드
- `전력 예측` 버튼 클릭

프론트 동작:

- 선택한 파일을 `File` 객체로 상태에 저장
- 버튼 클릭 시 `FormData`를 생성
- 백엔드 `POST /predict`로 CSV 파일 전송

### 2. RMSE 카드

백엔드가 반환한 RMSE 값을 `%` 단위로 표시합니다.

현재 FE는 아래 필드 중 하나를 읽도록 되어 있습니다.

- `rmse_percent`
- `rmse`
- `metrics.rmse_percent`
- `metrics.rmse`

### 3. 예측 Figure 영역

백엔드가 반환한 figure 이미지를 그대로 표시합니다.

현재 FE는 아래 필드 중 하나를 읽도록 되어 있습니다.

- `figure`
- `forecast_figure`
- `result.figure`

허용 형식:

- base64 문자열
- `data:image/...` 형식 문자열
- 이미지 URL 문자열
- `{ "base64": "...", "mime_type": "image/png" }`
- `{ "url": "..." }`

### 4. LLM 분석 보고서 영역

백엔드가 내려준 보고서 텍스트를 타이핑 효과로 렌더링합니다.

현재 FE는 아래 필드 중 하나를 읽도록 되어 있습니다.

- `llm_report`
- `report.full_text`
- `report.summary`
- `report`

## 실제 요청 방식

현재 FE는 아래 주소로 요청합니다.

- `POST http://localhost:7070/predict`

요청 형식:

- `multipart/form-data`

요청 필드:

- `file`: 업로드한 CSV 파일

예시:

```bash
curl -X POST http://localhost:7070/predict \
  -F "file=@sample.csv"
```

## 현재 FE가 기대하는 응답

가장 단순한 권장 응답 예시는 아래와 같습니다.

```json
{
  "rmse": 4.82,
  "rmse_percent": 4.82,
  "figure": "<base64 png string>",
  "llm_report": "향후 24시간 중 최대 수요는 18:00에 발생할 가능성이 높습니다.",
  "record_count": 8760,
  "horizon": "Next 24 Hours",
  "confidence": "95.18%",
  "mlflow": {
    "tracking_uri": "file:./mlruns",
    "experiment": "power-demand-forecast",
    "run_id": "abcd1234",
    "model_uri": "runs:/abcd1234/model"
  }
}
```

현재 백엔드 구현도 이 방향에 맞춰 작성되어 있습니다.

## FE가 실제로 읽는 응답 필드

### RMSE

우선순위:

- `rmse_percent`
- `rmse`
- `metrics.rmse_percent`
- `metrics.rmse`
- `result.rmse_percent`
- `result.rmse`

주의:

- 숫자로 오면 FE가 `%`를 붙입니다.
- 문자열에 `%`가 이미 있으면 그대로 표시합니다.

### Figure

우선순위:

- `figure`
- `forecast_figure`
- `result.figure`

규칙:

- plain base64면 FE가 `data:image/png;base64,`를 붙여 렌더링
- 이미 `data:image/...` 형식이면 그대로 사용
- URL이면 그대로 사용

### 보고서

우선순위:

- `llm_report`
- `report.full_text`
- `report.summary`
- `report`
- `result.llm_report`
- `result.report.full_text`

## 에러 응답 형식

현재 FE는 아래 구조를 읽을 수 있게 되어 있습니다.

```json
{
  "error": {
    "code": "INVALID_CSV",
    "message": "CSV 형식이 올바르지 않습니다."
  }
}
```

권장 에러 코드:

- `INVALID_FILE_TYPE`
- `EMPTY_FILE`
- `INVALID_CSV`
- `PREDICTION_FAILED`

## 현재 BE 내부 구조와 FE 관점

현재 백엔드는 CSV를 받으면 내부적으로 다음 흐름으로 처리하도록 구성되어 있습니다.

1. CSV 파싱
2. 시간별 데이터 전개
3. 모델 학습
4. MLflow에 실험 및 모델 기록
5. MLflow에서 모델 로드
6. 24시간 예측 수행
7. Figure 생성
8. RMSE와 보고서 포함 JSON 반환

프론트 입장에서는 중요한 점이 하나 있습니다.

- FE는 MLflow를 직접 호출하지 않습니다.
- FE는 오직 `POST /predict`만 호출합니다.
- MLflow는 백엔드 내부에서 실험 추적과 모델 저장을 담당합니다.

즉 구조는 아래와 같습니다.

```text
FE
  -> POST /predict
BE FastAPI
  -> CSV 검증
  -> 모델 학습 / 추론
  -> MLflow 로깅
  -> figure / rmse / report 반환
```

## CSV 전제

현재 BE 구현은 아래 형태의 CSV를 기준으로 동작합니다.

- `날짜` 컬럼 1개
- 시간 컬럼 24개
- 예: `1시`, `2시`, ..., `24시`

예시:

```csv
날짜,1시,2시,3시,...,24시
2024-01-01,59440,56989,55335,...,60735
2024-01-02,57950,55640,54423,...,69487
```

현재 BE는 최소 14일 이상의 시간별 데이터가 있어야 동작하도록 되어 있습니다.

## 실행 전제

FE 개발 서버:

```bash
cd FE
npm install
npm run dev
```

BE 서버:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 7070
```

## 구현 참고 파일

- FE 진입점: [`main.jsx`](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/FE/src/main.jsx)
- FE 메인 화면: [`App.jsx`](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/FE/src/App.jsx)
- FE 스타일: [`index.css`](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/FE/src/index.css)
- BE 앱 엔트리: [`main.py`](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/main.py)
- BE 예측 라우터: [`predict.py`](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/routers/predict.py)
- MLflow 학습 로직: [`train.py`](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/ml/train.py)
- 추론 파이프라인: [`infer.py`](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/ml/infer.py)

## 한 줄 정리

현재 FE는 `localhost:7070`의 `POST /predict`에 CSV를 보내고, 응답 JSON에서 `rmse`, `figure`, `llm_report`를 읽어 화면에 렌더링합니다. MLflow는 FE가 직접 쓰는 대상이 아니라, BE 내부에서 학습 이력과 모델을 관리하는 용도입니다.
