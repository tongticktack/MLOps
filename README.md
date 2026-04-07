# MlOps_for_power_demand_forcasting

전력 수요 예측용 실시간 관제 데모 프로젝트입니다.  
프론트엔드는 React SPA, 백엔드는 FastAPI + WebSocket이며, `model_src`의 모델/드리프트/재학습 로직을 사용합니다.

## 실행 순서

### 1. 백엔드 의존성 설치

루트 디렉터리에서 실행:

```bash
pip install -r requirements.txt
```

### 2. 프론트엔드 의존성 설치

```bash
cd FE
npm install
cd ..
```

### 3. 백엔드 실행

루트 디렉터리에서 실행:

```bash
uvicorn backend.main:app --reload --port 7070
```

백엔드 기본 주소:

- HTTP: `http://localhost:7070`
- WebSocket: `ws://localhost:7070/ws/mlops`

헬스체크:

```bash
curl http://localhost:7070/health
```

### 4. 프론트엔드 실행

```bash
cd FE
npm run dev
```

브라우저에서 보통 아래 주소로 접속합니다.

- `http://localhost:5173`

## 사용 방법

1. FE 화면에서 CSV 파일을 업로드합니다.
2. `스트림 시작` 버튼을 누릅니다.
3. FE가 `POST /predict`로 세션을 생성합니다.
4. FE가 `ws://localhost:7070/ws/mlops?session_id=...` 로 WebSocket 연결을 엽니다.
5. 백엔드가 시간순으로 `y`, `y_hat`, `error`, `rmse`, `retrain` 이벤트를 스트리밍합니다.
6. FE는 실시간 차트, RMSE, 데이터 수, 마지막 재학습 시각, 운영 보고서를 갱신합니다.

## 입력 CSV 형식

현재 업로드 CSV는 아래 형식을 전제로 합니다.

```csv
날짜,1시,2시,3시,4시,5시,6시,7시,8시,9시,10시,11시,12시,13시,14시,15시,16시,17시,18시,19시,20시,21시,22시,23시,24시
2024-01-01,59440,56989,55335,...
2024-01-02,57950,55640,54423,...
```

백엔드는 이를 내부적으로 `datetime`, `power_demand` 시간별 형식으로 변환한 뒤 모델에 전달합니다.

## 현재 백엔드 워크플로우

```text
CSV 업로드
-> POST /predict
-> 세션 생성
-> 모델 로드
-> sliding-window evaluation
-> WebSocket /ws/mlops 연결
-> y / y_hat / error / rmse / retrain 스트리밍
-> FE 실시간 차트 및 운영 지표 갱신
```

## 주요 파일

- FE 메인 화면: [FE/src/App.jsx](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/FE/src/App.jsx)
- FE 문서: [FE/README.md](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/FE/README.md)
- FastAPI 엔트리: [backend/main.py](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/main.py)
- 예측 라우터: [backend/routers/predict.py](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/routers/predict.py)
- CSV 변환 로직: [backend/ml/model.py](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/ml/model.py)
- 스트리밍 추론 로직: [backend/ml/infer.py](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/ml/infer.py)
- 모델 명세: [backend/model/SPEC.md](/Users/geonwook/workspace/MlOps_for_power_demand_forcasting/backend/model/SPEC.md)

## 참고

- 재학습 threshold는 현재 FE 입력이 아니라 백엔드 내부 기준을 사용합니다.
- 재학습은 `model_src`의 드리프트/쿨다운 규칙에 따라 동작합니다.
- 실제 모델 파일이 없거나 의존성이 빠지면 백엔드 실행 시 오류가 날 수 있습니다.
