# Power Demand MLOps Pipeline — 명세서

> 한국 시간별 전력 수요량(MWh) 예측 · 드리프트 감지 · 자동 재학습 시스템

---

## 1. 시스템 개요

```
[Input CSV]
    │  datetime (hourly) + power_demand (MWh) + 기상 4종
    ▼
[Sliding Window Inference]
    │  168h 슬라이딩 윈도우 → 1-step 예측 (벡터화, 누수 없음)
    ▼
[Dual RMSE 계산]
    │  RMSE_24h  (최근 24h 오차, 단기 경보)
    │  RMSE_168h (최근 168h 오차, 드리프트 확정)
    ▼
[상태 분류]
    │  normal / warning / drift
    ▼
[drift_log.csv 기록]
    │  매 24h 블록마다 1행
    ▼
[조건부 재학습]
    │  drift 확정 + 쿨다운(7일) 경과 시 자동 재학습
    ▼
[24h 자동회귀 예측]
    └─ 입력 윈도우 끝에서 24시간 앞 예측 생성
```

---

## 2. 모델 명세

### 2-1. 알고리즘

| 항목 | 내용 |
|------|------|
| 알고리즘 | XGBoost (`XGBRegressor`, `reg:squarederror`) |
| 하이퍼파라미터 탐색 | `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=3)` |
| 평가 지표 | RMSE (MWh) |
| 학습 데이터 | 2024년 전국 시간별 전력 수요량 + 기상 데이터 |
| 모델 파일 | `model/window_model.pkl` |

### 2-2. 입력 피처 (27개)

#### 기상 피처 (4개) — 예측 시점의 실측값 (누수 없음)
| 피처명 | 설명 | 단위 |
|--------|------|------|
| `기온` | 기온 | °C |
| `강수량` | 강수량 | mm |
| `습도` | 상대습도 | % |
| `적설` | 적설량 | cm |

#### 래그 피처 (9개)
| 피처명 | 설명 |
|--------|------|
| `lag_1` | 1시간 전 수요량 |
| `lag_2` | 2시간 전 수요량 |
| `lag_3` | 3시간 전 수요량 |
| `lag_6` | 6시간 전 수요량 |
| `lag_12` | 12시간 전 수요량 |
| `lag_24` | 24시간 전 수요량 (어제 동시각) |
| `lag_48` | 48시간 전 수요량 |
| `lag_72` | 72시간 전 수요량 |
| `lag_168` | 168시간 전 수요량 (지난주 동시각) |

#### 롤링 통계 피처 (4개)
| 피처명 | 설명 |
|--------|------|
| `roll_mean_24h` | 직전 24h 평균 수요량 |
| `roll_std_24h` | 직전 24h 수요량 표준편차 |
| `roll_mean_168h` | 직전 168h(7일) 평균 수요량 |
| `roll_std_168h` | 직전 168h 수요량 표준편차 |

#### 시간 주기 피처 (6개, cyclic encoding)
| 피처명 | 설명 | 범위 |
|--------|------|------|
| `hour_sin` / `hour_cos` | 시간(0~23) 사인/코사인 | [-1, 1] |
| `dow_sin` / `dow_cos` | 요일(0~6) 사인/코사인 | [-1, 1] |
| `month_sin` / `month_cos` | 월(1~12) 사인/코사인 | [-1, 1] |

#### 캘린더 이진 피처 (4개)
| 피처명 | 설명 |
|--------|------|
| `is_weekend` | 토/일 여부 (1/0) |
| `is_holiday` | 한국 공휴일 + 근로자의날 (1/0) |
| `is_october` | 10월 여부 (1/0) — 전력 수요 계절성 반영 |
| `long_weekend_flag` | 전날 또는 다음날이 공휴일인 연휴 여부 (1/0) |

> **기상 데이터가 없을 때**: 기상 피처 4개를 제외한 23개 피처로 자동 fallback.  
> 단, 학습 모델과 피처 수가 불일치하면 예측 오차가 커짐 — 기상 데이터 제공 권장.

### 2-3. 출력

#### 슬라이딩 윈도우 평가 출력 (`outputs/predictions.csv`)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `timestamp` | datetime (index) | 예측 대상 시각 |
| `y_true` | float | 실제 전력 수요량 (MWh) |
| `y_pred` | float | 모델 예측값 (MWh) |
| `error` | float | `y_true - y_pred` |
| `rmse_24h` | float | 해당 시점까지 최근 24h RMSE |
| `rmse_168h` | float | 해당 시점까지 최근 168h RMSE |

#### 24h 예측 출력 (`outputs/forecast_24h.csv`)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `timestamp` | datetime (index) | 미래 시각 (입력 끝 + 1h ~ +24h) |
| `y_pred` | float | 예측 전력 수요량 (MWh) |

> 자동회귀 방식: 예측값을 다음 스텝의 래그 피처로 활용.  
> 기상 피처는 미래값 불확실으로 인해 해당 범위에서 제외됨.

#### 지표 출력 (`outputs/metrics.json`)
```json
{
  "n_predictions": 576,
  "window_size": 168,
  "RMSE": 893.45,
  "MAE": 701.23,
  "MAPE": 1.82,
  "baseline_rmse": 801.11,
  "threshold": 961.33,
  "overall_status": "normal",
  "retrained": false,
  "total_blocks": 24,
  "normal": 22,
  "warning": 1,
  "drift": 0
}
```

---

## 3. MLOps 기준 (드리프트 감지 · 재학습)

### 3-1. 베이스라인 RMSE

드리프트 판단의 기준이 되는 **건강한 상태**의 RMSE.

- **기본**: 입력 데이터의 첫 168개 예측값으로 자동 계산
- **권장**: 검증된 기간(예: 1월)의 RMSE를 외부에서 주입 (`baseline_rmse` 파라미터)
- **임계값**: `threshold = baseline_rmse × 1.2`

```
baseline_rmse = 801 MWh  →  threshold = 961 MWh
```

### 3-2. 듀얼 RMSE 시그널

| 시그널 | 윈도우 | 역할 |
|--------|--------|------|
| `RMSE_24h` | 최근 24시간 | 단기 이상 조기 경보 — 노이즈에 민감 |
| `RMSE_168h` | 최근 168시간(7일) | 지속적 성능 저하 확정 — 노이즈 저항성 |

두 시그널을 분리하는 이유: 단일 임계값 사용 시 **단일 이벤트(명절, 한파)에 의한 오탐** 또는 **실제 분포 변화 미감지** 중 하나가 발생함.

### 3-3. 상태 분류 로직

```
매 24h 블록마다 1회 평가:

if RMSE_168h > threshold:   → "drift"    (지속적 성능 저하 확정)
elif RMSE_24h > threshold:  → "warning"  (단기 이상 신호)
else:                       → "normal"
```

| 상태 | 조건 | 의미 | 권장 조치 |
|------|------|------|-----------|
| `normal` | 두 RMSE 모두 threshold 이하 | 정상 운영 | 없음 |
| `warning` | RMSE_24h > threshold | 단기 이상 — 일시적일 수 있음 | 모니터링 강화 |
| `drift` | RMSE_168h > threshold | 7일간 지속적 성능 저하 | **재학습 트리거** |

### 3-4. 전체 기간 상태 집계 (`overall_status`)

```
해당 실행 구간에 drift 블록이 1개라도 있으면 → "drift"
drift는 없고 warning 블록이 있으면 → "warning"
모두 normal → "normal"
```

### 3-5. 재학습 트리거 조건

아래 두 조건이 **모두** 충족될 때 자동 재학습:

| 조건 | 기준 |
|------|------|
| 드리프트 확정 | `overall_status == "drift"` |
| 쿨다운 경과 | 마지막 재학습으로부터 **7일 이상** 경과 |

쿨다운 상태는 `retrain/last_retrain.json`에 영속 저장.

### 3-6. 재학습 데이터 구성

```
재학습 데이터 = 최근 3개월(≈2184h) + 168h 래그 번인(burn-in)
             = 총 2352h

2025 데이터가 부족하면 2024 데이터 말미로 보충.
```

재학습 후 모델은 즉시 `model/window_model.pkl`에 덮어씌워지고, 이후 예측에 반영됨.

### 3-7. 드리프트 로그 (`logs/drift_log.csv`)

```
timestamp,RMSE_24h,RMSE_168h,baseline_rmse,status
2025-01-08,1134.75,902.12,801.11,warning
2025-02-07,1767.39,1163.21,801.11,drift   ← 최초 드리프트
...
```

매 평가 실행마다 append. 기존 내용은 유지.

---

## 4. 입력 CSV 스펙

### 4-1. 필수 컬럼

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `datetime` | string (ISO 8601) | `2025-01-01 00:00:00` 형식, 1시간 간격 |
| `power_demand` | float | 전력 수요량 (MWh) |

### 4-2. 선택 컬럼 (기상 — 제공 시 정확도 향상)

| 컬럼명 | 타입 | 설명 | 누락 처리 |
|--------|------|------|-----------|
| `기온` (또는 `기온(°C)`, `기온(c)`) | float | 기온 (°C) | 컬럼 없으면 피처 전체 제외 |
| `강수량` (또는 `강수량(mm)`) | float | 강수량 (mm) | 0으로 대체 |
| `습도` (또는 `습도(%)`) | float | 상대습도 (%) | 선형 보간 |
| `적설` (또는 `적설(cm)`) | float | 적설 (cm) | 0으로 대체 |

### 4-3. 최소 행 수

| 목적 | 최소 행 수 |
|------|-----------|
| 슬라이딩 윈도우 평가 | **169행** (168h 윈도우 + 1 타깃) |
| 듀얼 RMSE 안정화 | **336행 이상 권장** (168h 초기화 + 168h 평가) |
| 24h 자동회귀 예측 | **168행** |

---

## 5. 백엔드 연동 API

### 5-1. 사용 방법

```python
# pipeline/ 디렉토리를 sys.path에 추가하거나 패키지로 설치 후:
from api import run_evaluation, get_forecast, get_drift_log, trigger_retrain, get_model_status
```

### 5-2. `run_evaluation()`

전체 평가 파이프라인 실행 (추론 → 드리프트 감지 → 조건부 재학습).

```python
result = run_evaluation(
    "data.csv",
    baseline_rmse=801.11,   # 없으면 입력 데이터에서 자동 계산
    allow_retrain=True,
    model_path=None,        # None이면 model/window_model.pkl 사용
)
```

**반환값 (`EvaluationResult`)**:
```python
result.status          # "normal" | "warning" | "drift"
result.baseline_rmse   # 801.11
result.threshold       # 961.33
result.period_rmse     # 해당 기간 전체 RMSE
result.mae             # MAE
result.mape            # MAPE (%)
result.n_predictions   # 평가된 예측 수
result.blocks.total    # 24h 블록 수
result.blocks.normal   # 정상 블록 수
result.blocks.warning  # 경고 블록 수
result.blocks.drift    # 드리프트 블록 수
result.retrained       # 재학습 실행 여부 (bool)
result.to_dict()       # JSON 직렬화용 dict
```

### 5-3. `get_forecast()`

다음 24시간 전력 수요량 예측.

```python
result = get_forecast("data.csv")
# result.forecast  → [{"timestamp": "2025-02-01T00:00:00", "y_pred": 5234.12}, ...]
# result.generated_at → "2025-01-31T23:00:00"
result.to_dict()
```

### 5-4. `get_drift_log()`

드리프트 로그 조회.

```python
entries = get_drift_log(
    start="2025-02-01",
    end="2025-02-28",
    status_filter="drift",   # None이면 전체
)
# entries → [DriftLogEntry(timestamp=..., rmse_24h=..., rmse_168h=..., status="drift"), ...]
```

### 5-5. `trigger_retrain()`

수동 재학습 트리거.

```python
result = trigger_retrain(
    cutoff_dt=None,   # None이면 현재 시각 기준
    force=False,      # True이면 7일 쿨다운 무시
)
# result.success      → True / False
# result.model_path   → "model/window_model.pkl"
# result.retrained_at → ISO 타임스탬프
# result.message      → 성공/실패 사유
```

### 5-6. `get_model_status()`

현재 모델 및 스케줄러 상태 조회.

```python
status = get_model_status()
# {
#   "model_exists": True,
#   "model_path": "model/window_model.pkl",
#   "last_retrain": "3d 2h ago",
#   "can_retrain_now": False
# }
```

---

## 6. CLI 사용

```bash
# 초기 학습
python model/train.py

# 드리프트 감지 + 자동 재학습 (전체 파이프라인)
python drift/detector.py --input data.csv

# 외부 베이스라인 주입 (권장)
python drift/detector.py --input data.csv --baseline-rmse 801.11

# 재학습 비활성화
python drift/detector.py --input data.csv --no-retrain

# 시나리오 검증 (Jan→Feb→Mar, 3-round 시뮬레이션)
python scenario_test.py
```

---

## 7. 파일 구조

```
pipeline/
├── api.py                      ← 백엔드 연동 진입점 (이 파일)
├── config.py                   ← 경로 / 상수 전체 관리
├── SPEC.md                     ← 이 명세서
│
├── model/
│   ├── train.py                ← 초기 학습 (2024 → window_model.pkl)
│   ├── preprocess.py           ← 피처 엔지니어링 (build_features)
│   └── load_model.py           ← 모델 로더
│
├── drift/
│   ├── detector.py             ← CLI 진입점
│   ├── evaluator.py            ← 슬라이딩 윈도우 평가 + 듀얼 RMSE
│   ├── decision.py             ← 상태 분류 + 로그 저장
│   ├── sliding_window.py       ← 1-step 추론 + 24h 자동회귀 예측
│   └── metrics.py              ← RMSE / MAE / MAPE / rolling_rmse
│
├── retrain/
│   ├── retrain.py              ← 재학습 (최근 3개월 데이터)
│   └── scheduler.py            ← 쿨다운 관리 (last_retrain.json)
│
├── logs/
│   └── drift_log.csv           ← 드리프트 이력 (append-only)
│
├── outputs/
│   ├── predictions.csv         ← 슬라이딩 윈도우 예측 결과
│   ├── forecast_24h.csv        ← 다음 24h 예측
│   └── metrics.json            ← 평가 지표 + 상태
│
└── scenario_test.py            ← 3-round MLOps 시나리오 검증
```

---

## 8. 의존성

```
holidays >= 0.46
numpy    >= 1.24
pandas   >= 2.0
scikit-learn >= 1.4
xgboost  >= 2.0
joblib   (xgboost 의존성으로 자동 설치)
```

```bash
pip install -r requirements.txt
```
