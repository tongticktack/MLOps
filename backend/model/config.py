"""
Central configuration: all file paths and hyperparameter constants live here.
Nothing should be hardcoded outside this module.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — legacy (Team A notebook, 2013-2022)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "solution_team_a" / "Power_Prediction_Team_A"
OUTPUT_DIR = Path(__file__).parent / "outputs"

POWER_DEMAND_CSV      = DATA_DIR / "power_demand_2013_2022.csv"
TEMPERATURE_CSV       = DATA_DIR / "average_temperature_in_2023.csv"
POWER_DEMAND_2023_CSV = DATA_DIR / "power_demand_2023.csv"

REGIONAL_FILES: dict[str, Path] = {
    "경남": DATA_DIR / "경남온습도.csv",
    "경북": DATA_DIR / "경북온습도.csv",
    "충남": DATA_DIR / "충남온습도.csv",
    "충북": DATA_DIR / "충북온습도.csv",
    "강원영서": DATA_DIR / "강원영서온습도.csv",
    "강원영동": DATA_DIR / "강원영동온습도.csv",
    "서울경기": DATA_DIR / "서울경기온습도.csv",
    "전북": DATA_DIR / "전북온습도.csv",
    "전남": DATA_DIR / "전남온습도.csv",
    "제주": DATA_DIR / "제주온습도.csv",
}

KOSPI_CSV             = DATA_DIR / "일별주가데이터_KOSPI.csv"
INDUSTRY_INDEX_CSV    = DATA_DIR / "전산업생산지수(계절조정).csv"
UTILIZATION_INDEX_CSV = DATA_DIR / "제조업 가동률지수(계절조정).csv"

# ---------------------------------------------------------------------------
# Paths — recent data (2022-2025)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECENT_DIR = PROJECT_ROOT / "data"

DEMAND_2022_CSV  = RECENT_DIR / "elect_demand_2022.csv"
DEMAND_2024_CSV  = RECENT_DIR / "elect_demand_2024.csv"
DEMAND_2025_CSV  = RECENT_DIR / "elect_demand_2025.csv"

CLIMATE_2024_CSV = RECENT_DIR / "ClimateByHour_2024.csv"
CLIMATE_2025_CSV = RECENT_DIR / "ClimateByHour_2025.csv"

# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------

FULL_INDEX_START = "2013-01-01 00:00:00"
FULL_INDEX_END   = "2023-12-31 23:00:00"

TRAIN_END_YEAR  = 2021
VAL_YEAR        = 2022
PREDICT_YEAR    = 2023

# Recent pipeline
RECENT_TRAIN_YEAR   = 2024
RECENT_PREDICT_YEAR = 2025

# ---------------------------------------------------------------------------
# Feature columns (must match notebook exactly)
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    # National temperature
    "temp", "HDD", "CDD",
    # Regional temperature + humidity (10 regions × 2)
    "temp_경남",   "humidity_경남",
    "temp_경북",   "humidity_경북",
    "temp_충남",   "humidity_충남",
    "temp_충북",   "humidity_충북",
    "temp_강원영서", "humidity_강원영서",
    "temp_강원영동", "humidity_강원영동",
    "temp_서울경기", "humidity_서울경기",
    "temp_전북",   "humidity_전북",
    "temp_전남",   "humidity_전남",
    "temp_제주",   "humidity_제주",
    # Temporal
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday", "long_weekend_flag", "is_october",
    # Macroeconomic
    "industry_index", "utilization_index",
    # Market
    "KOSPI_close", "KOSPI_volume",
]

# ---------------------------------------------------------------------------
# Model hyperparameter search space (mirrors notebook exactly)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

PARAM_DIST: dict = {
    "n_estimators":    [400, 500, 600, 700, 800, 900, 1000],
    "learning_rate":   [0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
    "max_depth":       [3, 4, 5, 6, 7, 8, 9],
    "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha":       [0, 0.1, 0.3, 0.5, 0.7, 1.0],
    "reg_lambda":      [0.5, 1, 1.5, 2, 2.5, 3.0],
}
N_ITER       = 20
CV_SPLITS    = 3
