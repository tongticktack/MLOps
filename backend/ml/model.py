from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


TARGET_COLUMN = "power_demand"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIMATE_FILES_BY_YEAR = {
    2022: PROJECT_ROOT / "data" / "ClimateByHour_2022.csv",
    2023: PROJECT_ROOT / "data" / "ClimateByHour_2023.csv",
    2024: PROJECT_ROOT / "data" / "ClimateByHour_2024.csv",
    2025: PROJECT_ROOT / "data" / "ClimateByHour_2025.csv",
}


def parse_power_csv(contents: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(contents)
    dataframe = _read_csv_with_fallback_encodings(buffer)

    if "날짜" not in dataframe.columns:
        raise ValueError("CSV에 '날짜' 컬럼이 필요합니다.")

    dataframe = dataframe.dropna(subset=["날짜"]).copy()
    dataframe["날짜"] = pd.to_datetime(dataframe["날짜"])

    if "24시" in dataframe.columns:
        next_day = dataframe[["날짜", "24시"]].copy()
        next_day["날짜"] = next_day["날짜"] + pd.Timedelta(days=1)
        next_day = next_day.rename(columns={"24시": "0시"})
        dataframe = dataframe.drop(columns=["24시"])
        dataframe = pd.merge(dataframe, next_day, on="날짜", how="outer")

    hour_columns = [column for column in dataframe.columns if column != "날짜"]
    if len(hour_columns) != 24:
        raise ValueError("CSV는 날짜 + 24개 시간대 컬럼(1시~24시)을 포함해야 합니다.")

    melted = dataframe.melt(id_vars=["날짜"], value_vars=hour_columns, var_name="hour_label", value_name=TARGET_COLUMN)
    melted["hour"] = melted["hour_label"].str.extract(r"(\d+)").astype(int)
    melted["datetime"] = pd.to_datetime(melted["날짜"]) + pd.to_timedelta(melted["hour"], unit="h")
    melted[TARGET_COLUMN] = pd.to_numeric(melted[TARGET_COLUMN], errors="coerce")
    melted = melted.dropna(subset=[TARGET_COLUMN]).sort_values("datetime").reset_index(drop=True)

    if len(melted) < 169:
        raise ValueError("최소 169시간 이상의 시간별 전력 데이터가 필요합니다.")

    normalized = melted[["datetime", TARGET_COLUMN]].copy()
    normalized = normalized.set_index("datetime").sort_index()
    normalized = _attach_climate_features(normalized)
    return normalized


def _read_csv_with_fallback_encodings(buffer: io.BytesIO) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        buffer.seek(0)
        try:
            return pd.read_csv(buffer, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949 형식을 확인하세요.") from last_error


def _attach_climate_features(demand_df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(set(demand_df.index.year.tolist()))
    climate_frames = []

    for year in years:
        climate_path = CLIMATE_FILES_BY_YEAR.get(year)
        if climate_path and climate_path.exists():
            climate_frames.append(_load_climate(climate_path))

    if not climate_frames:
        raise ValueError("모델이 요구하는 기상 데이터 파일을 찾을 수 없습니다.")

    climate_df = pd.concat(climate_frames).sort_index()
    climate_df = climate_df[~climate_df.index.duplicated(keep="last")]
    merged = demand_df.join(climate_df, how="left")

    for column in ["강수량", "적설"]:
        if column in merged.columns:
            merged[column] = merged[column].fillna(0)
    for column in ["기온", "습도"]:
        if column in merged.columns:
            merged[column] = merged[column].interpolate(method="time").bfill().ffill()

    missing = [column for column in ["기온", "강수량", "습도", "적설"] if column not in merged.columns]
    if missing:
        raise ValueError(f"기상 피처 컬럼이 누락되었습니다: {', '.join(missing)}")

    if merged[["기온", "강수량", "습도", "적설"]].isna().any().any():
        raise ValueError("기상 데이터 정렬에 실패했습니다. 기후 데이터 시간축을 확인하세요.")

    return merged


def _load_climate(path: Path) -> pd.DataFrame:
    climate = pd.read_csv(path, encoding="cp949")
    climate["일시"] = pd.to_datetime(climate["일시"])
    climate = climate.set_index("일시").sort_index()
    climate = climate.rename(
        columns={
            "기온(°C)": "기온",
            "강수량(mm)": "강수량",
            "습도(%)": "습도",
            "적설(cm)": "적설",
        }
    )
    return climate[["기온", "강수량", "습도", "적설"]]
