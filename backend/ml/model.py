from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "power_usage"


@dataclass
class ParsedDataset:
    stream_df: pd.DataFrame
    hourly_df: pd.DataFrame
    original_rows: int


def parse_power_csv(contents: bytes) -> ParsedDataset:
    text_buffer = io.BytesIO(contents)
    dataframe = _read_csv_with_fallback_encodings(text_buffer)

    if "날짜" not in dataframe.columns:
        raise ValueError("CSV에 '날짜' 컬럼이 필요합니다.")

    hour_columns = [column for column in dataframe.columns if column != "날짜"]
    if len(hour_columns) != 24:
        raise ValueError("CSV는 날짜 + 24개 시간대 컬럼(1시~24시)을 포함해야 합니다.")

    melted = dataframe.melt(id_vars=["날짜"], value_vars=hour_columns, var_name="hour_label", value_name=TARGET_COLUMN)
    melted["hour"] = melted["hour_label"].str.extract(r"(\d+)").astype(int)
    melted["datetime"] = pd.to_datetime(melted["날짜"]) + pd.to_timedelta(melted["hour"] - 1, unit="h")
    melted[TARGET_COLUMN] = pd.to_numeric(melted[TARGET_COLUMN], errors="coerce")
    melted = melted.dropna(subset=[TARGET_COLUMN]).sort_values("datetime").reset_index(drop=True)

    if len(melted) < 24 * 14:
        raise ValueError("최소 14일 이상의 시간별 전력 데이터가 필요합니다.")

    stream_df = melted[["datetime", TARGET_COLUMN]].copy()

    hourly_df = stream_df.copy()
    hourly_df["hour"] = hourly_df["datetime"].dt.hour
    hourly_df["dayofweek"] = hourly_df["datetime"].dt.dayofweek
    hourly_df["month"] = hourly_df["datetime"].dt.month
    hourly_df["day"] = hourly_df["datetime"].dt.day
    hourly_df["lag_24"] = hourly_df[TARGET_COLUMN].shift(24)
    hourly_df["rolling_24"] = hourly_df[TARGET_COLUMN].shift(1).rolling(24).mean()
    hourly_df = hourly_df.dropna().reset_index(drop=True)

    return ParsedDataset(stream_df=stream_df, hourly_df=hourly_df, original_rows=len(melted))


def _read_csv_with_fallback_encodings(buffer: io.BytesIO) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        buffer.seek(0)
        try:
            return pd.read_csv(buffer, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949 형식을 확인하세요.") from last_error
