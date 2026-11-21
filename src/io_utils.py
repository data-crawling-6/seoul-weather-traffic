"""팀 공통 TOPIS 입력 함수.

2024-02 이전의 ``08시``와 이후의 ``~08시``를 ``~08시``로 통일한다.
원본 CSV는 수정하지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


HOUR_COLS = [f"~{hour:02d}시" for hour in range(1, 25)]


def data_root() -> Path:
    return Path(
        os.getenv("SEOUL_TRAFFIC_DATA_ROOT", r"D:\seoul-weather-traffic-data")
    ).resolve()


def read_topis(month: str, root: Path | None = None, **read_csv_kwargs: object) -> pd.DataFrame:
    """``YYYY-MM`` TOPIS CSV를 읽고 시간 열 이름을 표준화한다."""
    if len(month) != 7 or month[4] != "-":
        raise ValueError("month는 YYYY-MM 형식이어야 합니다.")
    base = Path(root).resolve() if root is not None else data_root()
    path = base / "topis_speed" / "csv" / f"topis_speed_{month}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    kwargs = {"encoding": "utf-8-sig", "low_memory": False, **read_csv_kwargs}
    frame = pd.read_csv(path, **kwargs)
    rename = {
        f"{hour:02d}시": f"~{hour:02d}시"
        for hour in range(1, 25)
        if f"{hour:02d}시" in frame.columns
    }
    frame = frame.rename(columns=rename)
    missing = [column for column in HOUR_COLS if column not in frame.columns]
    if missing:
        raise ValueError(f"TOPIS 시간 열이 없습니다: {missing}")
    return frame


def read_topis_gaps(project_root: Path | None = None) -> pd.DataFrame:
    """전 링크 동시 결측으로 재현된 TOPIS 시스템 공백 41시간을 읽는다."""
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    path = root / "config" / "topis_gaps.csv"
    gaps = pd.read_csv(path, encoding="utf-8-sig")
    gaps["datetime"] = pd.to_datetime(gaps["datetime"], errors="raise")
    gaps["source_date"] = pd.to_datetime(gaps["source_date"], errors="raise")
    if len(gaps) != 41 or gaps["datetime"].duplicated().any():
        raise ValueError("TOPIS 시스템 공백은 중복 없는 41시간이어야 합니다.")
    return gaps
