"""팀 공통 TOPIS 파일 읽기 함수.

저장소 내부 ``data/raw/topis/csv``와 외부 데이터 루트를 모두 지원한다.
2024-02 이전 ``08시``와 이후 ``~08시``는 ``~08시``로 통일한다.
원본 CSV는 수정하지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOPIS_CSV_DIR = ROOT / "data" / "raw" / "topis" / "csv"
HOUR_COLS = [f"~{hour:02d}시" for hour in range(1, 25)]


def data_root() -> Path:
    return Path(os.getenv("SEOUL_TRAFFIC_DATA_ROOT", r"D:\seoul-weather-traffic-data")).resolve()


def _unify_hour_name(column: object) -> object:
    if isinstance(column, str) and len(column) >= 3:
        if column[:2].isdigit() and column.endswith("시"):
            return "~" + column
    return column


def _resolve_topis_path(month_or_path: str | Path, root: Path | None) -> Path:
    candidate = Path(month_or_path)
    if candidate.suffix.lower() == ".csv":
        return candidate.resolve()
    month = str(month_or_path)
    if len(month) != 7 or month[4] != "-":
        raise ValueError("month는 YYYY-MM 형식이거나 CSV 파일 경로여야 합니다.")
    if root is not None:
        base = Path(root).resolve()
        candidates = [base / "topis_speed" / "csv" / f"topis_speed_{month}.csv", base / f"topis_speed_{month}.csv"]
    else:
        candidates = [TOPIS_CSV_DIR / f"topis_speed_{month}.csv", data_root() / "topis_speed" / "csv" / f"topis_speed_{month}.csv"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(candidates[0])


def read_topis(
    month_or_path: str | Path,
    root: Path | None = None,
    validate_hours: bool = True,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    """월 문자열 또는 CSV 경로로 TOPIS 자료를 읽고 시간 열을 통일한다."""
    path = _resolve_topis_path(month_or_path, root)
    kwargs = {"encoding": "utf-8-sig", "low_memory": False, **read_csv_kwargs}
    frame = pd.read_csv(path, **kwargs).rename(columns=_unify_hour_name)
    if validate_hours:
        missing = [column for column in HOUR_COLS if column not in frame.columns]
        if missing:
            raise ValueError(f"TOPIS 시간 열이 없습니다: {missing}")
    return frame


def read_topis_gaps(project_root: Path | None = None) -> pd.DataFrame:
    """중복 없는 TOPIS 시스템 공백 41시간을 읽는다."""
    root = Path(project_root).resolve() if project_root is not None else ROOT
    gaps = pd.read_csv(root / "config" / "topis_gaps.csv", encoding="utf-8-sig")
    if "datetime" not in gaps.columns:
        raise ValueError("topis_gaps.csv에 datetime 열이 없습니다.")
    gaps["datetime"] = pd.to_datetime(gaps["datetime"], errors="raise")
    if len(gaps) != 41 or gaps["datetime"].duplicated().any():
        raise ValueError("TOPIS 시스템 공백은 중복 없는 41시간이어야 합니다.")
    return gaps
