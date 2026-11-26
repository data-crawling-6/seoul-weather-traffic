"""팀 공통 읽기 함수 (TOPIS 파일 · 모델 세트).

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


# ── 모델 세트 (Train · Dev · Test) ─────────────────────────────────
# 규칙은 config/periods.yaml 의 model_sets 한 곳에만 둔다 (노션 분석 설계 7.2 · 7.3).
# 폴더 data/processed/{train,eval} 과 split 열은 그대로 두고, 읽을 때 model_set 열을 붙인다.
# Test 시즌은 마지막 평가 전까지 열지 않도록 기본값으로 빼고, include_test=True 일 때만 남긴다.
MODEL_SET_NAMES = ("train", "dev", "test")


def _season_key(value: object) -> str:
    """날씨 표의 W2022 · S2023 표기를 2022-23_winter · 2023_summer 로 맞춘다. 이미 그 표기면 그대로."""
    text = str(value)
    if len(text) == 5 and text[0] in "WS" and text[1:].isdigit():
        year = int(text[1:])
        return f"{year}-{str(year + 1)[2:]}_winter" if text[0] == "W" else f"{year}_summer"
    return text


def _season_of_date(ts: pd.Timestamp) -> str:
    """날짜 → 시즌 이름 (src/weather.py season_of 와 같은 규칙: 10~4월 겨울, 5~9월 여름)."""
    if ts.month >= 10:
        return f"{ts.year}-{str(ts.year + 1)[2:]}_winter"
    if ts.month <= 4:
        return f"{ts.year - 1}-{str(ts.year)[2:]}_winter"
    return f"{ts.year}_summer"


def load_model_sets(project_root: Path | None = None) -> dict[str, str]:
    """periods.yaml 의 model_sets 를 {시즌: train/dev/test} 로 돌려준다."""
    import yaml

    path = Path(project_root or ROOT) / "config" / "periods.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sets = cfg.get("model_sets")
    if not sets:
        raise KeyError(f"periods.yaml 에 model_sets 가 없습니다: {path}")
    mapping: dict[str, str] = {}
    for name in MODEL_SET_NAMES:
        for season in sets.get(name) or []:
            if season in mapping:
                raise ValueError(f"시즌 {season} 이 {mapping[season]} 와 {name} 양쪽에 있습니다.")
            mapping[str(season)] = name
    return mapping


def add_model_set(
    frame: pd.DataFrame,
    season_col: str = "season",
    include_test: bool = False,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """season 열을 보고 model_set 열(train / dev / test)을 붙인 사본을 돌려준다.

    - season 열이 없으면 date · datetime · fcst 열로 시즌을 정한다.
    - model_sets 에 없는 시즌이 있으면 오류로 멈춘다.
    - include_test=False(기본)이면 test 행을 뺀다. 마지막 평가 때만 True 로 연다.
    """
    mapping = load_model_sets(project_root)
    result = frame.copy()
    if season_col not in result.columns:
        # season 열이 없는 표(예보 · AWS · 수위)는 날짜로 시즌을 정한다 (10~4월 겨울, 5~9월 여름)
        date_col = next((c for c in ("date", "datetime", "fcst") if c in result.columns), None)
        if date_col is None:
            raise KeyError(f"{season_col} 열도, 날짜 열(date · datetime · fcst)도 없습니다.")
        result[season_col] = pd.to_datetime(result[date_col]).map(_season_of_date)
    keys = result[season_col].map(_season_key)
    result["model_set"] = keys.map(mapping)
    unknown = sorted(set(keys[result["model_set"].isna() & result[season_col].notna()]))
    if unknown:
        raise ValueError(f"model_sets 에 없는 시즌입니다: {unknown}")
    if result[season_col].isna().any():
        raise ValueError(f"{season_col} 이 빈 행이 {int(result[season_col].isna().sum())}개 있습니다.")
    if not include_test:
        result = result.loc[result["model_set"] != "test"].reset_index(drop=True)
    return result


def load_processed(
    name: str,
    include_test: bool = False,
    season_col: str = "season",
    project_root: Path | None = None,
    **read_kwargs,
) -> pd.DataFrame:
    """data/processed/{train,eval}/{name}_{split}.csv|parquet 을 읽어 model_set 열을 붙인다.

    include_test=False(기본)이면 eval/ 폴더(Test)는 아예 읽지 않는다.
    예) load_processed("weather_hourly"), load_processed("road_congestion_length", include_test=True)
    """
    base = Path(project_root or ROOT) / "data" / "processed"
    parts = []
    for split in ("train", "eval") if include_test else ("train",):
        stem = base / split / f"{name}_{split}"
        if stem.with_suffix(".parquet").is_file():
            parts.append(pd.read_parquet(stem.with_suffix(".parquet"), **read_kwargs))
        elif stem.with_suffix(".csv").is_file():
            parts.append(pd.read_csv(stem.with_suffix(".csv"), encoding="utf-8-sig", **read_kwargs))
        else:
            raise FileNotFoundError(f"{stem}.parquet 또는 .csv 가 없습니다.")
    frame = pd.concat(parts, ignore_index=True)
    return add_model_set(frame, season_col=season_col, include_test=include_test, project_root=project_root)
