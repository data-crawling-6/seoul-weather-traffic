"""양서연 담당 TOPIS 통행속도와 시간별 날씨 결합.

원본을 수정하지 않고 월별 TOPIS를 순차 처리한 뒤 도로×방향×시간
단위 통행시간 표를 만든다. 2024-02 이전의 ``08시``와 이후의
``~08시`` 표기를 모두 지원한다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
TARGET_FUNCTION_TYPES = {"도시고속도로", "주간선도로"}
COMMUTE_HOURS = {8: "morning", 9: "morning", 18: "evening", 19: "evening", 20: "evening"}
ID_COLUMNS = [
    "일자", "도로명", "링크아이디", "방향", "거리", "기능유형구분",
]
WEATHER_COLUMNS = [
    "date", "hour", "temp_c", "precip_mm", "new_snow_3h_cm",
    "snow_depth_cm", "wind_ms", "humidity", "visibility_10m",
    "precip_type", "event_id", "event_intensity", "impact",
    "missing_obs", "warn_snow", "warn_rain", "warn_cold",
]
STUDY_PERIODS = [
    ("2022-23_winter", "train", "2022-11-27", "2023-03-10"),
    ("2023_summer", "train", "2023-05-09", "2023-09-23"),
    ("2023-24_winter", "train", "2023-11-27", "2024-03-10"),
    ("2024_summer", "train", "2024-05-09", "2024-09-23"),
    ("2024-25_winter", "train", "2024-11-27", "2025-03-10"),
    ("2025_summer", "train", "2025-05-09", "2025-09-23"),
    ("2025-26_winter", "eval", "2025-11-27", "2026-03-10"),
    ("2026_summer", "eval", "2026-05-09", "2026-09-23"),
]


def _next_version(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        candidate = directory / f"{stem}_v{version:02d}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_period(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["season"] = pd.NA
    result["split"] = pd.NA
    for season, split, start, end in STUDY_PERIODS:
        mask = result["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        result.loc[mask, ["season", "split"]] = [season, split]
    return result.loc[result["season"].notna()].copy()


def read_weather(weather_dir: Path) -> pd.DataFrame:
    paths = [
        weather_dir / "train" / "weather_hourly_train.csv",
        weather_dir / "eval" / "weather_hourly_eval.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"날씨 파일이 없습니다: {missing}")
    weather = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig", low_memory=False) for path in paths],
        ignore_index=True,
    )
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
    weather["hour"] = pd.to_numeric(weather["hour"], errors="coerce").astype("Int64")
    weather = weather.loc[
        weather["in_period"].astype("boolean").eq(True)
        & weather["hour"].isin(COMMUTE_HOURS)
        & weather["date"].dt.dayofweek.lt(5)
    ].copy()
    if weather.duplicated(["date", "hour"]).any():
        raise ValueError("날씨 date+hour 키가 중복됩니다.")
    available = [column for column in WEATHER_COLUMNS if column in weather.columns]
    return weather[available]


def process_topis_month(path: Path, weather: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    hour_lookup: dict[str, int] = {}
    for hour in COMMUTE_HOURS:
        candidates = [f"~{hour:02d}시", f"{hour:02d}시"]
        matches = [column for column in candidates if column in header]
        if len(matches) != 1:
            raise ValueError(f"{path.name}: {hour}시 열을 하나로 확정할 수 없습니다: {matches}")
        hour_lookup[matches[0]] = hour
    required = set(ID_COLUMNS) | set(hour_lookup)
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{path.name}: 필수 열이 없습니다: {missing}")

    raw = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=ID_COLUMNS + list(hour_lookup),
        dtype={"링크아이디": "string"},
        low_memory=False,
    )
    raw["date"] = pd.to_datetime(raw["일자"].astype("string"), format="%Y%m%d", errors="coerce")
    raw = _add_period(raw)
    raw = raw.loc[
        raw["date"].dt.dayofweek.lt(5)
        & raw["기능유형구분"].isin(TARGET_FUNCTION_TYPES)
    ].copy()
    long = raw.melt(
        id_vars=ID_COLUMNS + ["date", "season", "split"],
        value_vars=list(hour_lookup),
        var_name="topis_hour_label",
        value_name="speed_kmh",
    )
    long["hour"] = long["topis_hour_label"].map(hour_lookup).astype("Int64")
    long["distance_m"] = pd.to_numeric(long["거리"], errors="coerce")
    long["speed_kmh"] = pd.to_numeric(long["speed_kmh"], errors="coerce")
    valid = long["speed_kmh"].gt(0) & long["distance_m"].ge(0)
    long["link_travel_time_seconds"] = pd.NA
    long.loc[valid, "link_travel_time_seconds"] = (
        long.loc[valid, "distance_m"]
        / (long.loc[valid, "speed_kmh"] * 1000 / 3600)
    )
    long["observed_distance_m"] = long["distance_m"].where(valid)
    long["observed_link"] = valid.astype("int16")

    keys = ["date", "hour", "season", "split", "도로명", "기능유형구분", "방향"]
    grouped = long.groupby(keys, dropna=False, observed=True).agg(
        link_count=("링크아이디", "nunique"),
        observed_link_count=("observed_link", "sum"),
        total_distance_m=("distance_m", "sum"),
        observed_distance_m=("observed_distance_m", lambda values: values.sum(min_count=1)),
        travel_time_seconds=("link_travel_time_seconds", lambda values: values.sum(min_count=1)),
    ).reset_index()
    grouped["missing_speed_links"] = grouped["link_count"] - grouped["observed_link_count"]
    grouped["road_speed_kmh"] = (
        grouped["observed_distance_m"] / grouped["travel_time_seconds"] * 3.6
    )
    grouped["commute_period"] = grouped["hour"].map(COMMUTE_HOURS)
    grouped = grouped.rename(columns={
        "도로명": "road_name",
        "기능유형구분": "function_type",
        "방향": "direction",
    })
    merged = grouped.merge(weather, on=["date", "hour"], how="left", validate="many_to_one")
    merged["source_month"] = path.stem[-7:]

    stats = {
        "source_file": path.name,
        "source_rows": int(len(raw)),
        "output_rows": int(len(merged)),
        "road_count": int(merged["road_name"].nunique()),
        "missing_weather_rows": int(merged["temp_c"].isna().sum()),
        "all_speed_missing_rows": int(merged["travel_time_seconds"].isna().sum()),
    }
    return merged, stats


def run_merge(data_root: Path, project_root: Path) -> dict[str, object]:
    data_root = Path(data_root).resolve()
    project_root = Path(project_root).resolve()
    topis_paths = sorted((data_root / "topis_speed" / "csv").glob("topis_speed_*.csv"))
    if len(topis_paths) != 40:
        raise RuntimeError(f"TOPIS 월 파일은 40개여야 합니다: 현재 {len(topis_paths)}개")
    weather = read_weather(data_root / "weather" / "processed")

    parts: list[pd.DataFrame] = []
    monthly_stats: list[dict[str, object]] = []
    for index, path in enumerate(topis_paths, start=1):
        part, stats = process_topis_month(path, weather)
        parts.append(part)
        monthly_stats.append(stats)
        print(f"[결합 {index:02d}/40] {path.name}: {len(part):,}행, {stats['road_count']}개 도로")

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(
        ["date", "hour", "function_type", "road_name", "direction"],
        kind="stable",
    ).reset_index(drop=True)
    if combined.duplicated(["date", "hour", "road_name", "direction"]).any():
        raise ValueError("최종 date+hour+road_name+direction 키가 중복됩니다.")

    run_date = datetime.now(KST).strftime("%Y%m%d")
    output_path = _next_version(
        data_root / "processed",
        f"yang_seoyeon_topis_weather_commute_{run_date}",
        ".parquet",
    )
    combined.to_parquet(output_path, index=False, compression="snappy")

    summary_dir = project_root / "outputs" / "tables"
    sample_dir = project_root / "data" / "sample"
    log_dir = project_root / "logs"
    summary_path = _next_version(
        summary_dir, f"yang_seoyeon_topis_weather_merge_summary_{run_date}", ".csv"
    )
    sample_path = _next_version(
        sample_dir, f"yang_seoyeon_topis_weather_commute_sample_{run_date}", ".csv"
    )
    log_path = _next_version(
        log_dir, f"yang_seoyeon_topis_weather_merge_{run_date}", ".json"
    )
    pd.DataFrame(monthly_stats).to_csv(summary_path, encoding="utf-8-sig", index=False)
    sample_columns = [
        "date", "hour", "commute_period", "season", "split", "road_name",
        "function_type", "direction", "link_count", "observed_link_count",
        "travel_time_seconds", "road_speed_kmh", "temp_c", "precip_mm",
        "new_snow_3h_cm", "impact",
    ]
    combined[sample_columns].head(200).to_csv(sample_path, encoding="utf-8-sig", index=False)

    log = {
        "run_at": datetime.now(KST).isoformat(),
        "python_operation": "TOPIS+weather road-direction commute merge",
        "topis_file_count": len(topis_paths),
        "weather_rows_used": int(len(weather)),
        "output_rows": int(len(combined)),
        "date_min": combined["date"].min().isoformat(),
        "date_max": combined["date"].max().isoformat(),
        "unique_roads": int(combined["road_name"].nunique()),
        "duplicate_keys": int(combined.duplicated(["date", "hour", "road_name", "direction"]).sum()),
        "missing_weather_rows": int(combined["temp_c"].isna().sum()),
        "all_speed_missing_rows": int(combined["travel_time_seconds"].isna().sum()),
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
        "summary_path": str(summary_path),
        "sample_path": str(sample_path),
        "rules": {
            "weekdays_only": True,
            "commute_hours_end": sorted(COMMUTE_HOURS),
            "function_types": sorted(TARGET_FUNCTION_TYPES),
            "time_definition": "~08 means 07:00-08:00; joined to weather hour 8",
            "travel_time": "sum(distance_m / speed_kmh converted to seconds)",
            "missing_speed": "preserved as NaN; not converted to zero",
        },
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


if __name__ == "__main__":
    raise SystemExit("노트북 또는 명시적 함수 호출로 run_merge()를 실행하세요.")
