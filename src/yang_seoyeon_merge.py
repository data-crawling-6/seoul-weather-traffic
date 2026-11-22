"""양서연 담당 TOPIS 링크 기본표와 도로 통행시간표 생성.

원본을 수정하지 않는다. 링크 기본표는 전 시간·전 날짜를 보존하고,
도로 분석표에서 연구기간의 평일 출퇴근 시간과 공휴일 제외 규칙을
적용한다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import holidays
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.io_utils import HOUR_COLS, read_topis, read_topis_gaps


KST = ZoneInfo("Asia/Seoul")
TARGET_FUNCTION_TYPES = {"도시고속도로", "주간선도로"}
WEATHER_COLUMNS = [
    "date", "hour", "temp_c", "precip_mm", "new_snow_3h_cm",
    "snow_depth_cm", "wind_ms", "humidity", "visibility_10m",
    "phenomenon", "precip_type", "event_id", "event_intensity", "impact",
    "missing_obs", "warn_snow", "warn_rain", "warn_cold",
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


def load_periods(project_root: Path) -> tuple[list[dict[str, object]], dict[int, str]]:
    path = project_root / "config" / "periods.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    periods = config["periods"]
    for period in periods:
        for key in ["start", "end", "baseline_start"]:
            period[key] = pd.Timestamp(period[key])
    commute = {
        int(hour): label
        for label, hours in config["commute_hours_end"].items()
        for hour in hours
    }
    return periods, commute


def load_weather(weather_dir: Path) -> pd.DataFrame:
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
    weather["date"] = pd.to_datetime(weather["date"], errors="raise").dt.normalize()
    weather["hour"] = pd.to_numeric(weather["hour"], errors="raise").astype("int8")
    if weather.duplicated(["date", "hour"]).any():
        raise ValueError("날씨 date+hour 키가 중복됩니다.")
    return weather[[column for column in WEATHER_COLUMNS if column in weather.columns]]


def load_roads(project_root: Path) -> pd.DataFrame:
    roads = pd.read_csv(project_root / "config" / "roads.csv", encoding="utf-8-sig", dtype="string")
    roads = roads.loc[roads["selection_status"].eq("selected")].copy()
    if len(roads) != 90 or roads["road_id"].duplicated().any() or roads["road_name"].duplicated().any():
        raise ValueError("분석 도로는 ID·이름 중복 없는 90개여야 합니다.")
    return roads


def find_stable_links(topis_paths: list[Path], roads: pd.DataFrame) -> tuple[set[str], pd.DataFrame]:
    """44개월 모두 존재하며 기준월 90개 도로에 속한 링크를 찾는다."""
    link_sets: list[set[str]] = []
    for path in topis_paths:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=["링크아이디"],
            dtype="string",
            low_memory=False,
        )
        link_sets.append(set(frame["링크아이디"].dropna().str.strip()))
    stable_all = set.intersection(*link_sets)

    reference_path = next(path for path in topis_paths if path.stem.endswith("2025-12"))
    reference = pd.read_csv(
        reference_path,
        encoding="utf-8-sig",
        usecols=["도로명", "링크아이디", "방향", "기능유형구분", "거리"],
        dtype="string",
        low_memory=False,
    ).drop_duplicates("링크아이디")
    reference = reference.loc[
        reference["도로명"].isin(roads["road_name"])
        & reference["기능유형구분"].isin(TARGET_FUNCTION_TYPES)
    ].copy()
    reference = reference.merge(
        roads[["road_id", "road_name"]],
        left_on="도로명",
        right_on="road_name",
        how="inner",
        validate="many_to_one",
    )
    reference["link_id"] = reference["링크아이디"].str.strip()
    selected = stable_all & set(reference["link_id"])
    metadata = reference.loc[reference["link_id"].isin(selected), [
        "link_id", "road_id", "road_name", "기능유형구분", "방향", "거리"
    ]].rename(columns={
        "기능유형구분": "function_type",
        "방향": "direction",
        "거리": "reference_distance_m",
    })
    metadata["reference_distance_m"] = pd.to_numeric(metadata["reference_distance_m"], errors="coerce")
    if metadata["link_id"].duplicated().any():
        raise ValueError("기준월 링크 메타데이터가 중복됩니다.")
    return selected, metadata


def _assign_period_scope(frame: pd.DataFrame, periods: list[dict[str, object]]) -> pd.DataFrame:
    result = frame.copy()
    result["season"] = pd.NA
    result["split"] = pd.NA
    result["period_scope"] = pd.NA
    for period in periods:
        baseline = result["date"].between(period["baseline_start"], period["start"] - pd.Timedelta(days=1))
        analysis = result["date"].between(period["start"], period["end"])
        result.loc[baseline, ["season", "split", "period_scope"]] = [
            period["season"], period["split"], "baseline_buffer"
        ]
        result.loc[analysis, ["season", "split", "period_scope"]] = [
            period["season"], period["split"], "in_period"
        ]
    return result.loc[result["period_scope"].notna()].copy()


def process_link_month(
    path: Path,
    data_root: Path,
    stable_links: set[str],
    metadata: pd.DataFrame,
    weather: pd.DataFrame,
    periods: list[dict[str, object]],
    gap_datetimes: set[pd.Timestamp],
    korean_holidays: holidays.HolidayBase,
) -> tuple[pd.DataFrame, dict[str, object]]:
    month = path.stem[-7:]
    raw_all = read_topis(month, data_root)
    raw_all["link_id"] = raw_all["링크아이디"].astype("string").str.strip()
    raw_all["source_date"] = pd.to_datetime(
        raw_all["일자"].astype("string"), format="%Y%m%d", errors="raise"
    )
    candidate = pd.Series(False, index=raw_all.index)
    for period in periods:
        candidate |= raw_all["source_date"].between(
            period["baseline_start"] - pd.Timedelta(days=1), period["end"]
        )
    raw_all = raw_all.loc[candidate].copy()
    source_dates = raw_all[["source_date"]].drop_duplicates()
    hour_grid = pd.DataFrame({"hour_end": list(range(1, 25))})
    time_grid = source_dates.merge(hour_grid, how="cross")
    time_grid["topis_hour_label"] = time_grid["hour_end"].map(lambda value: f"~{value:02d}시")
    time_grid["datetime"] = time_grid["source_date"] + pd.to_timedelta(
        time_grid["hour_end"].eq(24).astype("int8"), unit="D"
    ) + pd.to_timedelta(time_grid["hour_end"].mod(24), unit="h")
    time_grid["date"] = time_grid["datetime"].dt.normalize()
    time_grid["hour"] = time_grid["datetime"].dt.hour.astype("int8")
    time_grid = _assign_period_scope(time_grid, periods)

    raw = raw_all.loc[raw_all["link_id"].isin(stable_links)].copy()
    long = raw.melt(
        id_vars=["link_id", "source_date", "거리"],
        value_vars=HOUR_COLS,
        var_name="topis_hour_label",
        value_name="speed_kmh",
    )
    long["hour_end"] = pd.to_numeric(
        long["topis_hour_label"].str.extract(r"(\d{2})")[0], errors="raise"
    ).astype("int8")
    long["datetime"] = long["source_date"] + pd.to_timedelta(
        long["hour_end"].eq(24).astype("int8"), unit="D"
    ) + pd.to_timedelta(long["hour_end"].mod(24), unit="h")
    long["distance_m"] = pd.to_numeric(long["거리"], errors="coerce")
    long["speed_kmh"] = pd.to_numeric(long["speed_kmh"], errors="coerce")
    long["row_present"] = True
    actual = long[["link_id", "datetime", "distance_m", "speed_kmh", "row_present"]]

    grid = time_grid.merge(metadata, how="cross")
    grid = grid.merge(actual, on=["link_id", "datetime"], how="left", validate="one_to_one")
    grid["distance_m"] = pd.to_numeric(
        grid["distance_m"].fillna(grid["reference_distance_m"]), errors="coerce"
    ).astype("float64")
    grid["in_period"] = grid["period_scope"].eq("in_period")
    grid["is_baseline_buffer"] = grid["period_scope"].eq("baseline_buffer")
    grid["is_weekday"] = grid["date"].dt.day_of_week.lt(5)
    grid["is_holiday"] = grid["date"].dt.date.isin(korean_holidays)
    grid["is_summer_vacation_peak"] = pd.Series(pd.NA, index=grid.index, dtype="boolean")
    grid["is_topis_system_gap"] = grid["datetime"].isin(gap_datetimes)
    grid["missing_reason"] = ""
    grid.loc[grid["row_present"].isna(), "missing_reason"] = "link_row_absent"
    grid.loc[grid["row_present"].notna() & grid["speed_kmh"].isna(), "missing_reason"] = "speed_missing"
    grid.loc[grid["speed_kmh"].le(0), "missing_reason"] = "nonpositive_speed"
    grid.loc[grid["is_topis_system_gap"], "missing_reason"] = "topis_system_gap"
    valid = grid["speed_kmh"].gt(0) & grid["distance_m"].ge(0) & ~grid["is_topis_system_gap"]
    grid["link_travel_time_seconds"] = pd.NA
    grid.loc[valid, "link_travel_time_seconds"] = (
        grid.loc[valid, "distance_m"] / (grid.loc[valid, "speed_kmh"] * 1000 / 3600)
    )
    grid["link_travel_time_seconds"] = pd.to_numeric(
        grid["link_travel_time_seconds"], errors="coerce"
    )
    grid = grid.merge(weather, on=["date", "hour"], how="left", validate="many_to_one")
    grid["source_month"] = month
    columns = [
        "link_id", "road_id", "road_name", "function_type", "direction",
        "date", "hour", "datetime", "season", "split", "period_scope",
        "in_period", "is_baseline_buffer", "is_weekday", "is_holiday",
        "is_summer_vacation_peak", "topis_hour_label", "distance_m", "speed_kmh",
        "link_travel_time_seconds", "is_topis_system_gap", "missing_reason",
        *[column for column in WEATHER_COLUMNS if column not in {"date", "hour"}],
        "source_month",
    ]
    result = grid[columns].copy()
    stats = {
        "source_file": path.name,
        "link_rows": int(len(result)),
        "stable_links_present": int(result["link_id"].nunique()),
        "weather_missing_rows": int(result["temp_c"].isna().sum()),
        "system_gap_rows": int(result["is_topis_system_gap"].sum()),
    }
    return result, stats


def aggregate_road_commute(link_frame: pd.DataFrame, commute_hours: dict[int, str]) -> pd.DataFrame:
    selected = link_frame.loc[
        link_frame["in_period"]
        & link_frame["is_weekday"]
        & ~link_frame["is_holiday"]
        & link_frame["hour"].isin(commute_hours)
    ].copy()
    valid = selected["link_travel_time_seconds"].notna()
    selected["observed_distance_m"] = selected["distance_m"].where(valid)
    selected["observed_link"] = valid.astype("int16")
    keys = [
        "date", "hour", "season", "split", "road_id", "road_name",
        "function_type", "direction",
    ]
    grouped = selected.groupby(keys, dropna=False, observed=True).agg(
        link_count=("link_id", "nunique"),
        observed_link_count=("observed_link", "sum"),
        total_distance_m=("distance_m", "sum"),
        observed_distance_m=("observed_distance_m", lambda values: values.sum(min_count=1)),
        travel_time_seconds=("link_travel_time_seconds", lambda values: values.sum(min_count=1)),
    ).reset_index()
    grouped["missing_speed_links"] = grouped["link_count"] - grouped["observed_link_count"]
    grouped["road_speed_kmh"] = grouped["observed_distance_m"] / grouped["travel_time_seconds"] * 3.6
    # 링크가 일부라도 빠진 행은 관측 링크만 합산되어 통행시간이 과소 추정된다.
    # 거리 비례 보정은 누락 링크의 속도가 같다는 가정을 만들므로 사용하지 않는다.
    incomplete = grouped["observed_link_count"] < grouped["link_count"]
    grouped.loc[incomplete, ["travel_time_seconds", "road_speed_kmh"]] = pd.NA
    grouped["commute_period"] = grouped["hour"].map(commute_hours)
    weather_columns = [column for column in WEATHER_COLUMNS if column not in {"date", "hour"}]
    weather_once = selected[["date", "hour", *weather_columns]].drop_duplicates(["date", "hour"])
    grouped = grouped.merge(weather_once, on=["date", "hour"], how="left", validate="many_to_one")
    return grouped


def convert_v03_to_v04(
    source_path: Path,
    output_path: Path,
    sample_path: Path,
    log_path: Path,
) -> dict[str, object]:
    """기존 도로표에서 링크 일부 누락 행을 NaN 처리해 v04를 재현한다.

    원본 v03는 수정하지 않으며 출력 파일이 이미 있으면 중단한다. 샘플은
    정상 관측 100행과 결측 처리 사례 100행으로 구성한다.
    """
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    sample_path = Path(sample_path).resolve()
    log_path = Path(log_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for path in (output_path, sample_path, log_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    road = pd.read_parquet(source_path)
    incomplete = road["observed_link_count"] < road["link_count"]
    partial = incomplete & road["observed_link_count"].gt(0)
    v03_nan_rows = int(road["travel_time_seconds"].isna().sum())
    road.loc[incomplete, ["travel_time_seconds", "road_speed_kmh"]] = pd.NA
    road.to_parquet(output_path, index=False, compression="snappy")

    sample = pd.concat(
        [road.loc[~incomplete].head(100), road.loc[incomplete].head(100)],
        ignore_index=True,
    )
    sample.to_csv(sample_path, encoding="utf-8-sig", index=False)
    v04_nan_rows = int(road["travel_time_seconds"].isna().sum())
    nan_ratio = v04_nan_rows / len(road)
    log = {
        "run_at": datetime.now(KST).isoformat(),
        "method_version": "v3_incomplete_links_as_nan",
        "source_v03_path": str(source_path),
        "output_v04_path": str(output_path),
        "output_rows": int(len(road)),
        "duplicate_keys": int(
            road.duplicated(["date", "hour", "road_id", "direction"]).sum()
        ),
        "v03_travel_time_nan_rows": v03_nan_rows,
        "v04_travel_time_nan_rows": v04_nan_rows,
        "newly_nan_rows": v04_nan_rows - v03_nan_rows,
        "partial_link_rows": int(partial.sum()),
        "all_links_missing_rows": int(road["observed_link_count"].eq(0).sum()),
        "incomplete_link_rows": int(incomplete.sum()),
        "v04_nan_ratio": nan_ratio,
        "review_threshold_ratio": 0.01,
        "exceeds_review_threshold": bool(nan_ratio >= 0.01),
        "wording": "링크 일부 누락 시 관측 링크만 합산되어 통행시간이 과소 추정되므로 NaN 처리",
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
        "sample_path": str(sample_path),
        "sample_composition": "정상 관측 100행 + 링크 결측 NaN 사례 100행",
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**log, "log_path": str(log_path)}


def run_merge(data_root: Path, project_root: Path) -> dict[str, object]:
    data_root = Path(data_root).resolve()
    project_root = Path(project_root).resolve()
    topis_paths = sorted((data_root / "topis_speed" / "csv").glob("topis_speed_*.csv"))
    if len(topis_paths) != 44:
        raise RuntimeError(f"TOPIS 월 파일은 44개여야 합니다: 현재 {len(topis_paths)}개")
    periods, commute_hours = load_periods(project_root)
    weather = load_weather(data_root / "weather" / "processed")
    roads = load_roads(project_root)
    gaps = read_topis_gaps(project_root)
    gap_datetimes = set(gaps["datetime"])
    korean_holidays = holidays.KR(years=range(2022, 2027))
    stable_links, metadata = find_stable_links(topis_paths, roads)
    if not stable_links:
        raise RuntimeError("44개월 공통 링크가 없습니다.")

    run_date = datetime.now(KST).strftime("%Y%m%d")
    processed_dir = data_root / "processed"
    link_path = _next_version(processed_dir, f"yang_seoyeon_topis_weather_link_hourly_{run_date}", ".parquet")
    road_path = _next_version(processed_dir, f"yang_seoyeon_topis_weather_commute_{run_date}", ".parquet")
    writer: pq.ParquetWriter | None = None
    road_parts: list[pd.DataFrame] = []
    monthly_stats: list[dict[str, object]] = []
    try:
        for index, path in enumerate(topis_paths, start=1):
            link_part, stats = process_link_month(
                path, data_root, stable_links, metadata, weather, periods,
                gap_datetimes, korean_holidays,
            )
            table = pa.Table.from_pandas(link_part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(link_path, table.schema, compression="snappy")
            writer.write_table(table)
            road_parts.append(aggregate_road_commute(link_part, commute_hours))
            stats["road_rows"] = int(len(road_parts[-1]))
            monthly_stats.append(stats)
            print(
                f"[결합 {index:02d}/{len(topis_paths)}] {path.name}: "
                f"링크 {len(link_part):,}행 / 도로 {len(road_parts[-1]):,}행",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    road = pd.concat(road_parts, ignore_index=True).sort_values(
        ["date", "hour", "road_id", "direction"], kind="stable"
    ).reset_index(drop=True)
    road.to_parquet(road_path, index=False, compression="snappy")

    summary_dir = project_root / "outputs" / "tables"
    sample_dir = project_root / "data" / "sample"
    log_dir = project_root / "logs"
    summary_path = _next_version(summary_dir, f"yang_seoyeon_topis_weather_merge_summary_{run_date}", ".csv")
    sample_path = _next_version(sample_dir, f"yang_seoyeon_topis_weather_commute_sample_{run_date}", ".csv")
    link_sample_path = _next_version(sample_dir, f"yang_seoyeon_topis_weather_link_sample_{run_date}", ".csv")
    log_path = _next_version(log_dir, f"yang_seoyeon_topis_weather_merge_{run_date}", ".json")
    stable_path = _next_version(summary_dir, f"yang_seoyeon_stable_links_{run_date}", ".csv")
    pd.DataFrame(monthly_stats).to_csv(summary_path, encoding="utf-8-sig", index=False)
    road.head(200).to_csv(sample_path, encoding="utf-8-sig", index=False)
    pq.ParquetFile(link_path).read_row_group(0).to_pandas().head(200).to_csv(
        link_sample_path, encoding="utf-8-sig", index=False
    )
    metadata.sort_values(["road_id", "direction", "link_id"]).to_csv(stable_path, encoding="utf-8-sig", index=False)

    unique_weather_hours = road[["date", "hour", "split"]].drop_duplicates()
    log = {
        "run_at": datetime.now(KST).isoformat(),
        "method_version": "v3_incomplete_links_as_nan",
        "topis_file_count": len(topis_paths),
        "stable_link_count": len(stable_links),
        "stable_road_count": int(metadata["road_id"].nunique()),
        "link_output_rows": int(sum(item["link_rows"] for item in monthly_stats)),
        "road_output_rows": int(len(road)),
        "weather_hour_count": int(len(unique_weather_hours)),
        "weather_hours_by_split": {
            str(key): int(value)
            for key, value in unique_weather_hours.groupby("split").size().items()
        },
        "road_duplicate_keys": int(road.duplicated(["date", "hour", "road_id", "direction"]).sum()),
        "road_weather_missing_rows": int(road["temp_c"].isna().sum()),
        "road_travel_time_missing_rows": int(road["travel_time_seconds"].isna().sum()),
        "road_partial_link_rows": int(
            ((road["observed_link_count"] < road["link_count"]) & (road["observed_link_count"] > 0)).sum()
        ),
        "road_incomplete_link_rows_as_nan": int(
            (road["observed_link_count"] < road["link_count"]).sum()
        ),
        "topis_system_gap_count": int(len(gaps)),
        "link_path": str(link_path),
        "link_bytes": link_path.stat().st_size,
        "link_sha256": _sha256(link_path),
        "road_path": str(road_path),
        "road_bytes": road_path.stat().st_size,
        "road_sha256": _sha256(road_path),
        "summary_path": str(summary_path),
        "road_sample_path": str(sample_path),
        "link_sample_path": str(link_sample_path),
        "stable_links_path": str(stable_path),
        "limitations": [
            "겨울 baseline_start 10/30~10/31 TOPIS 원자료가 Release에 없어 각 겨울 2일이 비어 있음",
            "여름 휴가 절정 주간의 날짜 정의가 미확정이라 is_summer_vacation_peak는 NA",
        ],
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    log["log_path"] = str(log_path)
    return log


if __name__ == "__main__":
    raise SystemExit("노트북 또는 명시적 함수 호출로 run_merge()를 실행하세요.")
