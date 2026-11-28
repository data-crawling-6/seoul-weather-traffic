"""C 담당 TOPIS 링크 기본표와 도로 통행시간표 생성.

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

import sys  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # python src/join/topis_weather_merge.py 로 바로 실행해도 src 를 찾게
    sys.path.insert(0, str(_ROOT))
from src.io_utils import HOUR_COLS, data_root as _data_root, derived_root, read_topis, read_topis_gaps, topis_csv_files


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


EXTRA_TOPIS_MONTHS = {"2022-07", "2022-08"}  # 범위 밖 사례(8/8 강남 침수)용, 결합에서 제외


def weather_dir(project_root: Path) -> Path:
    """B 날씨 표 위치: 저장소 data/shared/ (D 드라이브 복사본 대신, B 수정)."""
    return Path(project_root) / "data" / "shared"


def topis_csv(data_root: Path, month: str) -> Path:
    """TOPIS 월 CSV: data/raw/topis_speed/ (A TOPIS 원본, 팀 공유 드라이브에서 받아 푼 곳)."""
    folder = Path(data_root) / "topis_speed"
    old = folder / f"topis_speed_{month}.csv"  # zip 안 이름으로 풀어 둔 경우
    return old if old.is_file() and not (folder / f"{month}.csv").is_file() else folder / f"{month}.csv"


def load_weather(weather_dir: Path) -> pd.DataFrame:
    paths = [
        weather_dir / "train_dev" / "weather_hourly_train_dev.csv",
        weather_dir / "holdout" / "weather_hourly_holdout.csv",
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


def expected_source_dates(month: str, periods: list[dict[str, object]]) -> pd.DatetimeIndex:
    """그 달 달력 날짜 가운데 연구 기간 · 평소값 기간(baseline_start 하루 전 ~ end)에 드는 날 (코드 검토 11번).

    원본에 나온 날짜만으로 시간 격자를 만들면, 날짜가 통째로 빠진 날은 격자에도 없어 아무도 모른다.
    그래서 달력으로 '있어야 할 날짜'를 먼저 만들고 원본 날짜와 비교한다.
    """
    start = pd.Timestamp(f"{month}-01")
    days = pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D")
    keep = pd.Series(False, index=days)
    for period in periods:
        keep |= (days >= pd.Timestamp(period["baseline_start"]) - pd.Timedelta(days=1)) & (days <= pd.Timestamp(period["end"]))
    return days[keep.to_numpy()]


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
    present = pd.DatetimeIndex(raw_all["source_date"].drop_duplicates())
    expected = expected_source_dates(month, periods)
    missing_dates = expected.difference(present)
    # 빠진 날짜도 격자에 넣어 link_row_absent 로 남긴다 (그 날 전체가 결측으로 보이게)
    source_dates = pd.DataFrame({"source_date": present.union(expected).sort_values()})
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
        "expected_source_dates": int(len(expected)),
        "missing_source_dates": [d.strftime("%Y-%m-%d") for d in missing_dates],
    }
    if len(missing_dates):
        print(f"⚠️ {month}: 원본에 없는 날짜 {len(missing_dates)}일 → 격자에 결측으로 남김 ({', '.join(stats['missing_source_dates'][:5])} …)")
    return result, stats


def aggregate_road_commute(link_frame: pd.DataFrame, commute_hours: dict[int, str], *, include_baseline: bool = False) -> pd.DataFrame:
    selected = link_frame.loc[
        (link_frame["in_period"] | (include_baseline & link_frame["is_baseline_buffer"]))
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
    # A TOPIS 원본은 50개월: v04를 만든 44개월 + 겨울 평소값용 10월 4개(v05에서 따로 추가)
    # + 범위 밖 사례용 2022-07 · 08. 결합은 처음과 같은 44개월만 쓴다.
    topis_paths = [path for path in topis_csv_files(data_root / "topis_speed")
                   if path.stem[-2:] != "10" and path.stem[-7:] not in EXTRA_TOPIS_MONTHS]
    if len(topis_paths) != 44:
        raise RuntimeError(f"TOPIS 월 파일은 44개여야 합니다: 현재 {len(topis_paths)}개")
    periods, commute_hours = load_periods(project_root)
    weather = load_weather(weather_dir(project_root))
    roads = load_roads(project_root)
    gaps = read_topis_gaps(project_root)
    gap_datetimes = set(gaps["datetime"])
    korean_holidays = holidays.KR(years=range(2022, 2027))
    stable_links, metadata = find_stable_links(topis_paths, roads)
    if not stable_links:
        raise RuntimeError("44개월 공통 링크가 없습니다.")

    run_date = datetime.now(KST).strftime("%Y%m%d")
    processed_dir = derived_root() / "topis_weather"  # 결합표는 data/derived/topis_weather/
    link_path = _next_version(processed_dir, f"link_hourly_{run_date}", ".parquet")
    road_path = _next_version(processed_dir, f"commute_{run_date}", ".parquet")
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
    log_dir = project_root / "logs" / "ysy" / "topis_weather_merge"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _next_version(summary_dir, f"topis_weather_merge_summary_{run_date}", ".csv")
    sample_path = _next_version(sample_dir, f"topis_weather_commute_sample_{run_date}", ".csv")
    link_sample_path = _next_version(sample_dir, f"topis_weather_link_sample_{run_date}", ".csv")
    log_path = _next_version(log_dir, f"topis_weather_merge_{run_date}", ".json")
    stable_path = _next_version(summary_dir, f"stable_links_{run_date}", ".csv")
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
            "겨울 baseline_start 10/30~10/31 TOPIS 원자료가 받은 원본에 없어 각 겨울 2일이 비어 있음",
            "여름 휴가 절정 주간의 날짜 정의가 미확정이라 is_summer_vacation_peak는 NA",
        ],
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    log["log_path"] = str(log_path)
    return log


# [서연] 기존 v04의 링크 집합과 계산식을 유지한 10월 보완 버전.
def extend_v04_october(data_root: Path, project_root: Path, source_path: Path,
                       link_path: Path) -> dict[str, object]:
    data_root, project_root = Path(data_root), Path(project_root)
    source_path, link_path = Path(source_path), Path(link_path)
    output = source_path.with_name(source_path.name.replace("_v04.parquet", "_v05.parquet"))
    if output == source_path or output.exists():
        raise ValueError(f"v04 파일명 또는 v05 기존 파일 확인: {output}")
    old = pd.read_parquet(source_path)
    keys = ["date", "hour", "road_id", "direction"]
    if old.duplicated(keys).any():
        raise ValueError("기존 v04 키 중복")
    periods, commute = load_periods(project_root)
    weather = load_weather(weather_dir(project_root))
    gaps = set(read_topis_gaps(project_root)["datetime"])
    kr = holidays.KR(years=range(2022, 2027))
    # 기존 링크 기본표 첫 row group의 링크 ID를 사용하고 v04의 방향별 링크 수와 대조.
    pf = pq.ParquetFile(link_path)
    cols = ["link_id", "road_id", "road_name", "function_type", "direction"]
    meta = pf.read_row_group(0, columns=cols).to_pandas().drop_duplicates()
    if meta["link_id"].duplicated().any():
        raise ValueError("기존 링크 메타데이터 중복")
    meta["link_id"] = meta["link_id"].astype("string").str.strip()
    group = ["road_id", "road_name", "function_type", "direction"]
    expected = old[group + ["link_count"]].drop_duplicates()
    actual = meta.groupby(group, dropna=False).size().rename("link_count").reset_index()
    check = expected.merge(actual, on=group, how="outer", suffixes=("_old", "_new"), indicator=True)
    if not (check["_merge"].eq("both") & check["link_count_old"].eq(check["link_count_new"])).all():
        raise ValueError("기존 v04와 링크 구성 불일치: 기존 링크 기본표 확인 필요")
    ref = pd.read_csv(topis_csv(data_root, "2025-12"),
                      encoding="utf-8-sig", usecols=["링크아이디", "거리"], dtype="string").drop_duplicates("링크아이디")
    ref["link_id"] = ref["링크아이디"].str.strip()
    ref["reference_distance_m"] = pd.to_numeric(ref["거리"], errors="coerce")
    meta = meta.merge(ref[["link_id", "reference_distance_m"]], on="link_id", how="left", validate="one_to_one")
    if meta["reference_distance_m"].isna().any():
        raise ValueError("기준월 링크 거리 누락")
    parts, stats = [], []
    for year in range(2022, 2026):
        start = pd.Timestamp(year, 10, 1)
        end = pd.Timestamp(year, 10, 31)
        winter = [p for p in periods if p["start"].year == year and p["start"].month == 11]
        if len(winter) != 1:
            raise ValueError(f"{year} 겨울 기간 설정 확인 필요")
        scope = [{**winter[0], "baseline_start": start, "start": end + pd.Timedelta(days=1), "end": end + pd.Timedelta(days=1)}]
        path = topis_csv(data_root, f"{year}-10")
        if not path.is_file():
            raise FileNotFoundError(path)
        links, info = process_link_month(path, data_root, set(meta["link_id"]), meta, weather, scope, gaps, kr)
        links = links.loc[links["date"].between(start, end)].copy()
        road = aggregate_road_commute(links, commute, include_baseline=True)
        if road.empty:
            raise ValueError(f"{year} 10월 집계 결과 없음")
        info.update(road_rows=len(road), road_weather_missing_rows=int(road["temp_c"].isna().sum()),
                    road_travel_time_missing_rows=int(road["travel_time_seconds"].isna().sum()))
        stats.append(info)
        parts.append(road)
        print(f"[10월] {year}: {len(road):,}행", flush=True)
    added = pd.concat(parts, ignore_index=True)
    if set(added.columns) != set(old.columns):
        raise ValueError(f"v04 열 불일치: {set(added.columns) ^ set(old.columns)}")
    added = added[old.columns]
    result = pd.concat([old, added], ignore_index=True)
    if result.duplicated(keys).any():
        raise ValueError("결합 키 중복: 기존 데이터와 10월 중복 확인")
    pd.testing.assert_frame_equal(result.iloc[:len(old)].reset_index(drop=True), old.reset_index(drop=True), check_dtype=False)
    log_dir = project_root / "logs/ysy/topis_weather_merge"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(KST).strftime("%Y%m%dT%H%M%S%f")
    log_path = log_dir / f"traffic_october_v05_{stamp}.json"
    tmp = output.with_suffix(".tmp.parquet")
    if tmp.exists():
        raise FileExistsError(tmp)
    try:
        result.to_parquet(tmp, index=False, compression="snappy")
        reread = pd.read_parquet(tmp)
        pd.testing.assert_frame_equal(reread.iloc[:len(old)].reset_index(drop=True), old.reset_index(drop=True), check_dtype=False)
        if len(reread) != len(result) or reread.duplicated(keys).any():
            raise ValueError("저장 후 검증 실패")
        tmp.rename(output)
    finally:
        if tmp.exists():
            tmp.unlink()
    log = {"status": "성공", "source": str(source_path), "output": str(output),
           "old_rows": len(old), "added_rows": len(added), "output_rows": len(result),
           "stable_links": len(meta), "duplicate_keys": 0, "old_rows_preserved": True,
           "monthly_stats": stats, "output_sha256": _sha256(output),
           "limitations": "10월 평일 출퇴근만 추가. 연구 기간 변경 없음. 기존 v04에는 다른 baseline_buffer 날짜가 제외되어 있으므로 입력용 평소값 계산 전 별도 확인 필요. 10월 기상 결측은 보간하지 않음.",
           "log_path": str(log_path)}
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend-v04-october", action="store_true")
    parser.add_argument("--data-root", type=Path, default=_data_root())
    parser.add_argument("--source-v04", type=Path)
    parser.add_argument("--link-path", type=Path)
    args = parser.parse_args()
    if not args.extend_v04_october:
        parser.error("노트북 또는 명시적 함수 호출로 run_merge()를 실행하세요.")
    root = args.data_root
    result = extend_v04_october(root, Path(__file__).resolve().parents[2],
        args.source_v04 or derived_root() / "topis_weather/commute_20261122_v04.parquet",
        args.link_path or derived_root() / "topis_weather/link_hourly_20261122_v02.parquet")
    print(json.dumps(result, ensure_ascii=False, indent=2))
